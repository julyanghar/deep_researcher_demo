import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal, List
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv("keys.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def create_prompt(key_point, answer):

    return f"""You are given a **single key point** and a **report**.

    Your job is to determine whether the report:
    - **Supports** the key point (it affirms, explains, or reinforces the point),
    - **Omits** the key point (it does not mention or cover this point at all), or
    - **Contradicts** the key point (it says something that disagrees with or negates the point).

    Carefully read the key point and the report.

    Return your answer as a **JSON object** with two fields:
    - "label": One of "Supported", "Omitted", or "Contradicted".
    - "justification": Brief explanation on why you assigned this label.

    Respond strictly in JSON format:
    {{"label": label, "justification": justification}}
    Do **not** add any extra commentary or text outside the JSON.

    ---

    Key Point: {key_point}
    Report: {answer}
    """



class KeyPointRecall(BaseModel):
    label: Literal["Supported", "Omitted", "Contradicted"]
    justification: str


async def evaluate_single_key_point(semaphore, key_point, answer, model):
    async with semaphore:
        prompt = create_prompt(key_point["point_content"], answer)
        chat_pattern = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        # 429 退避重试(2026-07-26),同 eval_quality_async
        last = None
        for attempt in range(2):   # 日配额按请求计:重试会放大消耗,只留 1 次(2026-07-26)
            try:
                response = await client.beta.chat.completions.parse(
                            model=model,
                            messages=chat_pattern,
                            response_format=KeyPointRecall,
                            temperature=0
                )
                result = json.loads(response.choices[0].message.content)
                return key_point["point_number"], (result['label'], result['justification'])
            except Exception as e:                      # noqa: BLE001
                last = e
                if "429" not in str(e) and "rate" not in str(e).lower():
                    raise
                # 睡眠期间释放信号量,否则一次 429 会堵住整条流水线(2026-07-26)
                semaphore.release()
                try:
                    await asyncio.sleep(8)
                finally:
                    await semaphore.acquire()
        raise last

async def evaluate_answer(semaphore, answer, key_points, model):
    tasks = [evaluate_single_key_point(semaphore, key_point, answer, model) for key_point in key_points]
    results = await asyncio.gather(*tasks)
    return dict(results)

async def evaluate_query(semaphore, query_id, anwser_dir, key_point_dir, model):
    a_path = anwser_dir / f"{query_id}.a"
    p_path = key_point_dir / f"{query_id}_aggregated.json"

    if not a_path.exists():
        print(f"Warning: Missing answer file for query {query_id}")
        return query_id, None
    if not p_path.exists():
        print(f"Warning: Missing key point file for query {query_id}")
        return query_id, None

    answer = a_path.read_text().strip()
    with open(p_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        key_points = data["key_points"]

    try:
        evaluations = await evaluate_answer(semaphore, answer, key_points, model)
        return query_id, {
            "labels": evaluations,
        }
    except Exception as e:
        print(f"Error evaluating {query_id}: {e}")
        return query_id, None

async def evaluate_folder_async(subfolder_name, model, path_to_reports, key_point_dir):
    folder_path = Path(path_to_reports) / subfolder_name
    output_file = folder_path / f"evaluation_results_kpr_{model}.json"
    key_point_dir = Path(key_point_dir)

    all_results = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    print(f"Skipped queries: {len(all_results)}")

    semaphore = asyncio.Semaphore(int(__import__("os").getenv("DRGYM_SEM", "100")))

    query_ids = [p.stem for p in folder_path.glob("*.q") if p.stem not in all_results]
    query_ids = query_ids
    def _save():
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 增量落盘 + 配额耗尽早停(2026-07-25),同 eval_quality_async
    tasks = [asyncio.ensure_future(evaluate_query(semaphore, qid, folder_path, key_point_dir, model))
             for qid in query_ids]
    done_n, fail_streak = 0, 0
    for fut in tqdm_asyncio.as_completed(tasks):
        query_id, result = await fut
        if result is not None:
            all_results[query_id] = result
            fail_streak = 0
        else:
            fail_streak += 1
        done_n += 1
        if done_n % 10 == 0:
            _save()
        if fail_streak >= 30:
            print(f"[kpr] 连续 {fail_streak} 次失败(疑似配额耗尽),提前停止并保存 {len(all_results)} 条", flush=True)
            for t in tasks:
                t.cancel()
            break
    _save()

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder")
    parser.add_argument("--open_ai_model")
    args = parser.parse_args()

    path_to_reports = "/data/group_data/cx_group/deepsearch_benchmark/reports/"
    path_to_results = "/data/group_data/cx_group/deepsearch_benchmark/reports/"
    path_to_key_point = "key_point"

    print(f"Evaluating {args.subfolder} using {args.open_ai_model}")
    results = asyncio.run(evaluate_folder_async(args.subfolder, args.open_ai_model, path_to_reports, path_to_key_point))

    total_support_rate = 0
    total_omitted_rate = 0
    total_contradicted_rate = 0
    num_queries = len(results)

    for query_id, labels in results.items():
        supported_count = 0
        omitted_count = 0
        contradicted_count = 0
        for point_number, label in labels["labels"].items():
            if label[0] == "Supported":
                supported_count += 1
            elif label[0] == "Omitted":
                omitted_count += 1
            elif label[0] == "Contradicted":
                contradicted_count += 1
        
        total_points = len(labels["labels"])
        if total_points == 0:
            continue
        support_rate = supported_count / total_points * 100
        omitted_rate = omitted_count / total_points * 100
        contradicted_rate = contradicted_count / total_points * 100

        results[query_id]["support_rate"] = support_rate
        results[query_id]["omitted_rate"] = omitted_rate
        results[query_id]["contradicted_rate"] = contradicted_rate

        total_support_rate += support_rate
        total_omitted_rate += omitted_rate
        total_contradicted_rate += contradicted_rate

    avg_support = total_support_rate / num_queries
    avg_omitted = total_omitted_rate / num_queries
    avg_contradicted = total_contradicted_rate / num_queries

    print(f"\nAverages across {num_queries} queries:")
    print(f"  Average support rate: {avg_support:.2f}%")
    print(f"  Average omitted rate: {avg_omitted:.2f}%")
    print(f"  Average contradicted rate: {avg_contradicted:.2f}%")
        
    
    result_dir = Path(path_to_results) / args.subfolder
    result_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_path = result_dir / f"evaluation_results_kpr_{args.open_ai_model}.json"
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved detailed evaluation results to {detailed_path}")