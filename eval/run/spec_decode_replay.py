"""把 harvest 里的真实 prompt 顺序重放到 vLLM,测每次调用的 TTFT / decode tokens/s / 总延迟。
用于对比:开 / 不开 suffix 投机解码,推理效率差多少。

用法:
  python eval/run/spec_decode_replay.py --workload W.jsonl --out R.jsonl --label baseline [--limit N]
- 流式(stream=True)拿 TTFT + decode 时间;temperature=0 贪心(确定性,便于无损比对 + suffix 命中)。
- 顺序发(并发1)干净计时,先 warmup 2 条。
"""
import argparse, json, time, hashlib
import urllib.request


def _post_stream(url, model, messages, max_tokens):
    """流式 POST /v1/chat/completions;返回 (ttft_s, total_s, n_chunks, text)。"""
    body = json.dumps({
        "model": model, "messages": messages,
        "temperature": 0, "max_tokens": max_tokens, "stream": True,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
    parts = []
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = (obj.get("choices") or [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
                parts.append(piece)
    total = time.perf_counter() - t0
    return ttft or total, total, n, "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)          # baseline / suffix
    ap.add_argument("--url", default="http://localhost:30000/v1/chat/completions")
    ap.add_argument("--model", default="Qwen3-32B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.workload) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    # warmup(不计入)
    for r in rows[:args.warmup]:
        try:
            _post_stream(args.url, args.model, r["messages"], min(64, r.get("max_tokens") or 64))
        except Exception:
            pass

    fo = open(args.out, "w")
    for i, r in enumerate(rows):
        mt = r.get("max_tokens") or 1024
        try:
            ttft, total, n, text = _post_stream(args.url, args.model, r["messages"], mt)
            decode_s = max(total - ttft, 1e-6)
            rec = {
                "i": i, "tag": r["tag"], "label": args.label,
                "n_out": n, "ttft_s": round(ttft, 4), "decode_s": round(decode_s, 4),
                "total_s": round(total, 4),
                "decode_tps": round((n - 1) / decode_s, 2) if n > 1 else 0.0,
                "out_hash": hashlib.sha1(text.encode()).hexdigest()[:16],
                "out_len_char": len(text),
            }
        except Exception as e:  # noqa: BLE001
            rec = {"i": i, "tag": r["tag"], "label": args.label, "error": str(e)[:120]}
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
        print(f"[{args.label}] {i+1}/{len(rows)} {r['tag'][:20]} "
              f"tps={rec.get('decode_tps','?')} total={rec.get('total_s','?')}s", flush=True)
    fo.close()


if __name__ == "__main__":
    main()
