# vLLM 非确定性测试 —— 同一题跑两遍、对比完整推理轨迹

> **目的**:量化 deep research agent 在 **vanilla vLLM**(不开 LMCache blend)下的 **run-to-run 非确定性**——同一道题、temp=0、跑两遍,轨迹会不会一样?在哪开始分叉?
> **为什么要做**:全段实验比"prefill 轨 vs reuse 轨"。但如果**两遍 vanilla 自己就不一样**,那"prefill vs reuse 的差异"就被非确定性污染了,必须先量出这个**噪声底**,才能把复用的真实效应从噪声里分出来。

---

## 一、为什么 temp=0 还会不确定(原理)

temp=0 = 贪心解码(每步取 logit 最大的 token)。**给定相同 logits 它是确定的**,但 vLLM serving 下 **logits 本身不逐 bit 一致**:
- **batch 大小变 → 浮点累加顺序变**(浮点加法不满足结合律)→ 同一请求在不同 batch 里算,logits 有极微小差异;
- 部分 CUDA kernel 并行规约非确定;连续 batching / chunked prefill 也改变计算路径。

**短输出(决策 continue/complete)**:top-2 token logit 差距大 → argmax 翻不动 → 稳定。
**长输出(query plan / summary / 报告)**:几百步里总有某步 top-2 几乎平手 → 微噪声翻 argmax → 从那步起级联分叉。

---

## 二、实验设计

**对照**:同一道题,vanilla vLLM(无 blend、无 control),**同一 server 实例**,prefill 模式跑两遍,**抓所有 LLM call**,逐 call 找第一处分歧。

**关键控制**:
- **vanilla server**:用 [config_vanilla_32b.yaml](../../tmp/config_vanilla_32b.yaml)(= Qwen3-32B config 去掉 `kv-transfer-config` 行 → 无 LMCache connector → 无 blend),启动**不带任何 `LMCACHE_*` 环境变量**。
- **抓全部 LLM call**:env `DRIFT_HARVEST_TAGS='*'`(demo 的 harvest 机制,见 [llm.py](../../deep_researcher_demo/deep_researcher_demo/llm.py) 第 34 行 `_HARVEST_TAGS`)。tag 见 [agents.py](../../deep_researcher_demo/deep_researcher_demo/agents.py):`INITIAL_RESEARCH_QUESTIONS_JSON`(初始问题)→ `QUERY_PLAN_JSON`(子查询)→ `RESEARCH_SUMMARY_TEXT`(摘要)→ `SUPERVISOR_DECISION_JSON`(决策)→ `FINAL_REPORT_MARKDOWN`(终稿)。
- **搜索**:为把"搜索非确定性"和"LLM 非确定性"分开,搜索走 **replay 缓存**(见第五节)。

---

## 三、用到的代码

| 角色 | 文件 | 说明 |
|---|---|---|
| **跑一条轨** | [exp_fullseg_runner.py](../server/vllm/Exp_fullseg/exp_fullseg_runner.py) 的 `run_trajectory(question, sample_id, qdir, mode)` | mode='prefill' → `KV_REUSE_SEPARATOR=""`(无分隔符、标准 prefill);subprocess 调 `python -m deep_researcher_demo --quiet --output <report> <question>`,落 harvest + report。 |
| **agent 本体** | `deep_researcher_demo`(`python -m`) | 3×3 多轮:初始问题 → 每轮 researcher(query plan → 搜 → summary)→ supervisor 决策 → 终稿。 |
| **抓 LLM call** | [llm.py](../../deep_researcher_demo/deep_researcher_demo/llm.py) `DRIFT_HARVEST` + `DRIFT_HARVEST_TAGS` | 每个 LLM 调用按 tag 落 jsonl(`*`=全抓)。 |
| **搜索缓存** | [search.py](../../deep_researcher_demo/deep_researcher_demo/search.py) `wrap_with_cache`(record/replay) | 缓存目录 `eval/results/search_cache/q{sample_id}/`(注意前缀 q,见 search.py 第 331 行 `qdir = root/f"q{sample_id}"`);`search_cache.json`={query:[urls]},`pages_index.json`={url:{file,title}}。 |
| **测试脚本**(三版,递进) | `/home/yilin/tmp/run_det.py` / `run_replay.py` / `run_warmcompare.py` | 见第四、五节。 |

---

## 四、运行流程(三版,逐步收紧)

### 版本 1:record 模式(初版,证明分叉发生在 query plan)
脚本 `/home/yilin/tmp/run_det.py`:同一题 q1 连跑两遍(`run_trajectory(q,'q1',dir,'prefill')`),`SEARCH_CACHE=record`(联网搜 + 存缓存)。

```bash
# 启 vanilla server(无 LMCache,GPU 自选;32B 需 TP4)
env CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n lmcache vllm serve \
  --config /home/yilin/tmp/config_vanilla_32b.yaml --max-logprobs 200 --return-tokens-as-token-ids
# 跑两遍 + 抓全部 call
cd /home/yilin/deep_researcher_demo
EXP_SERVER=http://localhost:30000 MODEL=Qwen3-32B OPENAI_BASE_URL=http://localhost:30000/v1 \
SEARCH_CACHE=record DRIFT_HARVEST_TAGS='*' \
python /home/yilin/tmp/run_det.py
```

### 版本 2:replay 模式(把搜索钉死)
`SEARCH_CACHE=replay` → 搜索只走缓存、不联网(cache miss 才 fallback 联网一次并补缓存)。这样搜索≈query 的确定性函数,主要变量只剩 LLM。

### 版本 3:warm-cache + 全命中对照(最干净,`run_warmcompare.py`)
**问题**:replay 的 cold miss 会联网+补缓存 → run1 补的 run2 命中 → 两遍缓存不对称。
**解法**:先**预热**——replay 反复跑同题,cold→联网补,直到**缓存不再增长(该次全命中、零 cold)**;再跑**两遍对照**,每遍校验**0 新增(全命中、零联网)**,否则重跑该遍。这样两遍看到**完全相同的冻结缓存**→ 搜索完全对称、确定 → 只剩 LLM 非确定性。

```bash
SEARCH_CACHE 由脚本内设为 replay;DRIFT_HARVEST_TAGS='*'
python /home/yilin/tmp/run_warmcompare.py   # 预热到稳定 → 2 遍对照 → 存 trace_run{1,2}.md
```
判据用**缓存 query 数前后是否增长**(增长=有 cold 联网;不变=全命中)。

---

## 五、对比方法(逐 LLM call 找第一处分歧)

两遍的 harvest(`harvest_prefill.jsonl`)各是一串 `{tag, content, max_tokens}`。按**逻辑环节**(不被并发乱序干扰)比:
1. **初始问题**(`INITIAL_RESEARCH_QUESTIONS_JSON`,输入=原题、完全相同)→ 同 prompt 同输出?
2. **query plan**(`QUERY_PLAN_JSON`,子查询)→ 集合一致?
3. **summary**(`RESEARCH_SUMMARY_TEXT`)→ 集合一致?
4. **决策**(`SUPERVISOR_DECISION_JSON`)、**终稿**(`FINAL_REPORT_MARKDOWN`)。

轨迹存成可读 md:`/home/yilin/tmp/trace_run1.md`、`trace_run2.md`(每个 call 的 tag + 全文逐条列出,供人工查看)。

---

## 六、结果与结论

**版本 1(record,q1)实测**:
| 环节 | 两遍是否一致 |
|---|---|
| 初始问题 | ✅ **完全一致**(同 prompt → 同输出,这步复现了) |
| **query plan(子查询)** | ❌ **第一处分歧**:8 vs 3 个,首个 plan 就不同(`"...buffs categories"` vs `"...buffs obtain methods"`)|
| summary / 决策 / 终稿 | ❌ 级联全不同(决策 `[c,c,c]` vs `[complete]`,summary 24 vs 9 条)|

**结论**:
- **同一题、同 prompt、temp=0,两遍 vanilla vLLM 走出完全不同的轨迹。**
- **第一处分歧 = query plan**(子查询生成)——它发生在**搜索之前**,所以是**纯 LLM 非确定性**(不是搜索、不是缓存)。
- **单个 LLM call 的微小非确定性 → 整条 agent 轨迹两样。** 这就是噪声底,而且巨大。

**版本 3(warm-cache)实测——一个更强的结论:"完全用缓存"做不到。**
试图预热缓存到"全命中"以做完全确定的 replay 对照,**失败了**:
| 阶段 | 结果 |
|---|---|
| 预热 10 遍 | 缓存 208→301,每遍 +4~18 个**新** query |
| run1 对照(要求 0 cold)| 6 次尝试每次都有 cold(+1~16),**从未到 0** → 放弃 |
| run2 对照 | 同上 → 放弃 |
| 跑完 22 遍 | 缓存到 **364 仍在长** |

→ **agent 的 query plan 非确定性强到缓存永远填不满**——每遍都生成新子查询变体,query 空间被 LLM 非确定性撑成无界。**所以"两遍完全命中缓存、零联网"对这个 agent 不可能实现。** 这反过来是非确定性最硬的量化证据。
→ 要让**搜索完全确定**,只有 **strict replay(cold→空、不联网不补)** 一条路(搜索=query 的确定性函数,代价是 cold query 拿空结果);或干脆接受非确定、改用**统计噪声底**(多遍跑)而非追求单遍确定。

**对全段实验的含义**:
1. "prefill vs reuse"的轨迹差异**绝大部分被这个噪声底淹没**;
2. 复用轨"**稳定**更差"必须是**系统性效应**(`<|fim_pad|>` 分隔符),纯噪声是对称的;
3. **重跑必须**:① prefill 两遍当噪声底;② held 住分隔符的正确对照(复用 vs 全重算);③ 多题统计(单题被噪声主导)。

---

## 七、文件清单(可复现)
- server 配置:[config_vanilla_32b.yaml](../../tmp/config_vanilla_32b.yaml)
- 跑轨:[exp_fullseg_runner.py](../server/vllm/Exp_fullseg/exp_fullseg_runner.py)(`run_trajectory`)
- 抓 call:[llm.py](../../deep_researcher_demo/deep_researcher_demo/llm.py)(`DRIFT_HARVEST_TAGS`)、tag 定义 [agents.py](../../deep_researcher_demo/deep_researcher_demo/agents.py)
- 搜索缓存:[search.py](../../deep_researcher_demo/deep_researcher_demo/search.py)(`wrap_with_cache` record/replay)
- 测试脚本:`/home/yilin/tmp/run_det.py`(record 版)、`run_replay.py`(replay 版)、`run_warmcompare.py`(warm-cache 全命中对照版)
- 轨迹产物:`/home/yilin/tmp/trace_run1.md`、`trace_run2.md`;harvest 在 `/home/yilin/tmp/{det_run1,replay_run1,cmp_run1}/q*/harvest_prefill.jsonl`
