# suffix 投机解码 推理效率实验

> 问题:demo 的 `RESEARCH_SUMMARY_TEXT` / `FINAL_REPORT_MARKDOWN` 生成大量**逐字照搬输入 prompt**(见 [../prompt-overlap-analysis-v2/summary.md](../../prompt-overlap-analysis-v2/summary.md):report Coverage@8 均值 91%、summary 54%)。用 vLLM 0.18 内置的 **suffix 投机解码**(`method="suffix"`,拿输出去匹配 prompt 后缀树、批量命中被抄的整段),测:**开/不开,推理效率差多少?**
> suffix 机制大白话+调用链见 `/home/yilin/LMCache/claude-docx/18-vllm-suffix-decoding.md`。

---

## 一句话结论
- **FINAL_REPORT(长报告):~1.76x 加速**(墙钟 report 段 80s→45s)。稳,对参数不敏感。**值得开。**
- **RESEARCH_SUMMARY(短摘要):~0.87–0.96x,略慢**。**任何调参都翻不了正**——损失是"开 suffix 的固定开销"(不是配置没调对)。
- **净收益看工作负载里长报告的占比**;而报告恰是 decode 大头(每题 ~105s decode),所以对整个 pipeline 划算。

## 结论表(全部:全新 server 冷树 · batch=1 串行 · 单进程)

| 任务 | baseline | 默认 suffix | 加速 | 备注 |
|---|--:|--:|--:|---|
| **FINAL_REPORT** (n=20) | 128 char/s | 236 char/s | **1.84x** | 含全局树预热 |
| FINAL_REPORT · 关全局树 | 128 | 225 | **1.76x** | **最贴生产:纯 prompt 树** |
| **RESEARCH_SUMMARY** (n=60) | 136 char/s | 130 char/s | **0.955x** | 默认已是最优点 |

指标 = **配对聚合 char/s**(同 index,Σ字符/Σdecode时间)。为什么这么算见"度量坑③"。

## 为什么两类任务分化
- **REPORT 快**:报告长(~1万字)且**大段照搬它 prompt 里的 summary**(复述来源/要点,单段中位 226 字)。suffix 用 report 自己的 **prompt 后缀树**就能一次命中整段、一次验证接受一大串 → decode 步数大减。**summary 本就在 report 的 prompt 里,所以不靠跨请求的全局树也能加速**(关全局树仍 1.76x)。
- **SUMMARY 略慢**:摘要短(~1.7千字)、照搬是**碎片式**(中位单段仅 41 字、~38 个小段)→ suffix 每次只多吐几个 token、段间频繁 miss,收益不够抵**固定开销**。

## 调参 sweep:没有配置能让 summary 翻正

| 配置(summary,冷树)| vs baseline | 接受率 |
|---|--:|--:|
| prob=0.05(最激进) | 0.889x | 28% |
| **prob=0.10(默认)** | **0.955x**(最优)| — |
| prob=0.30 | 0.922x | 33% |
| prob=0.50 | 0.917x | 35% |
| max_spec_factor=0.5(短草稿)| 0.838x | 41% |
| 关全局树 cache=0 | 0.872x | 50% |

**两个反直觉、都指向同一结论**:
1. **默认 prob=0.1 就是最优点**,往激进(0.05)或保守(0.3/0.5)调都更慢。
2. **接受率越高 ≠ 越快**(cache=0 接受 50% 却只有 0.872x)。接受率是"草稿里被接受的比例",不代表每步净多吐了多少 token;草稿变短/固定开销没省 → 接受率高也白搭。

→ summary 的 ~5% 损失**不是浪费草稿**(那样调 prob 该有反应),**是固定开销**:vLLM 日志明说 **`Async scheduling ... will be disabled`**(开 suffix 就关异步调度)+ 每步建/查后缀树。**调 spec 参数消不掉**,所以 summary 摸不到 baseline。REPORT 侧同样对参数钝(prob 0.1→0.5、全局树开/关都在 1.76–1.84x)。**结论:用默认参数即可,别调。**

---

## ⚠️ 三个测量陷阱(踩过,值得记)

投机解码效率**强烈依赖测量环境**,不控好会得到完全错误的结论。

### 坑① 全局后缀树"热树"污染(最坑,曾误得 1.9x)
`suffix_decoding_max_cached_requests=10000` 默认开 → 全局树**跨请求缓存历史输出**。在**处理过相同/相似请求的 server 上重测**,树把之前的输出**背下来了**,命中虚高。
- 实测:一台 prob=0.5 的 server 因前面跑过一模一样的 60 条 summary(全局树背熟每条输出),重跑同样 prompt 时 summary 假显 **1.912x**(某条 decode 6.1s→1.1s,5.5x);**全新 server 冷树重测只有 0.917x**。
- 教训:**每个配置必须全新 server(冷树)测才可比**;或 `max_cached_requests=0` 关掉全局树。
- 这也是**真实 production 收益**的一面(长跑 server、重复/相似 query 越跑越快)——但那是另一回事,别和"配置对比"混。

### 坑② 并发 batch
投机解码收益**在 batch=1 最大**(decode 是显存带宽瓶颈、算力闲着 → 白捡草稿验证),**batch 越大越缩水**(算力打满后,验证草稿和真实请求抢算力)。
- 本实验**严格串行(concurrency=1)**→ 测的是**上界**;生产真并发下加速比会小于此。
- 曾有 bug:两 replay 进程同时打 server → batch=2 重叠 + 交错写文件,数据作废重跑。

### 坑③ 流式 delta ≠ token(度量口径)
`decode_tps` 按**流式 delta 数**算,而 spec decode 一个 delta 含多 token → **严重低估**,对 suffix 不可用。改用 **char/s = out_len_char/decode_s**(字符跨配置可比);又因 temp0 贪心在批处理下**非逐位确定**(两次输出长度不同),逐请求 mean 会被离群带偏 → 最终用**同 index 配对的聚合 char/s**(Σ字符/Σ时间)。

---

## 方法
- **隔离 replay**:从 online100 harvest 抽 60 summary + 20 report 真实 prompt([workload_summary.jsonl](../data/workload_summary.jsonl) / [workload_report.jsonl](../data/workload_report.jsonl)),temp0 贪心,`stream=True` 量 TTFT/decode,**串行发**、先 warmup 2 条。脚本 [../../eval/run/spec_decode_replay.py](../../../eval/run/spec_decode_replay.py)。
- **单 server 4 卡(TP4,GPU 1/2/3/7)**,每个配置**全新 server 冷起**。baseline=`config_no_lmcache.yaml` 原样;suffix=追加 `speculative-config: '{"method":"suffix", ...}'`(见 [config_suffix.yaml](../data/config_suffix.yaml) 及各 `config_suffix_*.yaml`)。
- 前置坑:vllm 0.18 的 suffix 需 `arctic_inference`;装的 0.2.0 只带 py3.11 的 `_C.so`,env 是 3.12 → 从 `csrc` 用 nanobind+cmake 重编 `_C` 到 3.12。

## 产物(本目录)
- 结果:`res_baseline.jsonl`(不开)、`res_suffix.jsonl`(默认)、`res_v1/c2/c1fresh.jsonl`(prob 0.05/0.3/0.5)、`res_flow.jsonl`(factor0.5)、`res_g0.jsonl`(关全局树)、`res_c1rep.jsonl`(prob0.5 report)。**注:`res_c1.jsonl` 是热树污染样本(0.917→假1.9x),仅留作反面教材。**
- 接受率:`metrics_*.txt`;配置:`config_suffix_*.yaml`。
- 全量日志(server 输出)在 `/home/yilin/modify-code-runs/suffix-spec-decode/`;suffix 机制文档 `/home/yilin/LMCache/claude-docx/18-vllm-suffix-decoding.md`;照搬统计 [../prompt-overlap-analysis-v2/summary.md](../../prompt-overlap-analysis-v2/summary.md)。

## 后续分析
- [overlap-replay.md](overlap-replay.md):用本实验实际输入输出逐条验证——"可收割 token 占比"与实测加速**单调对应**(打平点 ~15-20%),固定税 ≈18%,决定因素是连续可收割量而非任务类型。
- [per-request-speedup-variance.md](explore-idea/data-source/per-request-speedup-variance.md):**题级差异报告**——同是 summary,加速比横跨 0.71x–2.56x;字符级 Coverage@8(v2 头条指标)对 summary 加速 r=−0.23(反向),只有 token 级照抄量能预测;检索型题加速、分析型题拖慢;完美路由可把 summary 段从 0.93x 翻到 1.05x → 值得投"生成前预测器"。
- [questions-log.md](questions-log.md):提问索引 + Q8/Q9 机制详解。
- [optimization-proposal.md](optimization-proposal.md):**优化方案设计**(L0 全开净省 ~16%/题 → L3 prompt 共设计 → L2 生成前预测器+路由 → L4 降税;含路线图与验证方式,未实施)。

## 部署建议
- **开 suffix、用默认参数**(`{"method":"suffix"}`)。report 阶段 ~1.76x+、summary 略亏 ~5%(短、可忽略),净赚。
- 别为 summary 调参(翻不正),别信"热树"跑出的高加速(那是重复 query 的产物)。
- 真并发负载下先小流量验一遍加速比(batch>1 会缩水)。
