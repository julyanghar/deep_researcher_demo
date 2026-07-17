# Exp_M · 单题结果复核方法(用产物文件交叉验证)

> 目的:跑完一题后,**不只看"跑没报错",而是用产物文件交叉验证**四件事:
> ① 12 分支都实跑了;② 每个分支的**复用条件**对(重算哪些 summary/层、其余复用);③ **M 真抓到了**(非 0);④ demo 里 supervisor **真复用**。
> 数据:`/home/yilin/tmp/exp_m_test/`(单题验证用的独立目录)。

---

## 一、每个文件是什么、能核什么、**不能**核什么(关键)
| 文件 | 是什么 | 能核 | 不能核 |
|---|---|---|---|
| `q0/harvest.jsonl` | **demo 主轨**每次 LLM 调用的**完整 messages**(summary/decision/writer)| 重建 super_prompt、writer_prompt;summary 原文 | 分支重放(不在里头)|
| `q0/meta.json` | driver 落的元信息(两份 dump event + 两帧 summary 分段/位)| 分段数、summary 位数、事件号是否自洽 | 实际复用比例 |
| `q0/reports.json` | **12 个分支生成的报告** | **分支跑全没**、**复用是否真改变了输出** | 各分支的复用%(只看输出)|
| `q0/controls/branch_*.json` | 每分支的 control(要重算哪些位/层)| **选位对不对**(挡位×Ns)| server 有没有真照做 |
| `driver.log` | driver 进度 | 流程到哪(`[M]`/`[branch1]`/`[done]`)、M dump 事件 | — |
| `llm_calls.jsonl` | **demo 主轨**逐调用 timing/tag | demo 跑了几轮、几条 summary/decision | **❌ 不含 12 分支**(分支是 driver 直连 server,不过 demo 的 llm.py)|
| `$EXP_W_DIR/server.log` | vLLM+LMCache,含 `BLEND_PATH` | **demo 复用% + 每分支实际复用%**(server 侧铁证)| — |

> ⚠ **最容易踩的误解**:`llm_calls.jsonl` 看着像"所有 LLM 调用",其实**只有 demo 子进程的主轨**。12 个分支是 driver 用 `requests.post` 直接打 server 的、**不经过** demo 的日志,所以分支验证**不能靠它**,要靠 `reports.json` + `controls/` + `server.log`。

---

## 二、复核链路(几条独立的,互相印证)

### 链路 A:复用条件对不对(选位侧 + server 侧 双证)
1. **选位侧**(`controls/branch_*.json`):逐分支算实际重算量,对挡位。
   - token 分支:`len(recompute_token_idx) − 非summary位` = 重算的 summary 数,应 ≈ 挡位×Ns(Ns=命中区 summary 总数)。
   - layer 分支:`len(recompute_layers)` 应 = 挡位×64(50%→32、75%→48、25%→16)。
   - full_reuse:重算 summary = 0;full prefill:`{reuse_token_ranges:[]}` 全重算。
2. **server 侧**(`server.log` 的 `BLEND_PATH=CONTROL`):同一批分支在 writer prompt 上跑出的 `reuse=…%`/`recompute …/64 layers` 应和上面对得上。
   - **两侧都对 = control 写对了 且 server 真按它复用了。**
   - **⚠ server.log 是跨多次运行累积的,必须先隔离"本次 run"**,否则会混进上一题/中途测试的 blend:
     - 按 **本次 writer summary 长度**过滤(本次 `total=4443`;上一次是 8065、中途隔离测试是 super_prompt 8225);
     - 按 **本次时间窗口**(driver 起 ~ `reports.json` 写完)再卡一遍;
     - **去连续重复**(4 个 TP 每次 blend 打 4 条相同行);
     - 按 **driver 的分支顺序**(branch1 → writer-ΔKV-dump → 3 → 40,41,42,43 → 50,51,52,53,54,55)**逐次对位**。
   - **次数应 = 13**(branch1 + writer-dump + 11 分支);writer-dump 是 ΔK/ΔV 那遍(`reuse 0%`),不算实验分支但会出现在序列里。

### 链路 B:复用**真起作用**了吗(输出侧,最强)
- `reports.json` 里 **12 份报告应该互不相同**(尤其 full prefill 1 vs full reuse 3 应差很多)。
- 量化:对每份报告和 full-prefill(branch 1)算 `SequenceMatcher` 相似度。
- **若很多分支报告和 branch 1 一模一样 → control 没真生效(危险)**;若全不同、且 full reuse 最不像 → 复用确实在改变输出 ✓。
- 这条是端到端的:绕过 control/日志,直接看"复用有没有让结果变"。

### 链路 C:M 真抓到了吗
- `q0/controls/diag.json` 的 `M_rowsum_mean > 0` 且非均匀。
- 配 `driver.log` 的 `[M] decision=…tok (SEP 包装+丢末位) … event=N`,确认走了"决策段命中→进 blend 区→抓 q"那条(末位坑见 [m_capture_debug.md](m_capture_debug.md))。
- 深核:直接 load M dump,看决策位 q 的 `T` 是否 `> super_prompt_len`(决策位进了 dump),`compute_M(key_kind="kv_r")` 出来非 0。

### 链路 D:demo 里 supervisor 真复用
- `server.log` 搜 demo 阶段的 `BLEND_PATH=CacheBlend … reuse=…%`,应 ≈ 100%(原生 CacheBlend、ratio 0)。
- 若是 `reuse=0% TRUTH` 则说明 demo 被强制全重算(run_demo 没删 control 的旧 bug)。

### 链路 E:prompt/分段自洽(meta ↔ harvest)
- 从 `harvest.jsonl` 重建:`pick_decision_call`→super_prompt、`pick_writer_call`→writer_prompt,`find_segments` 切段。
- 对 `meta.json` 的 `segments`/`super_segments`/`*_summary_pos` 是否一致。
- writer 段数可能比 super 少几段(末段超缓存被截)→ `transfer_M` 按**内容**对齐处理,不是 bug。

---

## 三、这次(q0)的复核结果
| 链路 | 结果 |
|---|---|
| A 选位侧 | branch1 全重算;branch3 重算 summary 0(全复用);50%→2190、75%→3284、25%→1095(=挡位×4379);layer 32/48/16 层 ✅ |
| A server 侧 | 按 total=4443+窗口隔离本次,**按时间序排出 13 次 blend,逐次对到分支**(1→dump→3→40…55):reuse 0% / 0%(dump)/ 98.6% / 50.7% / 32层 / 50.7% / 32层 / 75.4% / 48层 / 50.7% / 32层 / 26.1% / 16层——顺序、次数、比例全吻合 ✅ |
| B 输出侧 | **12 份报告全不同**,与 full-prefill 相似度 0.08–0.37(full reuse 最低 0.084)→ 复用确实改变输出 ✅ |
| C M | `M_rowsum_mean=2552`(非 0);`[M] decision=194tok (SEP 包装+丢末位)`✅ |
| D demo 复用 | `BLEND_PATH=CacheBlend … reuse=100%` ✅ |
| E 自洽 | writer/super 各 25 段、summary 位 4499/4500、events 7/8 ✅ |

**结论:q0 全 12 分支实跑,复用条件(选位+server 双证)正确,复用真改变了输出,M 抓到,demo supervisor 复用。**

---

## 四、复核脚本(可复跑)
- 链路 A 选位:读 `controls/branch_*.json` 算重算量对挡位(见本会话脚本)。
- 链路 B 输出:`difflib.SequenceMatcher` 对每分支报告 vs branch1。
- 链路 C/D:`grep BLEND_PATH` server.log + load M dump 看 T。
- 一句话:**control 文件核"选得对不对",server.log 核"server 做没做",reports 核"有没有真效果",三者齐了才算复核通过。**
