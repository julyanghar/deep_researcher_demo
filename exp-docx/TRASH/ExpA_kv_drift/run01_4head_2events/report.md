# Exp0(Exp A)首批结果报告 —— KV 复用的逐层逐 token 漂移

> 一句话:在真实 deep-research loop(Qwen3-32B + CacheBlend)上,**直接在 blender 干活那一刻**拍下
> 「复用值 KV^r」和「真值 KV*」,测出 supervisor / writer 复用 worker 摘要时 KV 漂移多少。
> 首批 2 个复用事件:**浅层几乎不漂、深层最狠(~第 45 层)、V 比 K 漂 ~2.5 倍、少数 token 彻底漂飞**。

---

## 1. 要回答什么(大白话)

KV 复用 = **复用一份算好的笔记,而不是重读原文重做笔记**。
- worker 写摘要 `s_i` 时,模型给它算了份 KV(笔记)= `KV^w`。
- supervisor/writer 要用这摘要,本该在新语境**从头重算**一份 KV(真值)= `KV*`;
- 但复用是偷懒:把旧笔记搬过来、只调下"页码"(RoPE 位置)、内容不重算 = `KV^r`。

**Exp A 就问:`KV^r` 离 `KV*` 多远?** 逐层、逐 token 画出来 —— 哪层漂最狠、哪些 token 漂最狠,
为后面"该刷新哪些 KV"提供判据。

## 2. 怎么取数(关键设计 + 取舍)

**难点**:模型在 HTTP server 后面,客户端只看得到文字、摸不到 KV 张量。

**笨办法(放弃)**:录 prompt → 离线手动重算三份 KV(自己 prefill + FusedRope 旋转)。绕、还要担心重分词对齐。

**采用的聪明办法**:CacheBlend 复用时**本来就同时捏着两份 KV** —— 看
[blender.py:96](../../../lmcache/v1/compute/blend/blender.py#L96) `old_k = get_kv(...)` 是搬来的旧笔记(`KV^r`,已 RoPE 对齐),
[blender.py:112](../../../lmcache/v1/compute/blend/blender.py#L112) 模型新算的 `k` 是重做草稿;它本就靠比这两者(`diff_k`)挑哪些 token 要重做。
**把重算比例开到 1.0**,草稿就变成完整重算 = 真值 `KV*`。于是只要在那一刻把两份拍下来,**数据从产生处直接拿,不另搭重算**。

具体落点(LMCache **只改这一个文件**,env 门控、关时零开销):
- 抓取:[blender.py:114-126](../../../lmcache/v1/compute/blend/blender.py#L114) 在 RoPE 之后、top-k 挑拣之前,按层存 `old_k/old_v`(KV^r)与 `k/v`(KV*)。
- 落盘:[_flush_dump blender.py:228](../../../lmcache/v1/compute/blend/blender.py#L228) blend 结束把 64 层堆成 `[2,L,T,hidden]` 两份 + tokens 存一个 `.pt`。
- 开关:[blender.py:79](../../../lmcache/v1/compute/blend/blender.py#L79) `LMCACHE_BLEND_DUMP_KV`;真值靠 server 设 `LMCACHE_BLEND_RECOMPUTE_RATIOS=1.0`。

**为什么天然只覆盖 supervisor + writer**:blender 只在「有命中复用」时触发;worker 写摘要是**产地**(不 blend)。
本次 2 个 dump 的 system prompt 直接印证:`blend_00000` = "research supervisor…"、`blend_00001` = "Write a … research report…"。

**配套改动**(deep_researcher_demo):加 `SUPERVISOR_REASONING`([agents.py:36](../../../../deep_researcher_demo/deep_researcher_demo/agents.py#L36))让 supervisor 把每轮决策当 `r_t` 缓存、与 worker 输出交错([interleave_segments agents.py:56](../../../../deep_researcher_demo/deep_researcher_demo/agents.py#L56)),这样 supervisor 才有 summary **和** reasoning 两种可复用段。默认关、关闭即回归。

## 3. 怎么保证可信(先证后用)

- **命门:ratio=1.0 真等于 full prefill 吗?** [exp_a_validate_blender.py](../../../server/vllm/Exp_A/exp_a_validate_blender.py):同一段先 clean prefill 存下(那份是干净 full-prefill KV),再复用触发 blend 拍 `kv_star`,比两者 → **mean(1-cos)=2e-6**,等价。**PASS**。
- **复用真发生**:段级 hit 对账,supervisor 轮2 命中从 1126 跳到 1631(正好 +r_1 的 504 tok)→ reasoning trace 确被复用;writer 命中 1088 = 前缀+out_1。
- **RoPE 对 Qwen3-32B 有效**:server 启动自检 Max K error 0.012(<0.1)。

## 4. 结果

**实验配置**:Qwen3-32B(64 层,8 KV 头/head_dim 128),TP2,CacheBlend,`ratio=1.0`,1 个 query、1 轮。
2 个复用事件:supervisor decide(复用 out_1,1411 tok)、writer(复用 out_1,1373 tok)。

### 逐层漂移 dev(l)(KV^r vs KV*,1-cos;跨两事件平均)
| layer | 0 | 1 | 2 | 16 | 32 | 43 | 47 | 48 | 62 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|
| **K** | .0007 | .0017 | .0068 | .012 | .013 | **.087** | | | .044 | .025 |
| **V** | .0007 | .0079 | .015 | .023 | .075 | | **.225** | .204 | .173 | .172 |

全层均值 **K=0.036、V=0.092**;峰值层 **K@43、V@47**。

### 逐 token 漂移(看稀疏性)
| 事件 | T | K 1-cos 均值 | p90 | max |
|---|---|---|---|---|
| supervisor(blend_00000) | 1411 | 0.0362 | 0.083 | 1.000 |
| writer(blend_00001) | 1373 | 0.0349 | 0.081 | 1.000 |

两角色几乎一致(都复用同一段 out_1,只是语境不同)。

## 5. 关键发现(大白话)

1. **浅层几乎不漂**(layer 0~2 ≈ 0.001~0.007)→ 复用旧笔记在浅层很安全(sink/前缀稳)。
2. **深层最狠,峰值在 ~第 45 层(70% 深度),最后几层略回落** —— **不是** RelayCaching 在 Mistral 上说的"中层最狠"。这是"别照抄别人结论"的实证。
3. **V 比 K 漂 ~2.5 倍**(V 均值 0.092 vs K 0.036)→ 这版里 **V 才是复用误差的大头**,要刷新优先盯 V。
4. **稀疏**:绝大多数 token 漂得很小(均值 ~0.035),**极少数彻底漂飞(1-cos=1)** → 祸首是一小撮 token,印证"稀疏化只救该救的"思路。

## 6. 局限 + 下一步(待修)

1. **TP2 只抓到 8 KV 头里的 4 个**:两 worker 写同一文件互相覆盖(512=4×128)。结构能看但不全 → **dump 文件名带 TP rank、两半合并**。
2. **样本太少(2 个事件、1 轮、未采 r_t 漂移)** → 多跑 query + 轮2 探针,采 reasoning trace 的漂移、扩样本。
3. **max=1.0 离群 token** 疑似近零范数(分隔符?)→ 单独查、必要时剔除。
4. 曲线脚本按角色分别出(supervisor / writer 各一条),不只聚合。

## 7. 结果文件位置

| 内容 | 路径 |
|---|---|
| KV dump(supervisor)| `/home/yilin/tmp/exp_a_real_dump/blend_00000.pt` |
| KV dump(writer)| `/home/yilin/tmp/exp_a_real_dump/blend_00001.pt` |
| 漂移曲线图 | `/home/yilin/tmp/exp_a_curves.png` |
| harvest(prompt/段标签)| `/home/yilin/tmp/drift_harvest_expA.jsonl` |
| loop 运行日志 | `/home/yilin/tmp/exp_a_real_run.log` |
| server 日志(blend/dump)| `/home/yilin/tmp/exp0_server_32b_dump.log` |
| 命门验证日志 | `/home/yilin/tmp/exp_a_validate2.log` |
| 取数/算曲线脚本 | [exp_a_validate_blender.py](../../../server/vllm/Exp_A/exp_a_validate_blender.py)、[exp_a_curves.py](../../../server/vllm/Exp_A/exp_a_curves.py) |

> 配套:方法/改动细节见 [Exp0_implementation_changes.md](../../Exp0_implementation_changes.md);总实验方案见 `EfficientDeepResearchAgent_实验方案.md`。
