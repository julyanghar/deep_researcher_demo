# 税拆分实验进度报告(2026-07-05 上午)

> 一句话:开 suffix 投机后,每个 decode step 固定多花 ~10ms(空草稿口径 C0−A),这笔"税"里 **propose 只占 0.7%、可插桩 CPU 环节合计仅 ~1.6ms,大头在 GPU 执行路径**。真凶已定案并经 2×2 对照锁死(⚠️ 本文 §3.4d 的早期定案"每步29ms GPU空转/memcpy依赖"**已订正**,以 [how-we-found-the-fixed-tax.md](how-we-found-the-fixed-tax.md) 为准):**固定税 = graph+eager 混合模式的逐层边界管理(慢发射)**——那 455μs"空隙"大头是 graph 捕获的 FFN 在跑(nsys 默认不记录),不是空转;(二次修正)ngram 同样交这笔税(发射 24.8μs 同样慢),靠每步接受摊薄赚回,异步经 D/E 直测仅值 ~0.5ms;suffix 额外付自家 CPU 机件 ~1.6ms、k24 变长再付桶抖动 ~8.5ms;suffix 全 eager(Ceager)实测直接翻正(summary 1.00x / report 2.29x)。
> 目标两问(用户提出):①关异步调度导致多少损失、为什么;②每步 propose 各环节时间占比。**两问都已答**;衍生出的"17ms 固定费到底是什么"仍在收窄。
> 详细计划见 [tax-split-experiment-plan.md](tax-split-experiment-plan.md);逐条数据/脚本在 `/home/yilin/modify-code-runs/suffix-spec-decode/tax-split/`。

---

## 一、为什么做这个实验

旧结论说"suffix 固定税 ≈18%",但那是个**混合值**——探查 server 日志发现:baseline 一直开着异步调度、suffix 一直被强制关(vLLM 白名单规则)。所以 18% 里混着"关异步""propose CPU""验证开销"三样,从未拆开。**拆开它决定优化方向**:如果异步占大头 → EAGLE 类(能拿回异步)优先;如果 propose 占大头 → 降税三件套(增量匹配等)优先;如果都不是 → 得找新方向。

---

## 二、配置矩阵(投机类型 × 异步调度)

| 投机 \ 异步 | 开 | 关 |
|---|---|---|
| **无投机** | **A** baseline | **B** +`--no-async-scheduling` |
| **ngram_gpu**(GPU 检索,异步白名单成员) | **D** | **E** |
| **suffix**(CPU 检索,默认参数) | ✗ 结构禁止 | **C** |
| **suffix 空草稿**(factor=0,永不下注) | ✗ | **C0** |
| **suffix k=8**(截短草稿) | ✗ | **G0** |
| **suffix k=8 + padding**(定长补齐命中规整图) | ✗ | **G** |
| baseline + **enforce-eager**(完全无 CUDA graph) | **F** | — |

差分逻辑:相邻两配置只差一件事,相减 = 那件事的成本。
- A−B = 纯异步损失
- C0−A = "光开着 spec 机制"的固定费(无草稿无验证)
- C−C0 = "真下注"的代价(验证+元数据+按量)
- D/E = 唯一"投机可开、异步可开可关"的一对 → 测投机在场时的异步贡献
- F−A = CUDA graph 的价值
- G−G0 = padding 定长的价值

**口径**:replay=串行 batch=1 的显微镜(读 ms/token);e2e=local 模式多轮(题内并发 1~9,看生产口径)。加速比一律长度归一 char/s,避开 temp0 漂移(旧教训)。

---

## 三、已跑完的实验 + 结果

### 3.1 主链 A/B/C/D/E:两口径 × 10条/10题(TP4)——已全部完成

| 配置 | 每步 ms | replay SUM char/s | replay REP char/s | 接受率 | e2e SUM(C/B) |
|---|--:|--:|--:|--:|--:|
| A(异步开,无投机) | 23.9 | 72 | 81 | — | 锚点 |
| B(异步关,无投机) | 24.4 | 71 | 79 | — | 锚点 |
| **C(suffix)** | **41.7** | 64 | **173** | 53% | 中位 ≈1.03 |
| **D(ngram_gpu,异步开)** | **31.5** | **74** | **200** | 16%※ | 待 e2e40 |
| E(ngram_gpu,异步关) | 31.7 | 74 | 199 | 16%※ | — |

※ 16% 是**全局混合**(计数器不分 tag);分 tag 每步接受:SUM 0.31、REP 2.74(suffix 对应 0.43/3.07),接受率天差地别,单一 tag 不成立。

**读出的结论**:
1. **异步损失小**(A vs B):replay ~2%、e2e ~3%——**答问①**;
2. **suffix 每步固定多花 ~17.5ms**(24→41.7),且与草稿量基本无关(旧80条回归截距 15.2ms、草稿斜率仅 0.09);
3. **D 双赢**:ngram_gpu 单 server 保住异步,replay 下 summary 74>72(翻正)、report 200(超 suffix 173),尽管接受率只有 suffix 的 1/3——"异步保留+GPU drafter 无 CPU 跑腿"的执行优势压过猜题质量。**⚠️ 但此"双赢"仅 batch=1 成立:e2e 并发下 summary 反转成 0.75x 净亏(§3.7),部署候选已降级(§五.5)。**

### 3.1b ngram_gpu(D/E 配置)的算法——注明

**一句话:prompt-lookup 检索投机的 GPU 张量版**,和 suffix 同为"照搬输入"式查表,但把 suffix 的所有"聪明机关"拆光、换成一次全 GPU 并行滑窗匹配。源码 [ngram_proposer_gpu.py](../../../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/v1/spec_decode/ngram_proposer_gpu.py) `_find_first_and_extract_all_n_parallel`(:46-120)。

每个 decode step 做的事(全部在 GPU 前向时间线内,张量并行、无 CPU 环节):
1. 取输出结尾的 n-gram,长度从 `prompt_lookup_max`(=6)往下试到 `min`(=2);
2. 每个长度用 `unfold` 生成全部滑窗,和结尾 n-gram 逐窗比较(`==...all(dim=-1)`),`argmax` 取**最早一次出现**的位置——不是频率投票,就是"上次它在哪";
3. 各长度里选**能匹配上的最长 n-gram**,抄它后面 **固定 k=8** 个 token 当草稿(不足则 -1 占位);
4. 草稿进目标模型一次前向,rejection sampler 验证接受(同 suffix,无损)。

**与 suffix 的机制对照(解释为什么它猜得差却跑得快)**:

| | suffix(C) | ngram_gpu(D/E) |
|---|---|---|
| 数据结构 | 限深后缀树(带频次) | 无索引,前向内 GPU 滑窗匹配 |
| 多处匹配 | 频率投票+置信度连乘 | 取**最早一次**出现,不投票 |
| match 长度择优 | 枚举+score 择优 | 取最长可匹配 n-gram |
| 草稿长度 | 变长 0~24(胆量∝证据) | **定长 k=8** |
| 草稿源 | 本地树+**全局树**(跨请求) | 只本请求上下文,**无全局树** |
| 执行位置 | **CPU、采样后** → 异步被迫关 | **GPU 前向内** → **异步保留** |

三处机关缺失(无频率打分/无全局树/定长)让它猜得差(实测每步接受 SUM 0.31、REP 2.74,均略低于 suffix 的 0.43/3.07);但"全 GPU、无 CPU 取草稿跑腿、异步不关"让它每步固定税从 suffix 的 ~17ms 压到 ~7.5ms——**执行便宜压过猜题差,replay 净赢**(见 3.1 表)。这也是"不开两台 server 也能投机+异步兼得"的单 server 部署候选。

### 3.2 propose 分段(SUFFIX_TIMING 插桩)——已完成

C_t4(TP4-40条)实测,21600 次 propose 累计:

| 环节 | ms/步 | 说明 |
|---|--:|---|
| speculate(查两棵树) | 0.080 | 本地树+全局树各一次 |
| add_response(喂树) | 0.036 | 输出边生成边入树 |
| start_request(建树) | 0.014 | 新请求建 prompt 树 |
| pattern+stop_request | 0.020 | |
| **propose 总计** | **0.16** | **仅占每步 ~0.7%** |

**答问②**:propose 六个子环节全部加起来 0.16ms,查树(speculate)是其中大头但绝对值极小。**降税三件套(增量匹配等)判死**——它优化的就是这 0.16ms,无肉可砍。

### 3.3 六件事直测(STEP_TIMING 引擎侧插桩)——已完成

C_t4 实测,把"一个 spec 步比普通步多做的事"逐项称量:

| 项 | ms/步 | 占 17ms | 归类 |
|---|--:|--:|---|
| ④ rejection sampler | 0.82 | 4.8% | 批改老师:逐位比 argmax 定接受 |
| ② spec metadata 构建 | 0.44 | 2.6% | 为[真+草稿]每位置排 logits 索引 |
| ⑤ 取草稿同步 RPC | 0.15 | 0.9% | EngineCore 跨进程要草稿(曾被误判为大头) |
| propose(①③之外) | 0.16 | 0.9% | 见 3.2 |
| ⑤b 草稿登记 | ~0 | 0% | |
| **CPU 侧合计** | **~1.6** | **~9%** | 全部可插桩 CPU 环节 |
| **残差(GPU 执行路径)** | **~15.4** | **~91%** | ← 真凶在这里 |

### 3.4 假说排除实验(TP2-40条)——已完成

| 假说 | 判决实验 | 结果 |
|---|---|---|
| CUDA graph 形状错配 | F(enforce-eager)−A | 完全无图只慢 **0.7ms/步** → **阵亡**(解释不了 10ms) |
| padding 命中规整图能救 | G(k8+pad)−G0(k8) | 只救回 **2.4ms/步** → 小效应 |
| attention 后端被换 | 对比 server 日志 | 两边都 FLASH_ATTN v2 → **排除** |

**收窄结论**:那 ~15ms 残差既不在 CPU 环节、不在 attention 后端 → 嫌疑落到"spec 快路径被踢"。C0 佐证:零草稿零验证,仅"spec 机制开着"就 +9.9ms(C0_t4 33.9 vs A 24.0)。

### 3.4b nsys 发射数对比(2026-07-05):注意力确实被踢出 graph——但"发射数=税"随后被推翻

nsys(2025.5.1)独占 TP4 各录 25s 稳态 decode,A(无spec)vs C0(空草稿,前向等大):

| | A(无spec) | C0(空草稿) | 倍数 |
|---|--:|--:|--:|
| GPU kernel 发射数 | 25,059 | **455,943** | **×18** |
| cudaLaunchKernel CPU 时间 | 161ms | **22,776ms** | +22.6s |
| cudaEventSynchronize | 24.8s | 37.8s | +12.9s |

**暴涨的 45 万次发射几乎全是 3 个注意力 kernel**(A 里=0,因被封装进 CUDA graph;C0 里各 14.4 万次):`flash_fwd_splitkv_kernel` / `flash_fwd_splitkv_combine_kernel` / `reshape_and_cache_flash_kernel`。

**初步机制(部分)**:开 spec 后注意力从 CUDA graph 回放退化成逐层 eager 发射 split-KV flash attention(那 3 个暴涨的 kernel)。**但"逐 kernel 发射就是税"被 Fea 实验推翻**(见下)。

### 3.4c Fea 补充实验(用户建议:baseline+eager+异步关,对齐 C0 口径)——推翻"launch-bound",指向同步

之前 F(enforce-eager)异步开、C0 异步关,口径不齐。补 Fea=baseline+enforce-eager+**异步关**,三方 nsys 对比(25s窗口):

| | A(graph,async开) | Fea(eager,async关) | C0(spec空草稿,async关) |
|---|--:|--:|--:|
| kernel 发射数 | 25k | **3.5M(×140)** | 456k(×18) |
| 每步 wall(replay) | 24ms | ~25ms | 34ms |
| cudaEventSynchronize | 24.9s | **2.7s** | **37.8s** |

**决定性反转**:**Fea 发 350 万次 kernel(C0 的 7.7 倍)每步却只慢 0.7ms;C0 发得少反而慢 15ms** → **kernel 发射数量不是税**(Fea 的海量发射全被 batch=1 显存等待掩盖)。撤回"launch-bound"定论。

**新领先信号=同步气泡**:与"慢"对应的是 cudaEventSynchronize——**C0 最高 37.8s > A 24.9s ≫ Fea 2.7s**。C0 的 eager 注意力路径里有 CPU 阻塞等 GPU 事件的操作,破坏了"发射与计算重叠",这才是税。

### 3.4d 逐步时间线(cuda_gpu_trace)——旧定案已证伪,简记

**旧结论(已证伪)**:"每层写 KV 前 GPU 干等 455μs、64 层≈每步 29ms 空转;根因是投机路径的带依赖 memcpy;优化方向=让 spec 注意力回 graph。"

**证伪证据(四条)**:①node 级 trace(`--cuda-graph-trace=node`)让 graph 内 kernel 现形——455μs"空隙"被 34k gemm+8.5k silu(FFN)填满,残余 gap 仅 88μs(C0)/47μs(D);②算术:可见注意力仅 1.4ms/步,32B 前向不可能只干这点活;③memcpy 数量 C0 与 Fea 相同(64 个/步),非差异来源;④ngram 同样有 380μs gap 却比原生快——"有 gap"根本不是慢的原因。

**现结论**:空隙大头 = graph 捕获的 FFN 在跑(nsys 默认不记录);真税 = 每层 ~100μs 的 graph↔eager 混合边界 + suffix 异步被关;suffix 全 eager(Ceager)实测翻正(summary 1.00x/report 2.29x)。完整论证见 [how-we-found-the-fixed-tax.md](how-we-found-the-fixed-tax.md) §六~§七;原始 gap 统计数据在 tax-split/nsys/ 物证与 how-we-found §6.1。

### 3.5 k=8(草稿长度上限)的效应——⚠️ 再订正:40条上 k8>k24 稳定成立,"TP4 方向相反"是 10 条噪声

**先厘清 k=8**:是 `num_speculative_tokens`=8,即**每步草稿最长 8 个 token 的上限**,不是树深度(树深仍 24)。主要 binding report(大段照抄本会掏 20+,砍到 8);对 summary 几乎不 binding(草稿本就 1~3)。

| SUM char/s(聚合 Σchar/Σdecode_s) | A(无spec) | C(k24) | G0(k8) | 说明 |
|---|--:|--:|--:|---|
| TP4(10条,8条summary) | 72 | 64 | 56 | ~~k=8 最差~~ **10条单点噪声,被 40 条推翻** |
| **TP4(40条,30条summary)【主口径】** | **152** | 126 | **141** | **k8>k24 明确**(G k8+pad 更到 148=0.97x) |
| TP2(30条summary) | 89 | 待 | 98 | 与 40 条同方向 |

**k×pad 全扫点补钉(TP4-40)**:k8+pad8 **0.97x/2.11x** > k16+pad16 0.89x/2.00x > k24+pad24 0.85x/1.91x(SUM/REP)——单调递减,**k 越大越差、pad 放大救不回**。机制:每步接受长度 k24=2.33 vs k8=2.21,**k=8 已装下几乎全部可接受草稿(接受饱和)**,k=24 多出的上限全是白验证的按量税。完整账本见 [how-we-found-the-fixed-tax.md](how-we-found-the-fixed-tax.md) §七b。

**旧结论(已证伪,简记)**:基于 TP4-10 条(k8=56<k24=64<A=72),先写过"k=8 首次稳定净赚"(过度宣称)、后又写过"k8 在 TP4 最差 / TP2-TP4 方向相反"——**两次判断都被 40 条主口径推翻**(G0 141>C 126,k×pad 扫点单调)。根因同一个:10 条口径样本太少+冷启动,是单点噪声;以 40 条为准。逐条原始数据在 tax-split/smoke10/。

### 3.5b suffix→ngram 每步税拆解(纠正:padding只值2ms,不是"定长省7ms")

TP4-40 summary 每步ms拆解(减原生A=24ms):

| 配置 | 每步ms | 税 | 相邻省 | 归因 |
|---|--:|--:|--:|---|
| C suffix k24变长 | 42.5 | +18.5 | — | |
| G0 suffix k8变长 | 37.3 | +13.3 | 5.2ms | **截短**(草稿短、少验证位置=按量费,与graph无关) |
| G suffix k8+pad定长 | 35.3 | +11.3 | 2.0ms | **padding/定长**(固定形状,小效应,印证§3.5"padding只救~2.4ms") |
| D ngram k8 | 31.4 | +7.4 | 3.9ms | ngram GPU原生草稿路径等更深优势 |

**纠正(2026-07-05)**:一度口头说"定长省7ms是追上ngram关键"——错。7ms里截短占5ms(按量费)、padding(真定长)只占2ms。padding是小效应,与§3.5"padding只救2.4ms"一致。加速比上G仍0.97x(TP4未翻正)——每步ms降≠加速比翻正,因每步token(1.39)未涨。

**补格判决(已完成 7/5)**:pad×{混合,全eager} 2×2 实测——混合列 pad **赚 2.0ms**(G0 37.3→G 35.3),全 eager 列 pad **倒亏 3.5ms**(CeagerK8 33.7→CeagerK8pad 37.2;REPORT 列同型)。**"pad 收益经由 graph 桶稳定"实锤闭环**;全 eager 部署下 pad 是纯负资产。且 eager 下 k 也无所谓(CeagerK8 151≈Ceager 152 char/s)——k8/pad 整套讲究只是混合模式的伤药。详见 how-we-found §七b.1。

**但机制自洽**:k=8 省的是**按量费**(草稿短→过前向的位置少→KV读/验证少,TP4 每步省 5.3ms),这与"固定税=路径切换、与草稿长度无关"(C0 空草稿仍+9.9ms)**是两个分量、不矛盾**。k=8 削按量费、削不到固定税 → **救不了 summary**(summary 亏的根子是固定税)。这反而再次印证主线。report 侧 k=8=k24(173=173,截断不伤吞吐)。

### 3.6 现象侧:T3 兄弟互抄(离线 SAM 分析)——已完成

- 多轮 100 题(online100_v2):互抄 harvest 中位 **18.8%**,union 边际增益 **12.9%**(prompt 14.3%→prompt∪兄弟 27.7%,近翻倍);
- corr(互抄, prompt harvest)= **+0.40**("此消彼长"经全量数据撤回,是"可抄性公因子");
- 输入 URL 无预测力(Jaccard 0.05~0.16, r=0.13),互抄源于"相似响应"非"读同样网页";
- 支撑研究方向 D1(图谱感知草稿源),详见 [research-directions-copy-speculation.md](../explore-idea/research-directions-copy-speculation.md)。

### 3.7 e2e 五格(7/4 已跑,10 题 local 缓存模式)——replay 结论在真实 agent 下的第一次校验

结果目录:`eval/results/drbench/tax_{A,B,C,D,E}_e2e/`(每题 `llm_calls.jsonl` 逐调用 timing / `harvest.jsonl` / `report.md`)。口径:local 缓存搜索、10 题、题内并发 3;**tok/s 是纯 decode 段吞吐**(逐调用 completion_tokens/decode_s 加总,不含排队/工具时间);墙钟含全部环节。

| 配置 | 每题墙钟中位 s | 10题总墙钟 min | SUM tok/s | REP tok/s | TTFT 中位 s | e2e 接受率 |
|---|--:|--:|--:|--:|--:|--:|
| A 原生(异步开) | 352 | 52.8 | 25.3 | 40.6 | 3.75 | — |
| B 原生+关异步 | 319 | 59.5 | 24.7 | 39.5 | 3.88 | — |
| **C suffix k24** | **285** | **44.1** | **25.9** | 81.7 (**2.01x**) | 3.76 | **39.9%** |
| D ngram k8(异步开) | 318 | 47.5 | **19.0 (0.75x)** | **90.0 (2.22x)** | 4.16 | **4.9%** |
| E ngram k8+关异步 | 334 | 49.8 | 18.7 | 90.3 | 4.69 | 5.1% |

**读出的结论(与 replay 对照,三处一致、两处反转)**:

1. **suffix 的 summary 从 replay 0.83x 翻到 e2e ≈平手(1.02x)** —— e2e 热全局树+兄弟互抄把接受率抬到 39.9%(replay 冷树约一半),接受收益first-time盖过了固定税。**replay 冷树口径低估 suffix。**
2. **ngram 的 summary 从 replay 1.03x 反转成 e2e 净亏(0.75x)** —— 机制在接受率快照里:D 发了 **95 万 draft token(C 的 4.3 倍,定长 k8 几乎步步满发),只接受 4.6 万(4.9%)**。replay batch=1 时多余验证搭权重读的便车≈免费;e2e 题内并发=3 时算力共享,垃圾草稿的验证要付真钱。**"定长满发"在并发下是负资产。**
3. **report 两家都大赢且方向与 replay 一致**(C 2.01x / D 2.22x),照抄型长输出的收益在真实负载下坐实;
4. **异步调度在 e2e 也只值几个点**(A vs B、D vs E 的 decode 吞吐差 ≈0~2%,墙钟差 ~5%),与 replay"异步损失小"一致;
5. **每题墙钟 C 最快**(285s vs A 352s,快 ~19%)——尽管 summary 只是平手,report 段的 2x 直接压缩了单题关键路径。
6. TTFT:C 与 A 持平(3.75s);D/E 略高(4.16/4.69s)。

⚠️ 口径注意:各配置生成内容不同(输出长度/走向随配置变),墙钟还含搜索缓存/embedding 等非 LLM 时间,故**以 tok/s(长度归一)为主、墙钟为辅**;n=10,单题级差异看方向不看小数。

### 3.7b e2e 六格同天重跑(7/5,e2e5_*)——终版表,悬念全部落地

同天同条件(10 题 local,题内并发 3),decode 纯段 tok/s,加速比对同天 A:

| 配置 | 墙钟中位 | 10题总min | SUM tok/s | REP tok/s | TTFT | 接受率 |
|---|--:|--:|--:|--:|--:|--:|
| A 原生(纯graph,异步开) | 324s | 57.9 | 25.5 (1.00x) | 40.5 (1.00x) | 3.76s | — |
| C suffix k24 混合 | 316s | 45.6 | 26.1 (1.02x) | 80.1 (1.98x) | 3.78s | 40.2% |
| **Ceager suffix k24 全eager** | **278s** | **41.1** | **26.4 (1.03x)** | **95.1 (2.35x)** | 4.30s | 40.5% |
| CeagerK8pad k8+pad 全eager | 254s※ | 41.3 | 23.7 (0.93x) | 84.5 (2.09x) | 4.00s | 13.0% |
| D ngram k8 混合 | 316s | 44.9 | **18.4 (0.72x)** | 92.1 (2.28x) | 4.09s | **4.6%** |
| Deager ngram k8 全eager | 301s | 44.5 | 23.1 (0.90x) | 94.1 (2.33x) | 3.64s | 10.9% |

※CeagerK8pad 墙钟中位为题间方差噪声(总时长与 Ceager 打平),以 tok/s 为准。

**悬念落地**:
1. **Ceager 冠军坐实**:墙钟最快(比 A 快 14%)、SUM 唯一不亏、REP 最高(2.35x)——"并发下全 eager 吃亏"未发生;7/4 与 7/5 两轮跨天复现(A 锚点 25.3/40.6 vs 25.5/40.5,漂移可忽略);
2. **代价唯一且定位干净**:TTFT 4.30 vs C 3.78(+14%)——C 与 A 的 TTFT 相同(3.78/3.76),这 0.5s 纯是 enforce-eager 的 prefill 账单,不是 suffix 的;
3. **两刀合一被否**:CeagerK8pad 两段均输给 Ceager(pad 洪水:draft 3.6×、接受率 40.5%→13.0%),k8/pad 只属于混合模式;
4. **ngram summary 灾难同天复现**:D 0.72x(接受率 4.6%、90 万 draft 洪水);全 eager 救回一截(0.90x)但救不了满发机制;
5. **推荐部署配置(生产)**:`suffix + --enforce-eager`(k 默认、不 pad)——replay/e2e 双口径、跨两天四轮数据一致的冠军;TTFT 敏感负载需注意 +14%。

---

## 四、还没跑完的实验 + 为什么要跑

| 实验 | 状态 | 为什么要跑 |
|---|---|---|
| ~~TP4-40条 全链~~ | ✅ 已全部完成 | 主口径已升到 40 条,含 2×2(C/Ceager/D/Deager)与 k×pad 扫点(Kpad16/24) |
| ~~固定税定位~~ | ✅ 已定案(经一次订正) | 混合边界+异步关,见 [how-we-found-the-fixed-tax.md](how-we-found-the-fixed-tax.md) |
| ~~node 级 trace~~ | ✅ 已完成(读数修正过) | FFN 现形填满空隙(34k gemm+8.5k silu,活跃 91.5%)——"gap≠空转"拿到直接影像;但 node 开销会把气泡转成忙碌,**不能定量气泡**("88μs/层"系 n=1 误读已撤回),气泡定量以差分为准 ~130μs/层,见 how-we-found §6.3 |
| ~~nsys batch=3 机制证明~~ | ✅ 已完成(三中二,①证伪原猜想) | ②flash_fwd 10.8→22μs ✓、③GraphLaunch 恒64/步 ✓,**但①gap 455→447μs 没缩**——"气泡被GPU忙碌挤掉"被否;真机制=**每步成本恒定(33.2→33.5ms 伺候3路)+纯摊薄**,每token税÷3。见 how-we-found 追问区改写 |

> **三指标怎么读(判决版:三中二,①脱靶——原猜想被证伪)**。原猜想:"GPU活变多+CPU杂务不变→每层气泡被 GPU 忙碌从时间线上**挤**出去"(即 GPU 忙的窗口长到盖过 CPU ~460μs/层的节拍,GPU 不再饿着),预言 gap 缩到只剩 FFN 的 ~380-400μs:
> ①**gap**(=隐身FFN + 真气泡之和):**实测 455→447 没缩**——GPU 虽变忙但远没到 CPU 节拍的天花板,气泡原地不动,"挤掉"证伪;
> ②**flash_fwd 时长**:10.8→22μs ✓——"GPU 每层活变多"是真的,只是量不够形成挤力;
> ③**GraphLaunch 次数/步=64 恒定** ✓:排除"batch 下 vLLM 换了执行路径、税本身消失"的替代解释。
> **判决**:②③中、①脱靶 → 每层微结构(含气泡)在并发下原样不动;真机制 = **每步成本恒定(33.2→33.5ms 伺候 3 路)+ 纯摊薄**(每 token 税÷3),不是"挤掉"。
| **e2e5 六格重跑**(10 题,eager 格先行,含 CeagerK8pad 两刀合一) | 🔄 跑动中(Ceager 进行中) | 核心悬念:Ceager 的 replay 战绩(1.00x/2.29x)在并发=3 下是否保持;CeagerK8pad 测固定税+按量税叠加,并顺带出 replay40 |

| ~~batch 扫描~~ | ✅ 已完成(7/5 晚) | **gap 单调缩:b1 8.1 → b2 5.5 → b4 4.5 ms/token(-44%)**——"并发下税差收窄"效应实锤(两配置接受轨迹对称,gap 列干净);机制经 nsys b3 判明为**摊薄主导**(每步成本恒定,"挤掉/掩盖"已证伪);与 e2e summary 仅差 2% 定量吻合 |

**残量终审升级路线(若今晚 C0piece/C0eager 后仍有未归因,按序升级;均未排队)**:
- **第0步(免费)**:挖现成 cpusample_C0.nsys-rep 的 CPU 采样栈(7/5 中午采过、只读了 API 就转向,可能已躺着答案);
- **第1步(一枪流,~6min)**:nsys 全开打 C0eager——`--trace=cuda,nvtx,osrt --python-sampling=true` + NVTX 步切分。钱只可能藏四处(GPU kernel/API 内部/Python 代码/内核态等待),之前三轮 nsys 都关着 CPU 采样(半盲);全开后四处同屏,5-6ms/步量级无处可藏;
- **第2步(兜底,数学上不可能漏)**:worker+engine 双进程闭合秒表,一步切成首尾相接具名段,Σ段≡步长,残量必落具名段,哪段大再切哪段,两轮内点名。

**census 分析计划(预注册,7/5 夜;数据=census_Ceager/census_A 40题+SUFFIX_TRAJ 轨迹)**:
- **A1 最优配置加速表**:Ceager vs A 同天,SUM/REP 分开:聚合加速比+分桶普查(>1.05/平/<0.95 比例)+**逐请求命中率分布**(轨迹 Σacc/Σdr,req_id join tag);
- **A2 命中率×题型**:按题聚合 summary 命中率排行,关联 ①prompt 长度 ②照抄输入率(harvest) ③兄弟互抄可得性(调用时同题已完成输出量);产出"易抄题型"画像(头尾各5题定性命名)→ 草稿源创新选题依据;
- **A3 早 miss 门控**:①P(低命中|前 N 个有草稿步全拒),N=2..8 的 precision/recall;②纯时间策略模拟(省=砍掉草稿位×现场单价[步时~批组成回归],亏=误杀接受×步时)→ **N-每题净省秒曲线**+推荐 N;③迟到爆发占比(命中率曲线后段翘头)→ 定策略形态:一刀切/可重开/叠 ctx 条件。
这三项对后续创新点(自适应投机开关+题型感知草稿源)是决定性输入。

**候选实验(等用户拍板,均不在队列)**:

1. **suffix 强开异步**——(2026-07-05 晚降级,基本可砍)原动机"直测异步值多少"已被 D vs E 抢答:ngram 混合下异步开/关只差 **0.5ms/步**(31.4 vs 31.9,与 A/B 的 0.5ms 一致),"ngram 免税靠异步"的旧说法已在 how-we-found §7.3 修正为"税照交、靠接受摊薄;suffix 附加费大头是自家 CPU 机件 1.6ms 而非异步"。唯一残余问号:并发下异步价值是否上升(e2e5 同天 A/B 无法测,7/4 A vs B 墙钟差 ~5%)——除非并发复核显示显著,否则不值得做 hack。
   **怎么兼容(不是光 hack assert)**——不兼容有两条依赖,各补一刀:
   - **依赖②数量**:异步要提前建好下一步输入缓冲,必须预知每请求占几个槽;suffix 变长草稿(0~k)不可预知。vLLM 自己钉了铁证:`disable_padded_drafter_batch=True` 与 async 直接互斥([vllm.py:707](../../../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/config/vllm.py#L707)),即异步硬性要求 padded drafter batch。**解法=k8+pad8 定槽**(每步恒 1+8,dummy 必被拒无损)——已有现成插桩;
   - **依赖①内容**:suffix propose 吃 CPU list 的采样结果(`sampled_token_ids: list[list[int]]`),要 GPU→CPU 同步;白名单方法的草稿链全程 GPU 张量(`valid_sampled_token_ids_gpu`),CPU 可蒙眼跑——这是它们能进白名单的结构原因。**实验档解法=worker 侧 propose 前强制同步一次拿 CPU ids**(sync 模式本来每步也做,非新增开销),引擎侧调度重叠照常生效,足够直测异步价值;生产档(草稿滞后一步/树挪 worker)等实验档证明收益够大再做。
2. **e2e 40 题大样本**(此前用户指示暂停)——e2e5 六格出方向后,用大样本把"生产口径推荐配置"钉死(并发 1~9、题间方差、T3 互抄红利都只有大样本能压住)。
3. **固化最优配置**——若 CeagerK8pad(全 eager+k8+pad8)在 e2e 胜出,部署侧只是"启动参数 `--enforce-eager` + env `SUFFIX_PAD_DRAFTS=8` + k=8 配置"三行改动,不动 vLLM 代码;跑一轮验收即可上生产。

跑完 TP4-40 全链后,能解出的全部量:异步主效应(A−B)、投机时异步效应(D−E)、纯固定费(C0−A)、验证载荷(C−C0)、GPU vs CPU drafter 框架费差(D−A vs C−B)、graph 价值(F−A)、padding 价值(G−G0)——每个桶两条以上独立通路交叉验证。

---

## 五、对优化优先级的冲击(基于已有数据)

1. **降税三件套(增量匹配/跳步/C++下沉):判死** —— 它优化的 propose 只占 0.16ms;
2. **EAGLE"拿回异步"的卖点:贬值但没死透** —— 无投机下异步只值 ~2%;但投机在场时"异步关"是混合边界税露出来的原因之一(见 how-we-found §7.3),"拿回异步"对 suffix 类仍有结构意义;
3. **头号标的(订正,已实测):suffix 全 eager 去掉 graph+eager 混合边界** —— ~~让 spec 注意力走 graph 捕获~~(旧标的基于"29ms空转"误判,撤回);Ceager 实测 replay summary 0.83x→**1.00x**、report 1.85x→**2.29x**,只需 `--enforce-eager`;**待过 e2e 并发关**(e2e5 已排队,并发下 batch 变大、graph 价值上升,全 eager 可能吃亏);次选:让 suffix 进异步白名单(系统改造,上限更高);
4. **部署捷径:suffix k=8+pad(按量税甜点)** —— 再订正:40 条主口径上 k8>k24 稳定成立(§3.5),k×pad 扫点单调;k=8 砍按量税(5.2ms)+pad 小赚(2ms),但削不到固定税,须与第 3 条叠加;
5. **ngram_gpu 单 server 方案:降级** —— replay 双赢,但 e2e 实测 summary 净亏 0.75x(4.9% 接受、95 万 draft 洪水,并发下垃圾草稿付真钱,§3.7);report 仍 2.22x。只适合 report-heavy 负载。

## 相关
- 实验计划:[tax-split-experiment-plan.md](tax-split-experiment-plan.md)(六配置详解、六件事机制)
- 优化方案:[optimization-round2-analysis.md](../explore-idea/optimization-round2-analysis.md)
- 研究方向:[research-directions-copy-speculation.md](../explore-idea/research-directions-copy-speculation.md)
- 逐条物证:`~/modify-code-runs/suffix-spec-decode/tax-split/interim-results.md`
