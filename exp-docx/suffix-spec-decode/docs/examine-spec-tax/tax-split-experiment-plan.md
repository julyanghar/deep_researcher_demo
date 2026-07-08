# 税成分拆分实验计划(§3.9 落地版):异步调度损失 + propose 分段占比

> 目标两问(用户提出):①**关异步调度导致多少性能损失、为什么**;②**每步 propose 各环节的时间占比**。
> 它裁决的事:18% 固定税里"关异步"(降税三件套砍不到、EAGLE 能拿回)和 "propose CPU"(三件套能砍)各占多少 → 直接决定 [optimization-round2-analysis.md](../explore-idea/optimization-round2-analysis.md) §3.5/§3.7 的优先级。
> 代码部分已按 /modify-code 完成并通过 CPU 侧验收,审查报告:`/home/yilin/modify-code-runs/suffix-spec-decode/tax-split/review-report.md`。

## 一、一个影响旧结论的前提发现

探查 server 日志实锤:**旧对照里 baseline 的异步调度一直是开的、suffix 一直是被强制关的**(vLLM 规则:无投机默认开,suffix 投机自动关)。所以过去说的"suffix 固定税 ≈18%"其实是"关异步 + propose CPU + 验证开销"的**混合值,从未拆开**。本实验补上这个对照缺口。

## 二、实验矩阵:三配置 × 两口径

**配置全家福**(随实验推进从 3 个扩到 6 个;差分逻辑:相邻两个只差一件事):

| 投机 \ 异步调度 | 开 | 关 |
|---|---|---|
| **无投机** | **A**(baseline 默认) | **B**(+`--no-async-scheduling`) |
| **ngram_gpu**(GPU 检索投机,lookup 2~6/k=8) | **D** | **E**(+`--no-async-scheduling`) |
| **suffix**(CPU 检索投机,默认参数+插桩) | ✗ 结构禁止(白名单强制关) | **C** |
| **suffix 空草稿**(`max_spec_factor=0`) | ✗ | **C0** |

各配置一句话:
- **A** 锚点(vLLM 默认:无投机时异步自动开);
- **B** 只关异步 → **A−B = 无投机时的纯异步损失**(实测 ~2%);
- **C** 正主(suffix) → **B−C = suffix 全部额外成本 − 投机赢回(winnings)**;带 SUFFIX_TIMING 分段插桩;
- **C0** suffix 但预算系数=0 → **永远出空草稿**:投机机制全程在岗(每步照样 propose、走 spec 引擎路径、每步同步取草稿),但从不下注、从无验证。**C0−B = "光开着投机机制"的纯固定费**(同步路径+propose+记账),**C−C0 = "真下注"的代价**(验证+元数据+按量费)±winnings——把 +17ms 固定费再拆两半;
- **D** ngram_gpu(prompt-lookup 的 GPU 张量版,异步白名单成员)→ 双重身份:①"单 server 投机+异步兼得"的部署候选;②**GPU-drafter 判别器**:D 若没有 C 的 +17ms 固定费 → 它是 CPU-drafter 同步特有;D 若也有 → vLLM spec 管线通病(上游);
- **E** ngram_gpu+显式关异步 → **D−E = "投机在场时"的异步损失**(suffix 开不了异步,这是测该量的唯一干净途径;与 A−B 之差 = 交互项)。

**两口径**(每配置各跑一遍,v2 的教训:投机解码评估必须端到端):

1. **replay 10 条**(workload_smoke.jsonl = 8 summary + 2 report,串行 batch=1):干净口径,直接读 **ms/token**。**batch=1 的含义(Q 澄清)**:一"条"=一次 LLM 调用(生成一个 summary 或一个 report),replay 脚本逐条发、等完再发下一条 → server 的 decode batch 任意时刻恰好 1 个序列;(A/B 无投机,一步=一 token,B−A 的每步差就是被暴露的 CPU 段大小——回答"为什么");
2. **端到端 10 题**(run_drbench --n 10,l0v2 同款,local 搜索缓存):生产口径。--concurrency 1 是**题级串行**(一次一道题),但**题内** 3 researcher × 3 并行 subquery 扇出 → 最多 9 个 LLM 调用同时在生成,server 的 decode batch 在 1~9 之间波动(v2 实测 summary 段峰值 9 路)。验证"税在并发下被稀释/放大"到什么程度——注意异步损失随 batch 的走向不是必然缩小:CPU 排班工作量随 batch 近似线性涨,而小 batch 区间 GPU 步时长(memory-bound)只缓慢增长,**CPU/GPU 占比在 batch 1→9 可能反而变大,异步损失可能高于 batch=1 的 2%**——B_e2e 就是这个问题的判决。

同一配置的 replay 和 e2e **各用全新 server**——防 C 的全局后缀树把 replay 的输出背下来污染 e2e(热树坑)。共 6 台 server 顺序起。

**每台 server 起来后先过预检断言**(不过就停,不烧卡):A 的日志必须有 "Asynchronous scheduling is **enabled**",B 必须有 "**disabled**",C 必须有 "Async scheduling not supported with suffix"。

## 三、插桩(回答"propose 各环节占比")

`SUFFIX_TIMING` 环境变量门控(默认关=零行为变化,CPU 侧已验证 patch 前后草稿逐位一致),两处:

- `suffix_decoding.py` propose 分五段累计计时:**建树(start_request)/喂树(add_response)/取 pattern/查树猜稿(speculate)/收尾(stop_request)**,外加 propose 总时长;每 200 次 propose 打一条 `[SUFFIX_TIMING]` 累计日志到 server log;
- `cache.py` speculate 里**本地树 vs 全局树**分别计时,打 `[SUFFIX_TIMING][cache]`。

分析时:各段 ms/步、占 propose 比例、propose 占整步比例(与 B→C 的 Δms 交叉验证)。

## 四、运行机制(这台机器抢卡凶,全自动化)

runner(`run_tax_split.sh`,nohup 常驻):

- **每 5 秒**查一次 {1,2,3,7} 空闲:4 张空 → 10s 双查确认 → **TP4** 原配置;只有 2~3 张空 → 宽限一轮 → **TP2** 变体(tp=2/maxlen=24576/util=0.90,已备好);
- 起 server 前**再验一次**本组卡空闲(第一次尝试就是栽在"检查后 30 秒被人插队"的 OOM 竞态上,6 台全灭);每阶段失败自动**重试一次**;
- 顺序:A_replay → A_e2e → B_replay → B_e2e → C_replay → C_e2e;
- 结束(无论成败)**保底留一台 server 占卡**(用户要求:启动后不释放);
- 监控:`tail -f /home/yilin/modify-code-runs/suffix-spec-decode/tax-split/tax_status.txt`。

## 五、分析计划(实验完成后)

1. **异步损失** = B vs A:replay 口径给配对长度归一 char/s 比 + **Δms/token 绝对值**(=每步被暴露的 CPU 段);e2e 口径给墙钟与逐题 decode 比(看并发稀释);
2. **suffix 剩余半边** = C vs B:全体聚合(净效应,含投机赚的)+ **低 harvest 子集**(纯税视角;冒烟集里 i56/i38 两条 harvest≈2%,样本小只看方向);
3. **propose 分段占比表**(§三的日志)+ 与 2 的 Δms 交叉验证;
4. **复核**:C vs A 应与旧"18% 税"同量级(同为混合口径);
5. **决策**(§3.9 规则):异步占大头 → 降税三件套降级、EAGLE 升级;propose 占大头 → 三件套按计划做。

## 六、口径与风险(诚实小字)

- **n=10 是方向性冒烟规模**(用户定):结论看方向和量级,不下精细百分比;若拆分结果两半接近,需加样本再判。
- **TP2 与旧 TP4 数字不可比**(绝对速度不同);A/B/C 只要同一 TP 内互比,拆分结论成立。若中途 TP 变了(不会:TP 在 wait_gpu 决定后锁死),数据作废。
- temp0 两次运行输出有漂移 → replay 用同 index 配对长度归一(坑③口径)。
- e2e 的 C 配置含全局树跨请求收益(v2 已证),所以 C vs B 的 e2e 差 = 税 − 全局树红利,解读时分开说。
- TP2 下 e2e 的 KV 预算紧(并发 9 路可能触发抢占),三配置同等受影响,组内比值仍有效。

## 七、当前状态(2026-07-04)

- ✅ 插桩完成,CPU 验收 A1~A5 全过(patch 前后行为逐位一致、打点自洽、门控 AST 审计 16/16、脚本 DRYRUN 正确);
- ✅ 第一次运行的事故已归因并修复:GPU 空闲检查与 worker 初始化之间 ~30s 窗口被外部任务插队 → 全部 OOM;已加双查+起前复验+重试;
- 🕐 **runner 在岗**(5s 轮询),当前 {1,2,3,7} 被外部任务占用,等窗口自动开跑;
- 产物目录:`/home/yilin/modify-code-runs/suffix-spec-decode/tax-split/`(blueprint.md / review-report.md / run-final.log / 两个 patch / 冒烟 workload / TP2 配置)。


## 八、中期发现:税的大头是"投机步骤的框架+验证开销"(2026-07-04 晚,详解)

实测拆账(replay,batch=1):异步 ~2%(e2e 口径 ~3%)、propose 总共 **0.17ms/步 ≈0.7%**(speculate 0.077/add_resp 0.047/start_req 0.018/其余 0.03)——两个原嫌疑人都不是大头。C/B(summary)= 0.898,其中还含 winnings 正贡献,故**框架+验证开销 ≈ 7%~15%**(winnings 拆分后钉死)。

**一个 spec 步比普通 decode 步多做的六件事**(⏱=量级估计;[固定]=逐步固定,[按量]=随草稿 token 数):

1. **变长记账[固定]**:scheduler 处理 spec_token_ids(排进 token 预算、KV slot 按最坏预留、验证后回填接受数)——普通步每请求恒 +1,spec 步 0~25 变长,Python 数据结构操作,⏱ ~百 μs;
2. **spec metadata 构建[半按量]**:_calc_spec_decode_metadata 为 [真+草稿] 每位置准备 logits 索引/位置/attention 元数据,多一套小张量拼装+拷贝,⏱ 百 μs~ms;
3. **前向多算草稿位置——KV 读是被低估的大头[按量]**:"验证免费搭车"的老论断只算了权重读取(不随位置数变),**漏了 attention 的 KV 流量随位置数线性涨**:每个草稿位置都要扫全部历史 KV——prompt 1 万 token 时,一个位置的 KV 读 ≈2.5GB(全 TP)≈0.2~0.3ms;平均 2.9 个草稿/步 → **~1ms/步 ≈4%**,且 prompt 越长越贵(这或许还解释了为什么长 prompt 的 summary 税感更重)。logits 多行只是 GEMM 增量(lm_head 权重反正读一遍),免费;
4. **rejection sampler(批改老师)[固定+微按量]**:投机的**安全底线**——草稿是猜的,不能直接当输出,必须有人判定"哪些确实是大模型自己也会说的"。做法(temp0):一次前向反正算出了每个位置大模型自己想说什么(argmax),批改老师把草稿逐位和 argmax 比,第一个不对处斩断、其后全弃、用 argmax 顶上;全对再奖 1 个 bonus。名字来源:温度>0 时要按概率比值做随机接受/拒绝(统计学的拒绝采样),temp0 是退化特例。**无损性就是它保证的**。成本:一个小 kernel+几次比较,⏱ 几十 μs,小头;
5. **每步取草稿跑腿费(原名"无异步路径的显式同步",该名有误导——见 §八.2 澄清)[固定]**:EngineCore(指挥部,CPU 进程)和 GPU worker(车间)是**分开的进程**。异步开时指挥部发完指令不等结果、直接排下一步;开了 suffix 后草稿产在 worker 侧、指挥部**必须拿到草稿才能排下一步** → 走 core.py:413 的同步分支:**每步结束后显式向 worker 要一次草稿(跨进程 RPC 一问一答:序列化+socket+进程唤醒),纯串行插在两步之间**。GPU-drafter(EAGLE/ngram_gpu)无此环节(草稿躺在 GPU 张量里,下一步直接用,指挥部不必"拿到"它)。⏱ 毫秒级×每步一次——**当前 +17ms 固定费的最大嫌疑**,C0 称重、D 判别;
6. **变长输出回传[固定]**:sampled_token_ids 从"每请求 1 个"变"1~k+1 个"的 ragged 结构,GPU→CPU 拷贝+Python 处理,⏱ 百 μs。

**怎么钉死(已列入分析计划)**:①逐请求回归:C 相对 B 的每步超时 = 截距(固定框架费)+ 斜率×草稿数(按量费),再加 prompt 长度×草稿数交互项验证 KV-读假说;②D/E(ngram_gpu 开/关异步)与 C 共享同一 spec 执行路径、不同 proposer——D−A/E−B 分离出"GPU-drafter 的框架+验证费"做对照;③若按量费显著 → score 地板复活重估(垃圾草稿的真实成本比原假设高)。

**优先级冲击**:降税三件套(§3.5)判死(propose 无肉可砍);优化对象转向"减少垃圾草稿位置"(按量费)与 vLLM spec 步骤的框架效率(上游)。


### 八.1 关键实测更新(2026-07-04 深夜):固定费 ≈ +17ms/步,与草稿量无关

同日 B vs C 逐条对照(replay,10 条):**B 每步 24.3~25.4ms 极稳,C 每步 39.0~44.1ms——suffix 每步固定多花 +15~19ms(≈普通步的 70%),且与草稿量(1.0~1.8 稿/步)基本无关**;旧 80 条回归截距 15.2ms、草稿斜率仅 0.09ms/稿,跨批次互证。**结论:三桶里"逐步固定费"是绝对大头,按量费(KV 读/验证)在当前草稿量下很小。**

- 最大嫌疑:⑤ CPU-drafter 每步同步取草稿的路径(EngineCore↔worker RPC)+ spec 记账固定部分;
- 推论:①report 的 3.3x 步数压缩被固定费吃成 2.19x——干掉它 report 直奔 3x+;②score 地板维持死刑(垃圾草稿边际成本确实小);③**D/E 是判别器**:D(ngram_gpu,GPU-drafter)若无此 17ms → CPU-drafter 同步特有("搬上 GPU"价值大涨);若也有 → vLLM spec 管线通病(上游);
- **C0 配置已排队**(suffix + max_spec_factor=0,永远空草稿,spec 机制全开不下注):C0−B = 同步+propose+记账的纯固定部分;C−C0 = 验证+元数据部分。
- 执行链:C_e2e(跑动中)→ D(replay+e2e)→ E(replay+e2e)→ C0(仅 replay)。replay 均为 10 条 smoke 集(用户确认足够);e2e 均为 local 模式 10 题。


### 八.2 D 初步结果 + 术语勘误(2026-07-04 深夜)

D_replay 落数:**每步 A 24ms → D 31.5ms → C 41.5ms**;净速度 SUM D/A=1.026(summary 翻正)、REP D/A=2.487(超过 suffix 的 2.15);D 接受率仅 16% 但执行便宜。初步解剖:**通用 spec 管线 +7.5ms/步(D−A,含重验证载荷),CPU-drafter 额外 ≈+10ms/步(C−D)**。
**术语勘误(重要)**:"C−D 的 10ms"不是"异步调度损失"——A−B 已证异步**重叠的价值**只有 ~0.5ms/步(排班段本来就短);10ms 是 CPU-drafter **凭空多出的串行工作**(跨进程取草稿 RPC),哪怕异步开着这趟跑腿也得发生。两个量概念不同、互不矛盾。GPU-drafter 免此费的原因:草稿不用回 CPU。中期汇总:tax-split/interim-results.md;六件事直测待 C-prof。

## 相关

- 方案定位与决策规则:[optimization-round2-analysis.md](../explore-idea/optimization-round2-analysis.md) §3.5/§3.9;六问判定:[spec-optimization-six-questions.md](../spec-optimization-six-questions.md)
- 旧实验主结论与三大测量坑:[summary.md](../summary.md);端到端判决:[l0-l3-v2-results.md](../l0-l3-v2-results.md)
- 机制文档:[../../18-vllm-suffix-decoding.md](../../../18-vllm-suffix-decoding.md)
