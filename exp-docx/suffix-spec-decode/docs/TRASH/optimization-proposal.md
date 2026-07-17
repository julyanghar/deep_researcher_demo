> ⚠️ **过期警示**:本文档含已被 7/6 之后实测**推翻**的早期结论。最新准确结论以 [LATEST-CONCLUSIONS.md](LATEST-CONCLUSIONS.md) 为唯一权威——凡与之冲突的本文结论一律作废。

---

# suffix 投机解码优化方案(设计文档,未实施)

> 基于三份实测报告([summary.md](summary.md) / [overlap-replay.md](overlap-replay.md) / [per-request-speedup-variance.md](explore-idea/data-source/per-request-speedup-variance.md))的优化设计。只讲方案与取舍,均未动手;每个方案附"怎么验证成不成"。

## 背景一页纸(为什么要优化、优化什么)

实测事实(全部冷树、batch=1):
- **report 恒赚**:1.27x–2.46x,无一亏,平均 ~1.8x;
- **summary 两拨命运**:0.71x–2.56x,15/60 真加速、33/60 明显亏,聚合 0.93x;
- **固定税 ≈18%**:开 suffix 后每步都交(异步调度被关+查树),不照抄的请求纯亏 18%;
- **能预测**:token 级照抄量(harvest)与加速比 r=0.55(summary)/0.90(report),且题间可分;
- **约束**:vLLM 的 speculative-config 是 **server 级**,不能逐请求开关;只有 4 张卡。

优化空间一句话:**把"该开才开"做出来**——report 全开稳赚,summary 里挑出检索型的开、放过分析型的,summary 段就能从 0.93x 翻到最高 1.05x(完美路由上界)。

---

## 方案分层(从"今天就能上"到"研究方向")

### L0:全局开 suffix,默认参数——立即可做,净赚
**做法**:server 配置加一行 `speculative-config: '{"method":"suffix"}'`,参数全默认(实测 prob=0.1 就是最优点,别调)。

**净收益测算**(每题的 LLM decode 段,拿实测均值直接算)。⚠️ **强依赖工作负载画像**:workload 抽自 **online100(v1,单轮:6 summary/题)**;而 **online100_v2(3 轮迭代)是 18.1 summary/题**(实测计数:v1 600 条/100 题,v2 1812 条/100 题),summary 权重差 3 倍,两种画像分开算:

| 画像 | summary 段 | report 段 | 合计 | 净变化 |
|---|--:|--:|--:|--:|
| **v1 单轮·实测**(6×14.7s→16.3s)| 88.2→97.8s(+9.6)| 80.1→43.4s(−36.7)| 168.3→141.2s | **净省 ~27s/题(~16%)✓** |
| **v2 多轮·实测**(18.1×13.1s→16.0s)| 237→289s(+52)| 134→69s(−65)| 371→358s | **净省 ~13s/题(~3.5%)≈勉强为正** |

(v2 行为 2026-07-02 实测:60 summary 按轮分层 + 20 report 从 online100_v2 harvest 重放,冷树 batch=1。)

**v2 实测的两个新事实**(比早前估算更极端、方向相反地部分抵消):
1. **v2 的 summary 比 v1 伤得更重:x0.81**(v1 是 x0.955),且三轮一致(R1 0.79 / R2 0.83 / R3 0.81,无轮次漂移)——多轮 summary 的照抄画像更碎,亏 ~19%;
2. **v2 的 report 赚得更多:x1.87、单条省 65s**(134→69s;v1 单条才省 37s)——27 份 summary 喂进 prompt,报告更长、可抄更多,完全符合预测。

**结论**:v1 画像全开净赚 ~16%;**v2 画像全开只剩 ~3.5%,而且这是 batch=1 串行的上界口径**——端到端真实并发(题内 3 路)下投机收益还会缩水,**净收益很可能归零甚至转负**(正在跑的 L0-V2 端到端实验直接回答这一点)。**多轮画像下,L3(把 summary 翻正)/L2(躲开)决定净收益正负,不是优化项而是前提。**
**两个附带红利**(实测数字是保守下界):①生产长跑 server 的**全局树会焐热**——重复/相似 query 越跑越快(我们当污染排除的 1.9x,在生产里是真收益);②demo 一题内 6 个 researcher 常引用重叠来源,全局树跨调用命中会比冷树测的多。
**风险与验证**:投机收益随并发缩水(batch 大了没闲算力捡)。上线前做**并发档位压测**(1/2/4/8 并发跑同 workload,找 suffix 收益归零的并发红线);若 vLLM 支持按 batch 大小自动关投机的开关则配上(需查当前版本 SpeculativeConfig,不支持就用压测红线控并发)。

### L1:按阶段分池——绕开"server 级配置"的最笨但稳的办法
**做法**:两套 server——report/writer 流量走开 suffix 的池,summary/researcher 流量走不开的池。demo 的 LLM 调用有统一入口和 tag,按 tag 切 base_url 改动很小(属改代码,走 /modify-code)。

**代价(这是主要问题)**:4 张卡切两池只能各 TP2。Qwen3-32B 权重 bf16 ~62GB → TP2 每卡驻 ~31GB,48GB 卡放得下但 **KV cache 空间大缩** → 并发能力/长上下文余量都变差;而且 summary(6 次/题)和 report(1 次/题)流量不均,静态分池必有一池闲。
**判定**:除非 L0 压测发现分析型 summary 的亏在真实负载里被放大,否则**不推荐**——为省 summary 那 ~10s/题把服务架构复杂化,不划算。它的真正价值是给 L2 当路由底座。

### L2:生成前预测器 + 请求级路由——核心研究方向
**目标**:一个轻量分类器,在**发请求之前**从 prompt 判断"这条会不会大量照抄"(harvest 会不会过 ~19% 打平点),决定路由到开/不开 suffix 的池。

**为什么可行**(已验证的三件事):
1. 目标信号题间可分(0.71x 和 2.56x 不是噪声,是两类题);
2. 与加速强相关(harvest r=0.55/0.90);
3. **标签免费**:online100/online100_v2 的 harvest 已存 1812 条 summary 的完整输入输出,离线跑 token 级 overlap 分析(现成脚本 `analyze_replay_overlap.py`)就能批量打标,**不花一分钟 GPU**。

**特征设计**(全部生成前可得,从 prompt 提取):
- 问题类型信号:疑问词/指令形态(找资源/名录/数据 vs 分析/评价/综述)、是否要求罗列;
- 来源结构信号:检索块里 **URL 密度**(URL 是天然长匹配)、列表/表格比例、块的平均长度;
- 压缩比代理:prompt 长度 vs max_tokens(要求高压缩 → 转述型 → 照抄少);
- 语言(中文罗列型在样本里偏多,作为特征而非规则)。

**模型与验收**:逻辑回归/小 GBDT 足够(几百维、几千样本);验收两层——①分类 AUC;②**模拟路由吞吐**:拿已测的 80 条 speedup 分布回放,路由后 summary 段吞吐应落在 0.93x(全开)与 1.05x(完美)之间、显著高于 1.0x 才算成。
**代价**:要 L1 的双池底座(或未来 vLLM 支持逐请求开关投机——值得盯上游);预测器打折后收益有限(summary 段最高 +12pp)。**判定:值得做,但优先级在 L0 压测之后**——若 L0 全开在真实并发下净赚依旧,L2 的边际收益就只有 summary 那 ~10s/题。

### L3:workload 共设计——让 summary 变得"可收割"(顺手的小实验)
**思路反过来**:不是让解码适应输出,而是让输出适应解码。给 RESEARCH_SUMMARY 的 prompt 加一条"**引用来源原文,不要改写;逐条列出,保留 URL**"。
- **速度**:输出从转述型变成转录型 → harvest 上去 → summary 从亏变赚(实测转录型 summary 已到 1.4-1.5x);
- **附带好处**:逐字引用本身可能**减少幻觉/改写失真**,对下游 report 的引用核验(FACT)也更友好;
- **风险**:summary 变长(decode 变多)、压缩/提炼质量下降,可能伤 report 质量。
**验证**:小样 20 题 A/B——速度看 harvest 与实测加速,质量用 kimi-k2.5 评委比 report 终稿(评委选择依据见既往实验教训)。改 prompt 属改代码,走 /modify-code。

### L4:降固定税——深水区,暂只记方向
18% 的税由两部分组成:异步调度被关(大头,vLLM 实现层面,上游才能解)+ 每步建/查树。可试的小实验:`max_tree_depth` 从 24 调小(12/8)看树开销降多少、report 收益掉多少——**唯一没扫过的参数**,一次 server 重启的成本。上游层面(suffix 与 async scheduling 兼容)投入大,除非要贡献 vLLM,不建议自己动。

---

## 与 KV-reuse 主线的关系(正交,可叠加)
LMCache blend/KV 复用省的是 **prefill(TTFT)**,suffix 省的是 **decode**——两者作用在请求的不同段,理论上正交叠加:report 这种"长 prompt(复用 summary KV)+ 长输出(照抄 summary)"的调用两头都吃。**但兼容性未测**:本实验用的是 no_lmcache 配置,`speculative-config` 与 LMCache connector 同开是否稳定、blend 改写 KV 后 suffix 的 prompt 树是否照常工作,列为集成测试 TODO(一次 server 起动 + 跑 20 条即可验)。

## 前置确认(2026-07-01 动手前核实的四件事)

**① suffix 可调参数的完整清单**(vLLM 0.18,grep 全库 `suffix_decoding` 核实,server 级、启动时定死):
`method="suffix"`、`num_speculative_tokens`(None→自动=tree_depth)、`suffix_decoding_max_tree_depth=24`、`suffix_decoding_max_cached_requests=10000`、`suffix_decoding_max_spec_factor=1.0`、`suffix_decoding_min_token_prob=0.1` —— **就这 6 个,没有更多**。
另有两个 arctic C++ API 有、但 **vLLM 没暴露**的旋钮(proposer 调用时不传,恒默认):`max_spec_offset=0`(草稿预算加项)、`use_tree_spec=False`(树状草稿,vLLM 恒走单路径)。要动它们只能改 vLLM 代码。

**② "生成的 token 是否加入树"的开关**:
- **全局树:有开关** —— `max_cached_requests=0` 即"生成的输出不缓存进全局树"(prompt 树保留)。**这个对比实验已经做了**(g0):关掉后 summary 0.955x→0.872x、report 1.841x→1.757x,即"缓存历史输出"贡献约 +8pp/+8pp。
- **本请求的局部树:没有开关** —— proposer 无条件把新 token 追加进局部树(`suffix_decoding.py:73` add_active_response→`cache.py:209`),配置层关不掉。想做"prompt-only 局部树"消融(隔离"抄自己前文"贡献多少)需要小改 proposer(加 env 开关跳过局部 extend),**属改代码,走 /modify-code**;这个消融值得排上——overlap 分析只量了"抄 prompt",summary 的命中里"抄自己"(重复格式/实体名)占比未知。
- 注意区分:pattern(匹配用的上下文)永远含已生成 token,开关只影响**树里有什么**。

**③ "异步调度被关"是 server 级、启动时决定,不是按请求**:配置校验时(`vllm/config/vllm.py:719-731`)只要 speculative method 不是 EAGLE/MTP/NGram-GPU 系,就全 engine 禁用 async scheduling(显式开 async + suffix 则直接报错拒绝启动)。**vLLM 0.18 没有任何按请求开关投机的能力**——这正是 L1/L2 需要双池路由的原因。

**④ workload 出处与画像修正**:本实验 workload 抽自 **online100(v1,单轮,6 summary/题)**,已抽验 3 条 expected 仅命中 v1;v2 是 3 轮×3 researcher×每 researcher 最多 3 sub-query,实测 18.1 summary/题(q1 达 27)。**L0 的净收益已按两种画像分开修正(见上表)**。

## 建议路线图
> ⚠️ 2026-07-02 更新:L0 与 L3 已在 v2 画像下实测完毕,判决见 [l0-l3-v2-results.md](l0-l3-v2-results.md)。下表为实测后的修订版。

1. ~~L0~~ **已实测,采用**:端到端(10 题×2 配置,题内并发 3)墙钟 **−29%**、decode x1.16;**summary 端到端不亏(x1.00)**——同题 18 份 summary 共享来源、全局树跨请求命中把冷树的 19% 亏填平。**replay 冷树口径对多智能体负载系统性过度悲观(3.5% vs 29%),评估投机解码必须端到端测。**
2. ~~L3~~ **已实测,当前形态否决**:机制成立(summary 吞吐 x1.49)但输出膨胀 +72% 吃光收益倒贴(净时间 +15%)、质量平手(kimi-k2.5 pairwise 4/3/2)、1/10 题 report prompt 撑爆。E2 证明 summary 已不亏 → L3 从"前提"降回低优先级。
3. **L2 预测器:搁置**——它的前提("summary 拖后腿,要躲开")被 E2 推翻。
4. 剩余值得做的一次性验证:更高并发档位压测、LMCache connector 兼容测试、(可选)prompt-only 局部树消融。

## 一句话
**v2 多轮画像端到端实测:直接全局开 suffix(默认参数)= 墙钟 −29%,summary 不亏、report 翻倍——冷树 replay 的悲观账被"同题跨请求全局树命中"翻案;L3/L2 不再必要。**
