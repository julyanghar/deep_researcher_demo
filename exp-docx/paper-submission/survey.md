## survey:dr-agent-characteristics
deep research agent 系统层特征文献盘点（明确 vs 空白）。①调用图：Anthropic 博客给出 orchestrator-worker 结构（lead+3-5 并行 subagent+引用 pass）；GPT Researcher 文档给 planner/researcher/editor/writer/publisher 五角色、~3min/~$0.1/次；TokenCake(2510.18586) 以 Deep Research 为评估应用但只定性"agent 少、依赖链深"。②上下文累积：Agentic AI Workload Characteristics(2605.26297) 测得首轮输入数万 token、后续每轮仅增数百 token、缓存生效后 decode-dominated、双长尾（多轮型 vs 大上下文型）——但 benchmark 不含 deep research。③输入输出重叠：LLMA(2304.04487) 证明 RAG 输出大量 span 抄自检索文档、复制校验 2x 无损加速；deep research 报告的照抄比例无人量化（空白）。④output-becomes-input：SuffixDecoding(2411.04975) 证 agentic 输出自重复、后缀树投机至 5.3x；DroidSpeak/KVCOMM 利用跨 agent 复用，KVCOMM 报 >70% KV 复用、TTFT 430→55ms(7.8x)。⑤长输出占 e2e 比重：仅 OpenAI 官方"5-30 分钟/查询"、ADRA-Bank 测 per-task latency+token；无人做"搜索等待 vs prefill vs 报告 decode"分解（空白）。⑥共享前缀/KV 机会：Preble 首个共享 prompt 工作负载研究（prompt 比 output 长 37-2494x、85-97% token 跨请求共享）；SGLang 命中 50-99%；CacheBlend ~15% 重算换 TTFT 2.2-3.3x；TokenCake 测峰值 18.5% GPU KV 池被空转 agent 占用；deep research 子代理实际共享率无实测（空白）。⑦token 量级：Anthropic 报 agent≈4x chat、多agent≈15x chat。⑧最大空白：TraceLab(2606.30560) 只刻画 coding agent，deep research 至今无公开 trace 级刻画（逐调用 timing、上下文增长曲线、内容重叠率、decode 占比、扇出统计）——直接的研究机会。以上①②④⑥数字均为文献明确；③⑤⑧的"空白"判断是基于检索未见的推断。
关键数字:
- Anthropic 博客: agent ≈4x chat token, 多agent研究系统 ≈15x chat; lead+3-5 并行 subagent (anthropic.com/engineering/multi-agent-research-system)
- Agentic AI Workload Characteristics (arXiv:2605.26297): 首轮输入数万 token、后续轮仅数百 token, 缓存后 decode-dominated, 双长尾
- Preble (arXiv:2407.00023): prompt 比 output 长 37x-2494x, 85%-97% token 跨请求共享
- KVCOMM (arXiv:2510.12872, NeurIPS'25): 多agent >70% KV 复用率, 5-agent TTFT 430ms→55ms ≈7.8x
- SuffixDecoding (arXiv:2411.04975): agentic 输出自重复, 后缀树投机最高 5.3x 加速, AgenticSQL 2.9x 吞吐
- CacheBlend (EuroSys'25 best paper, arXiv:2405.16444): 非前缀 KV 复用 ~15% token 重算, TTFT 降 2.2-3.3x
- SGLang RadixAttention (arXiv:2312.07104): 多调用程序 cache 命中率 50%-99%
- TokenCake (arXiv:2510.18586): 峰值 18.5% GPU KV 池被空转 agent 占用; Deep Research 应用 e2e Parrot 3.5-4.2ks vs TokenCake 496-646s
- LLMA (arXiv:2304.04487): RAG 输出照抄检索文档 span, 复制校验得 2x 无损加速
- OpenAI Deep Research 官方: 5-30 分钟/查询; GPT Researcher: ~3 分钟、~$0.1/报告
- 空白: TraceLab (arXiv:2606.30560) 仅刻画 coding agent; deep research 无公开 trace 级系统刻画(内容重叠率/decode占比/上下文增长曲线均无人测)

## survey:related-work-positioning
(a)判决路由模块——赛道已挤：SpecDec++(2405.19715)在draft hidden state上训acceptance头做动态草稿长度(2.04-2.26x)；BanditSpec(2505.15141)训练-free bandit在线选投机超参/方法；Not-a-Bandit/HedgeSpec(2510.20064,Amazon)full-information在线选drafter，实验称胜EAGLE-3和BanditSpec；AdaEDL(2410.18351)/DISCO/DSDE(2509.01083)做熵/小网络早停；CAS-Spec(2510.26843)级联自投机动态路由。确凿搜不到"用target hidden state预测本次调用输入→输出覆盖率来做per-call策略路由"的工作——剩余新颖性=信号来源(覆盖率预测)+路由retrieval/suffix类策略+agent负载定位，属增量。(b)agent轨迹特化EAGLE——无直接命中论文(确凿搜不到agent-trace-trained draft head已发表工作)，但通用位置全被占：DistillSpec(ICLR'24,10-45%增速)、Online Speculative Decoding(2310.07177,ICML'24,在线蒸馏适配query分布)、OmniDraft(2507.02659在线自适应drafter)、Draft-OPD(2605.29343 on-policy蒸馏治SFT平台期)、RL rollout上训drafter(2511.13841)；且Baseten/NVIDIA NeMo已把"自有数据训EAGLE-3头"做成工业教程，EAGLE-3.1(vLLM,2026-05)又把长上下文acceptance提2x。单独成文弱，须靠agent特有观察(跨调用重复结构/acceptance规律)撑。(c)decode-to-prefill KV reuse——直接命中：RelayCaching(2603.13289,2026-03 preprint)正是上游agent decode产生的KV给下游prefill复用(AutoGen/MetaGPT场景，AIME/coding实验)，机制叙事已被占但尚无venue；正式发表的最近邻KVCOMM(NeurIPS'25,>70%复用率)、DroidSpeak(NSDI'26跨模型分层复用)、CacheBlend(EuroSys'25选择性重算)都在prefill侧；2026 preprint还有TokenDance(2604.03143)/LRAgent(2602.01053)/QKVShare扎堆，另有负结果论文(2601.08343,judge任务复用伤质量)可引。顶会正式论文层面decode侧KV复用似仍空缺，窗口正快速收窄。整系统先例：Parrot(OSDI'24语义变量)、Teola(原语数据流图,venue本次未核实)、Ayo(ASPLOS'25细粒度编排)、Autellix(NSDI'26程序级调度,4-15x vs vLLM)、Murakkab(2508.18298)、Continuum(2511.02230 KV TTL)、FASER(2604.20503投机相位管理)——已占切口全在调度/编排/KV生命周期层。未被占切口：利用agent负载文本自相似性同时打prefill(KV复用)+decode(特化投机+覆盖率路由)的cross-layer联合serving系统，无正式顶会先例，是三件合体的最佳故事线；单件投稿则(c)被RelayCaching抢跑、(a)(b)只剩增量。未确认项：BanditSpec/Not-a-Bandit正式venue、RelayCaching是否已中会。
关键数字:
- SpecDec++ 训acceptance预测头做动态草稿长度，2.04-2.26x加速，比基线投机再+7-11%（arXiv 2405.19715）
- HedgeSpec(Not-a-Bandit) full-information在线选drafter，声称胜EAGLE-3与BanditSpec（arXiv 2510.20064, Amazon Science）
- DistillSpec 知识蒸馏对齐draft，较标准投机解码提速10-45%（arXiv 2310.08461, ICLR 2024）
- KVCOMM 多agent负载KV复用率>70%，NeurIPS 2025 poster（arXiv 2510.12872）
- RelayCaching 正是decode KV给下游prefill复用，2026年3月preprint尚无venue——组件(c)的直接抢跑者（arXiv 2603.13289）
- DroidSpeak 跨LLM KV共享已中NSDI 2026（arXiv 2411.02820）
- Autellix 程序级调度较vLLM吞吐提升4-15x，NSDI 2026（arXiv 2502.13965）
- EAGLE-3.1 长上下文acceptance length较EAGLE-3最高2x（vLLM blog 2026-05-26）——(b)的移动基线
- 确凿搜不到：hidden state预测输入输出覆盖率做投机路由的已发表工作；agent轨迹上特化训练EAGLE的已发表论文
- 整系统层面：Parrot(OSDI'24)/Ayo(ASPLOS'25)/Autellix(NSDI'26)/Continuum等全在调度编排层，prefill+decode联合的agent-aware加速系统在顶会仍空缺