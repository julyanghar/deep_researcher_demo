# EAGLE 仲裁 idea:照抄/分析双 drafter —— 评估与落地

> **缘起**:用户提出——deep research agent 的 LLM 调用分两类(照抄 / 分析);照抄用 suffix 投机(输出与输入重叠大、效果好),分析用 EAGLE-3;再加一个"判决模块"判这次调用是哪类。
> **一句话结论**:**框架对**(= 项目已有的 D2 混合仲裁),但——**判决模块不用训**(类型是免费的 tag,且 vLLM 投机是 per-server 不能 per-request 切 → 两 server + tag 路由即可);**真正承重、还没验的是"EAGLE 对分析类到底有没有用"**,该**先做零 GPU 上界**再决定投入,别从训模型起手。
>
> **🔴 2026-07-08 实测更新(见 [§实测判决](#-实测判决2026-07-08现成-eagle3-两类逐题全输-suffix分析臂赌注证伪)):现成 eagle3 head(中英都试)在本 workload 上 summary/report 逐题全面败给 suffix,分析臂核心赌注证伪。当前最优 = 单 suffix server,连"两 server 混合"都暂不值得。**

---

## 一、idea 原文(用户提出)

1. 任务分两类:**照抄类**(输出与输入重叠大)/ **分析总结类**。
2. 照抄类 → 直接用 **suffix 投机**(效果明显);分析类 suffix 效果差 → 用 **EAGLE-3**(自训或现成)。
3. 加**判决模块**:判这一次 LM 调用是照抄还是分析。可用 **Qwen3-32B + 小 LoRA** 或 **MLP** 训练,信号简单:**输入 = 大模型 hidden state,输出 = 输入与输出的覆盖率**。

---

## 二、站得住的部分(有数据)

| 主张 | 证据 |
|---|---|
| 两类分化真实 | report 照搬 91%、cov24 46%、suffix 命中 **59.5%**;summary 改写、cov24 0.4%、命中 **26.7%** |
| suffix 对照抄类好 | report e2e 2.29×;20 题 e2e 投机让 **report 段减半**(2176→1080s) |
| "按类换 drafter" 框架 | = 项目 D2「copy-state 感知混合仲裁」([explore-idea/research-directions-copy-speculation.md](explore-idea/research-directions-copy-speculation.md) §二) |

**所以分类和 suffix 臂是你的现成资产,不用重证。**

---

## 三、⭐ 关键基建约束 + 落地方案(核心结论)

**实测 vLLM 0.18**:
- **投机方法是 per-server**(`vllm/config/speculative.py` 的 `SpeculativeConfig` 是启动配置),`ChatCompletionRequest` **无 speculative 字段** → **同一 server 不能按请求切 suffix/eagle**。
- **EAGLE-3 原生支持**(`vllm/v1/spec_decode/eagle.py` + `extract_hidden_states.py`,`method=eagle3`)。
- **多 LoRA / per-request LoRA 支持**(`LoRARequest`、`config/lora.py`、`model_manager.py`)。

**→ 仲裁怎么落**:
- ❌ **不能**:单 server 内按请求切投机方法(vLLM 不支持,要改 proposer 源码=大工程)。
- ✅ **能且简单**:**起两个 server**(一个 suffix、一个 eagle3),**agent 按 tag 把请求发给不同 base_url**。report call → suffix server,summary call → eagle3 server。**应用层改个 base_url,零 vLLM 改动、零路由模型。**

**这条直接干掉"判决模块"**:agent 本来就知道这次是 report 还是 summary(tag),直接路由——**不需要训 LoRA/MLP 去"判"是不是照抄**。

---

## 四、逐点技术答复(对应用户 4 问)

**Q1 EAGLE-3 自训 vs 现成**
- vLLM 0.18 原生支持 `method=eagle3`,需要 **Qwen3-32B 的 eagle3 draft head** checkpoint。
- **先找现成**(EAGLE repo / HF 搜 Qwen3-32B eagle3);没有再自训(要 GPU+几天+数据管线,投入大)。
- **但先做 Q4 上界**,上界不够整条 EAGLE 臂不用建,找/训都免了。

**Q2 判决模块:LoRA / MLP / 现成小模型**
- **首选:都不用** —— 两 server + tag 路由已解决"判类型"。
- 若坚持要 tag 之上的学习型细化:**用 MLP,不用 LoRA**。LoRA 是改"生成"的适配器,做分类是牛刀杀鸡;eagle3 本来就 `extract_hidden_states`,hidden state 免费有,MLP 头最轻。现成小模型 = 多一次前向 + 和 tag 冗余。
- ⚠️ 项目已判:tag 之上逐实例细化头顶只 **±7pp**(见 [explore-idea/idea-verdicts-and-standing.md](explore-idea/idea-verdicts-and-standing.md) §一.3 题型感知调度),MLP 这点收益很可能不值。

**Q3 LoRA 能否 per-request 装载/卸载**
- **能**(vLLM 0.18 多 LoRA:按请求选、动态增删)。
- **但对本架构没用**:①真正想 per-request 切的是**投机方法**,那个不能切;②LoRA 是生成适配器不是路由器;③路由用 tag 就够。**LoRA 热插拔与本 idea 无关。**

**Q4 要不要先确认 EAGLE-3 投机效果**
- **必须,第一道生死闸**,两级:
  1. **零 GPU 上界**(先做):模拟器在录好轨迹上算"混合 oracle"= 每步 `max(suffix 实际接受, EAGLE 假设 μ≈4)`,看 summary 段能拉多高。撑不起 → 整条 EAGLE 臂枪毙。半天、零 GPU([explore-idea/proposal-mechanism-acceptance-uplift.md](explore-idea/proposal-mechanism-acceptance-uplift.md) §三路线3)。
  2. **真实测**(上界过了才做):现成/自训 eagle3 head 在 summary 生成上实测接受率,验证 μ≈4。

---

## ★ 实测判决(2026-07-08):现成 eagle3 两类逐题全输 suffix,分析臂赌注证伪

跳过零 GPU 上界,直接拿**现成 Qwen3-32B eagle3 head 实测**(两个都试:英文 [RedHatAI/Qwen3-32B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3)/speculators 格式、中文 [AngelSlim/Qwen3-32B_eagle3](https://huggingface.co/AngelSlim/Qwen3-32B_eagle3)/标准 eagle3 格式;TP2 卡5,7,`num_speculative_tokens=3`)。方法:**回放 20 题 run 里 suffix 处理过的同一批真实 prompt**(summary=RESEARCH_SUMMARY_TEXT、report=FINAL_REPORT_MARKDOWN),`/metrics` 差分 accepted/draft,与 suffix 同口径对齐。

**聚合接受率(AL = mean acceptance length = 每次前向出几 token,1.0=没投机):**

| drafter | summary(分析) | report(照抄) |
|---|---|---|
| **suffix**(现成免训) | **~26-32%** | **~57%** |
| eagle3·RedHat(英文) | 13.3% · AL 1.40 | 20.3% · AL 1.61 |
| eagle3·AngelSlim(中文) | 10.7% · AL 1.32 | 17.4% · AL 1.52 |

**逐题(AngelSlim,20 题):** summary 7.3–16.7%(中位 10.4%)、report 13.8–24.3%(中位 18.2%);suffix 逐题(traj 真实,3 题)summary 23.5–29.3%。→ **eagle3 单题最好成绩(summary 16.7%、report 24.3%)都低于 suffix 单题最差(summary 23.5%),20 题无一翻盘。**

**意外发现:中文 head 假设是反的。** `corr(中文占比, eagle3 接受率) = -0.60`(summary)、-0.33(report)——**越中文,eagle3 接受率越低**。起作用的是**任务分布/文体**(两 head 都通用 chat 训、跟"研究摘要/报告"文体差太远),**不是语言**;中文分析性长句反而是通用 head 最不擅长的。语言普查:summary 内容 ~90-100% 中文、report 混合(照抄英文源、中文才 25-65%)。

**结论**:现成 eagle3 head(中英都试)**在本 workload 上 summary/report 逐题全面败给 suffix,分析臂核心赌注证伪**。要救只能**自训本域 head**(summary 需 ~11%→>27%,3× 提升),是另一个大实验、payoff 不确定。**当前最优 = 单 suffix server**;连"两 server 混合"都暂不值得(eagle3 在照抄类也远输 suffix,顶不上互补的一臂)。

物证:`~/modify-code-runs/eagle3-test/`(accept_redhat_v2.log、accept_angelslim.log、perq_angelslim.{log,json}、measure_*.py、launch_*.sh)。

### per-request 切投机方法:改动很大 + 没必要(源码实测)

翻 vLLM 0.18 源码坐实:`self.drafter` 启动时按 method **定死单个对象**(`vllm/v1/worker/gpu_model_runner.py:518-574`);`propose_draft_token_ids`(同文件 `:4283-4496`)是 method 大分支 + 每支 `assert isinstance(self.drafter, XxxProposer)` + **整批一次 propose**(对当前 step 所有请求一起草稿,非逐请求);SamplingParams / 请求层**无任何 spec 字段**;eagle 还有自己的 draft KV cache。要 per-request 切须同时:①同时载多 drafter ②请求加 spec 字段一路透传 ③把整批 propose **拆成按 method 分子批、各自跑、再拼回**(要重切 attention metadata / draft KV / 批量 verify)——**热路径大改、数百行、多周、回归风险高**。而**两 server + tag 路由**零 vLLM 改动即达同效;叠加上面实测结论,连混合都不用建。

---

## 五、承重赌注 + 一个可能的大简化

**承重赌注 = "EAGLE 对分析类有没有用",尚未验**:suffix 在分析段命中 ~0(改写没得抄);EAGLE 理论上吃"分布相似",但分析/创作是**高熵**内容,EAGLE 在这类上通常也就 2-3×,能否把 summary 从 1.00× 拉起来**未知**。

**可能的大简化**:若 **eagle3 在照抄类也够好**(通用 drafter),**可能连 suffix 都不需要**——单一 eagle3 server 搞定两类,架构最简。**混合只在"suffix 在照抄类明显强于 eagle3、且 eagle3 在分析类明显强于 suffix"(真互补)时才值得。** 零 GPU 上界同时回答"要不要混合 / eagle3 单干就行"。

---

## 六、判决表 + 建议顺序

| 环节 | 判 |
|---|---|
| 两类分类 | ✅ 真、现成资产 |
| suffix 照抄臂 | ✅ 已证 |
| **EAGLE 分析臂** | ❌ **已实测证伪**(现成中英 head 逐题全输 suffix,summary 中位 10.4% vs suffix ~27%)→ 除非自训本域 head(3× 提升,大投入)|
| **判决模块(训 LoRA/MLP)** | ❌ 过度工程,类型信号免费(tag)→ 砍 |
| per-request 切投机方法 | ❌ 源码实测=热路径大改(数百行/多周);且没必要(两 server + tag 路由零改动)→ 弃 |
| 框架新颖性 | 🔶 待文献核查(cascade drafting / SpecDec++) |

**建议顺序(别把钱花在最贵、最可能白做的地方)**:
1. **零 GPU 混合 oracle 上界** → 判 EAGLE 值不值 + 要不要混合(用现成 SUFFIX_TRAJ / harvest 轨迹,半天出判决);
2. 值 → **找/验 Qwen3-32B 现成 eagle3 head**,实测分析段接受率;
3. 真需要混合 → **两 server + agent 按 tag 路由**(零 vLLM 改动、零路由模型);
4. **别做**:训判决器(tag 免费替代)、指望 per-request 切投机(vLLM 不支持)。

---

## 七、和项目已有资产的关系

- 本 idea = 项目 **D2 混合仲裁**的一个具体化;**路由信号**项目主张用**免费的 tag + suffix match 状态**(逐步、机制可解释),不是训模型——本文与之一致。
- **prompt/并行侧的实证**(投机 report −29%、并行 report 赚/summary 亏、去冗余 −13%)见 [../parallel-generation-experiment-v2.md](../parallel-generation-experiment-v2.md);它佐证"投机价值 ∝ 照抄内容",正是本 idea 分类的动机。
- 论文骨架见 [explore-idea/paper-skeleton.md](explore-idea/paper-skeleton.md);本仲裁若成立,是其"机制贡献"的一个候选臂。

## 相关链接

- D2 / 混合仲裁 / 资产盘点:[explore-idea/research-directions-copy-speculation.md](explore-idea/research-directions-copy-speculation.md)
- 混合 oracle 上界(零 GPU 生死闸):[explore-idea/proposal-mechanism-acceptance-uplift.md](explore-idea/proposal-mechanism-acceptance-uplift.md)
- 各 idea 判决(题型调度 ±7pp):[explore-idea/idea-verdicts-and-standing.md](explore-idea/idea-verdicts-and-standing.md)
- 并行/prompt e2e 实证:[../parallel-generation-experiment-v2.md](../parallel-generation-experiment-v2.md)
