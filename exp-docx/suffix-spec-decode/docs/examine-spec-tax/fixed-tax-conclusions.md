# 固定税破案·结论精简版（只保留最终定论）

> 本文是 [how-we-found-the-fixed-tax.md](how-we-found-the-fixed-tax.md) 的精简版：**只留每节最终结论**，删掉 8 轮修正的弯路。要看破案过程/踩坑史/证据链，去看全文。
> **一句话**：开 suffix 投机后每步固定多花 ~10ms，**8.1ms 是 graph+eager 混合模式的逐层边界税**（主体=驱动锁等待 4.5ms）、1.9ms 是 spec 管线；`--enforce-eager` 一刀消掉 8.1，端到端 +28%。

## TL;DR：五常数成本模型

每个 decode 步的耗时 = 五个可加常数（TP4，summary 段，实测残差 0）：

| 常数 | 值 ms/步 | 什么时候交 |
|---|--:|---|
| ① graph 基线 | 24.0 | 每步（读一遍 32B 权重 + 整图回放） |
| ② eager 附加 | +0.5 | 用 `--enforce-eager` 时（放弃整图的净代价） |
| ③ spec 管线 | +1.9 | 投机开着就交（与执行模式、草稿量都无关；propose+采样回传+memcpy+GIL） |
| ④ 混合边界税 | +8.1 | 仅 graph+eager 混合模式交；主体=**驱动锁等待 4.5**（+图段启动 1.4+杂项 2.2）；**全 eager 一刀消掉** |
| ⑤ 草稿费 | 激活 5.8 + 0.62×草稿位置数 | 这一步"有草稿"就交 |

核对（预测=实测）：A=①=24.0；F=①+②=24.5；C0(混合空草稿)=①+③+④=34.0；Ceager=①+②+③+⑤=33.9；七配置全对上。

---

## 一、靶子：C0 空草稿比 A 每步多 10ms

三配置每步耗时：A(不投机)=24.0，C(suffix)=42.5，**C0(suffix 但永远空草稿)=34.0**。C0 草稿=0、前向和 A 一样大，却仍多花 10ms——这 10ms 与"猜没猜"无关，是"投机机制开着"本身的代价 = 每步固定税（= ④8.1 + ③1.9）。用 C0 破案是为了隔离纯机制开销。

## 二～四、概念底座（kernel / CUDA graph / 空转气泡）

- **kernel launch**：CPU 执行 `cudaLaunchKernel` 把一个计算任务派给 GPU；异步（发完不等 GPU）。一步 decode 要发几百个。
- **CUDA graph**：把一步内几百个 kernel 的发射录成一张"图"，之后一次回放、GPU 连续跑不等 CPU。vLLM 默认给 decode 开（A 用它，整模型录成一张大图）。
- **GPU 空转气泡**：decode 是显存带宽瓶颈，GPU 常闲等显存；若它闲下来时 CPU 没及时递下一个 kernel，就多一段干等 = 气泡。

## 四b、两个关键事实（破案钥匙）

1. **nsys 默认不记录 graph 内部 kernel**——图里的 FFN 在 GPU 上算，但逐 kernel 表里查无此人，时间线看着"空"。
2. **graph+eager 混合模式（piecewise）**：开投机后 vLLM 只把**注意力**挖出来改 eager 手工发，FFN/norm 仍留 graph。每层要在"回放 FFN 图"和"手工发注意力"之间切一次，64 层切 64 次——**这就是混合边界税的来源**。（注意力必须 eager：草稿长度每步变、张量形状跟着变，graph 要求形状固定。）

## 五、差分排除（结论）

- **CPU 可插桩环节合计仅 ~1.6ms**（propose 0.16 + sampler 0.82 + metadata 0.44 + RPC 0.15），排除为大头；
- **kernel 发射数不是税**：Fea(全 eager baseline)每步发 350 万个 kernel 却不慢，C0 发得少反而慢——数量被"GPU 等显存"掩盖与否才是关键；
- 嫌疑最终落到 **graph+eager 混合边界**（§六、§七定案）。

## 六、那 455μs"空隙"是什么（定案）

nsys 时间线上，C0 注意力流每层 reshape 前有个 **455μs 空隙**。**它不是空转**：
- **算术反证**：可见注意力 kernel 每层才 ~22μs，×64=1.4ms/步；若 455μs×64≈29ms 真空转，GPU 每步只干 1.4ms——32B 前向不可能；那 ~24ms FFN 计算没在 trace 里，只能藏在空隙里；
- **直接影像**：开 `--cuda-graph-trace=node` 重录，空隙里冒出 34k gemm + 8.5k silu（FFN），窗口 GPU 活跃率 91.5%。

**结论**：455μs ≈ FFN/norm/proj 执行 ~325μs（graph 未记录）+ 真气泡 ~130μs/层。**真税是每层多出的 ~100μs 混合边界开销（≈每步 6-8ms），不是 455μs。**

## 七、2×2 锁定真凶（核心结论）

四格吞吐（char/s，加速比对 A）：

| 配置 | 模式 | SUMMARY | REPORT |
|---|---|--:|--:|
| A 原生 | 纯 graph | 152 (1.00x) | 130 (1.00x) |
| C suffix k24 | graph 混合 | 126 (0.83x) | 241 (1.85x) |
| **Ceager suffix k24** | **全 eager** | **152 (1.00x)** | **298 (2.29x)** |
| D ngram k8 | graph 混合 | 157 (1.03x) | 279 (2.14x) |
| Deager ngram k8 | 全 eager | 155 (1.02x) | 291 (2.24x) |

**三个结论**：
1. **慢的是"混合"不是"eager"**：suffix 全 eager(Ceager)从 0.83x 翻回 1.00x、report 冲 2.29x；eager 反而是最快模式。
2. **"有 gap"不是根因**：ngram 同样混合、同样有 379μs gap，却比原生快——D 混合 157 ≈ Deager eager 155。
3. **ngram 也交混合执行税**（发射同样慢 24.8μs、每步 31.4≈C0 34.0），它靠**每步多吐接受 token 摊薄**赚回，不是"免税"。

**终审定案（五常数）**：C0eager(空草稿+全 eager)实测 26.4 → **固定税 8.1 属混合形态、spec 管线仅 1.9**。混合边界税主体是 `cuLaunchKernel` 内**驱动私有条件变量的锁等待 4.5ms/步**（栈实锤：graph 段与 eager 发射交错触发驱动串行化；全 eager 对照 0.00s）。异步调度经 D/E 直测只值 0.5ms，非主因。

## 七b、按量税与 k/pad 选择（结论）

固定税之外还有**按量税**（草稿费 ⑤的按量部分）：每多一个草稿位置多付 0.62ms 验证。结论：
- **k 越大越差**：接受在 k=8 就饱和（每步接受 2.21→2.33，k24 只多 +0.12），k24 多出的上限全是白验证；
- **k×pad 扫点单调**：k8+pad8 (0.97x/2.11x) > k16 > k24+pad24 (0.85x/1.91x)；
- **pad 只在混合模式赚（+2ms，靠 graph 桶稳定），全 eager 下倒亏 3.5ms**（dummy 纯按量成本、无桶可稳）；
- **全 eager 下 k 无所谓**（CeagerK8 151 ≈ Ceager 152）——"k=8 甜点/pad 补形状"整套都是混合模式的伤药，推荐配置(全 eager)用默认 k、不 pad。

## 七c、真实 e2e 校验（结论）

replay 是 batch=1 冷树显微镜；e2e(local、并发 3、热树)是生产口径。同天六格重跑：**Ceager 是全场冠军**——端到端墙钟中位 1.28x（比 A 快 28%）、SUMMARY 聚合 1.07x、REPORT 2.37x、QUERY_PLAN 1.54x，decode 口径 40 题零回归；**唯一代价 TTFT +14%**（纯 enforce-eager 的 prefill 账单）。两刀合一(CeagerK8pad)被否（pad 洪水把接受率打到 13%）。

## 八、之前对不上的现象，现在闭合

1. **F 关 graph 只慢 0.7ms，C0 却慢 10ms**：F 关整张大图、退成的 eager 被显存等待掩盖；C0 是混合，每层切换处的 CPU 活露出来。
2. **Fea 发 350 万 kernel 不慢，C0 发 45 万反而慢**：数量不重要，能否被显存等待掩盖才重要。
3. **税和"猜没猜"无关（空草稿也交）**：混合边界是执行结构，每层每步都发生。
4. **ngram 也混合凭什么不慢**：它也交混合税，靠接受摊薄 + 没有 suffix 的额外 CPU 机件。

## 九、优化含义（最终）

1. **头号（已验证）**：`suffix + --enforce-eager` 去掉混合边界 8.1 → replay 0.83x→1.00x、e2e +28%，只改一个启动参数；代价 TTFT +14%。
2. **异步调度：基本判死**（batch=1 直测 0.5ms，全 eager 下为 0）。
3. **按量税**：混合模式用 k8+pad 砍；全 eager 下不需要。
4. **剩余系统标的**：草稿费激活 5.8 中的 **~4.5ms 未归因**（sampler/metadata 之外，嫌疑 logits 多行处理/spec 采样 kernel/输出拷贝）——税表最后一块砖，消掉可让冷树 summary 从 1.00x 到 ~1.15x。

## 相关

- 破案全过程/8 轮修正史：[how-we-found-the-fixed-tax.md](how-we-found-the-fixed-tax.md)
- 五常数模型全表：[suffix-vs-native-cost-anatomy.md](suffix-vs-native-cost-anatomy.md)
- 结果速览：[tax-split-progress-report.md](tax-split-progress-report.md)
- suffix 机制本身：[../../../18-vllm-suffix-decoding.md](../../../18-vllm-suffix-decoding.md)
