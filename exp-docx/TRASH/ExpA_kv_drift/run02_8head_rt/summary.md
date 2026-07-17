# Exp A · run02(8 头 · 含 r_t)结果

> 改进版:补全 8 个 KV 头(run01 只 4 头)、按 supervisor/writer 分角色、采到 r_t 复用、剔重复与 warmup。
> 这是当前**正式结果**(run01 是首跑、4 头、已被本跑取代)。
> 图:[curves.png](curves.png) · 数据:[metrics.json](metrics.json)

## 配置
- 模型 Qwen3-32B(64 层,8 KV 头 / head_dim 128),TP2,CacheBlend,**`ratio=1.0`**(取真值 KV*)。
- 2 个 query 各跑 loop + 1 个轮2探针(复用 out_1+r_1)。
- 共 9 个 blend 事件 → 去重(2)+ 滤 warmup(2)→ **5 个真实复用事件**:supervisor ×3、writer ×2(其中 event3 = 探针,含 r_t 漂移)。

## 逐层漂移 = KV^r 偏离真值 KV*(**3 个度量**:方向 1-cos / 绝对 ‖Δ‖ / 相对 ‖Δ‖/‖KV*‖)

3 联图见 [curves.png](curves.png)(左=方向、中=绝对、右=相对)。**绝对与方向/相对给出不同的"祸首层",这点很关键。**

**方向(1-cos)+ 相对**:峰都在中深层,V≈2.5×K。
| 1-cos | 峰值层 | 全层均值 | | 相对 | 峰值层 | 均值 |
|---|---|---|---|---|---|---|
| supervisor K | L48: 0.069 | 0.029 | | K | L48: 0.30 | 0.18 |
| supervisor V | L44: 0.172 | 0.073 | | V | L44: 0.47 | 0.29 |
| writer V | L44: 0.185 | 0.081 | | V | L44: 0.53 | 0.33 |

**绝对 ‖KV^r − KV*‖(新增)**:V 峰**跑到最末层**,和上面不一致 ——
| 绝对‖Δ‖ | 峰值层 | 全层均值 | L0 | L32 | L48 | **L62** |
|---|---|---|---|---|---|---|
| supervisor K | L45: 6.9 | 4.1 | 0.26 | 3.75 | 6.53 | 5.04 |
| supervisor V | **L62: 24.2** | 3.8 | 0.00 | 1.21 | 4.77 | **24.23** |
| writer V | **L62: 27.4** | 4.3 | 0.00 | 1.39 | 5.23 | **27.38** |

## 关键发现(大白话)

1. **浅层几乎不漂(L0≈0.001),深层最狠(峰值 L44~48,约 70% 深度),末层略回落。** 不是 RelayCaching 在 Mistral 上说的"中层最狠"——Qwen3-32B 是**偏深层最狠**。即"该刷新的是中后层,浅层/sink 安全"。
2. **V 比 K 漂得狠**,且**方向/相对**指向中层(V@L44)、**绝对**指向末层(V@L62)——两者别混:
   - 按 1-cos / 相对:V 是 K 的 ~2.5 倍,峰在中深层 L44。
   - **按绝对 ‖Δ‖(关键)**:V 的原始扰动在**最后几层 L62-63 暴涨到 ~24-27,是中层(~5)的 5 倍**,而那儿的 1-cos/相对只是中等——因为末层 ‖V*‖ 本身巨大,归一化把它压平了。
   - **为什么绝对更要紧**:注意力输出 = V 的加权和,吃的是**绝对 V 差**。所以真正注入注意力的数值扰动集中在**最末层 V**;只看方向/相对会误以为中层最该修。**刷新策略:V 优先,且末层 V 的绝对扰动是最大隐患。**
3. **supervisor ≈ writer**:两角色逐层曲线几乎重合(都复用同一段 worker 输出,只是语境不同)→ 漂移由"内容+层"决定,和哪个角色复用关系不大。
4. **彻底漂飞(1-cos=1)的 token 全是 `<|fim_pad|>` 分隔符**(K* 范数 ~23,非近零;它是段边界标记,KV 完全随上下文变,换语境必然全漂)。**内容 token 只温和漂(均值 ~0.03、p90 ~0.07)。** → 所谓"稀疏",主要是分隔符这种边界 token 当离群,真正内容的漂移是平滑温和的;度量/gate 时应**排除分隔符**。
5. **r_t(reasoning trace)复用**(event3,supervisor 复用 out_1+r_1)的整体漂移与纯 summary 复用同量级(~0.02)。

## 相比 run01 修了什么
- **4 头 → 8 头**:run01 因 TP2 两 worker 写同一文件互相覆盖,只剩半数头;现在 dump 文件名带 TP rank、离线拼回整 8 头。
- **去重**:event 0≡1、6≡7 是同一请求重复落盘(token 逐一相同),按内容去重。
- **滤 warmup**:T=65/104 的小事件只复用前缀、漂移≈0,剔除。
- **按角色分**:supervisor / writer 各一条曲线。

## 仍待办 / 局限
- **样本仍偏小**(2 query / 5 事件)→ 多跑 query 扩样本。
- **r_1 单独漂移**:event3 把 out_1+r_1 一起测了,要单独看 r_1 得按 `<|fim_pad|>` 切出 r_1 token 段再算。
- **重复落盘根因待查**(同一 supervisor decide 被 blend 两次——疑似 JSON repair 重发或引擎重算;不影响结论,去重已处理)。
- attention M(Exp D)未做(FlashAttention 拿不到逐 token 权重)。

## 文件
- 图 [curves.png](curves.png)、指标 [metrics.json](metrics.json)(本目录)。
- 原始 KV dump(700M×N,太大不进 git):`/home/yilin/tmp/exp_a_real_dump/blend_<事件>_tp<rank>.pt`。
- 取数/算法脚本:[exp_a_curves.py](../../../server/vllm/Exp_A/exp_a_curves.py)、blender dump hook 见 [blender.py](../../../lmcache/v1/compute/blend/blender.py#L114);方法详解见 [Exp0_implementation_changes.md](../../Exp0_implementation_changes.md)。
