# blender 的 BLEND_PATH 日志:怎么加的、怎么用

> 给 [blender.py](../../lmcache/v1/compute/blend/blender.py) 的两条分支各加一行 `logger.warning("BLEND_PATH=...")`,
> **grep server 日志就能确凿知道每个 blend 实际走了哪条路、复用/重算各占多少**,不靠猜。
> 这是当初揪出"control-truth 把复用静默变成全重算"那个 bug 的关键手段(见 [claude-docx/14](../../claude-docx/14-blend-control-truth-pitfall.md))。

---

## 一、为什么加(动机)

blend 有三条可能的路,但**从外部完全看不出实际走了哪条**:
1. **纯 prefill**:没有分隔符 → blend 根本不触发。
2. **Normal CacheBlend**:自动 top-k% 偏差重算、其余复用。
3. **Control**:人工指定哪些复用/重算(Exp A 诊断 / 探针轨用)。

之前靠"读代码 + 读环境变量"猜,结果踩了大坑:server 残留 `LMCACHE_BLEND_CONTROL` 把"复用实验"静默变成 control-truth 全重算,测了一大圈才发现。**教训:别猜,让代码自己报。** 于是在两个分支各打一条 warning,事后 grep 即可对账。

## 二、加在哪(三条路对应的日志)

都在 `process_qkv`(blend 的逐层融合点,[blender.py:110](../../lmcache/v1/compute/blend/blender.py#L110))里,**两个互斥分支各一条**;纯 prefill 因为不进 blend,所以**没有** BLEND_PATH 日志(以"无日志"本身作为判据)。

| 实际路径 | 触发条件 | 日志 | 行 |
|---|---|---|---|
| **Control** | `self._control is not None` | `BLEND_PATH=CONTROL ...` | [:178](../../lmcache/v1/compute/blend/blender.py#L178) |
| **Normal CacheBlend** | `_control is None` 且 `layer_id in check_layers` | `BLEND_PATH=CacheBlend ...` | [:206](../../lmcache/v1/compute/blend/blender.py#L206) |
| **纯 prefill** | 无分隔符 → 不触发 blend | (无日志) | — |

## 三、两条日志的确切代码 + 字段含义

### 3.1 Control 分支 — [blender.py:172-185](../../lmcache/v1/compute/blend/blender.py#L172)
```python
if self._control is not None:
    reuse_idx = self._controlled_reuse_indices(k.shape[0], k.device)
    if layer_id == 0:                       # ← 只在第 0 层打一次(每个 blend 一条,不刷屏)
        _ns = 0 if reuse_idx is None else int(reuse_idx.numel())   # 保留复用的 token 数
        _n = int(k.shape[0])                                       # 命中总 token 数
        logger.warning(
            "BLEND_PATH=CONTROL total=%d tok | reuse=%d (%.1f%%) | "
            "recompute=%d (%.1f%%) -> %s",
            _n, _ns, 100.0 * _ns / max(1, _n), _n - _ns,
            100.0 * (_n - _ns) / max(1, _n),
            "TRUTH=full-prefill(reuse 0%)" if _ns == 0 else "partial-reuse",
        )
```
**字段**:`total` 命中总 token;`reuse` 强制保留复用的 token 数+占比;`recompute` 重算的 token 数+占比;末尾 `TRUTH=...` / `partial-reuse` 一眼区分"全重算(复用空)"还是"部分复用"。
**判读**:`reuse=0 (0.0%) -> TRUTH` = 这个 blend 在做全重算(= full prefill,**没复用**)——当初的 bug 现场就是这条。

### 3.2 Normal CacheBlend 分支 — [blender.py:198-211](../../lmcache/v1/compute/blend/blender.py#L198)
```python
if layer_id in self.common_metadata.check_layers:   # ← 只在 check 层打(每个 blend 一条)
    diff_k = torch.sum((k.float() - old_k.float()) ** 2, dim=[1])
    total_len = diff_k.shape[0]
    topk_num = max(int(total_len * self.common_metadata.recomp_ratios[0]), 1)  # 重算 token 数
    logger.warning(
        "BLEND_PATH=CacheBlend total=%d tok | recompute(top-k)=%d (%.1f%%) | "
        "reuse=%d (%.1f%%) | ratio_cfg=%s (layer%d)",
        total_len, topk_num, 100.0 * topk_num / max(1, total_len),
        total_len - topk_num, 100.0 * (total_len - topk_num) / max(1, total_len),
        self.common_metadata.recomp_ratios[0], layer_id,
    )
```
**字段**:`total` 命中总 token;`recompute(top-k)` 自动挑偏差最大的多少 token 重算+占比;`reuse` 复用多少+占比;`ratio_cfg` 配置的重算比例(`LMCACHE_BLEND_RECOMPUTE_RATIOS`);`layer` 第几层。
**判读**:`ratio_cfg=0.0` 时 `topk_num=max(int(N*0),1)=1` → `recompute=1 / reuse=N-1 (≈99%)` = **纯复用**(C 臂)。`ratio_cfg=0.15` → 重算 ~15%。

## 四、设计选择(为什么这么打)

1. **`logger.warning` 而不是 `info`** —— info 常被 server 日志级别过滤掉看不到;warning 保证出现在 `server_*.log`。(同理后来给 demo 搜索加 `SEARCH_PATH=` 也用 warning。)
2. **每个 blend 只打一条**,不是每层都打:Control 用 `if layer_id == 0`,CacheBlend 用 `if layer_id in check_layers`(通常就第 1 层)——否则 28 层 ×N 请求会把日志刷爆。
3. **直接打 token 数 + 百分比**:不用再去算,grep 出来就能读"复用 X% / 重算 Y%"。
4. **纯 prefill 不打**:它压根不进 blend,用"**0 条 BLEND_PATH**"反过来证明走的是纯 prefill(vanilla 臂就靠这个判定)。

## 五、怎么用(核实命令)

```bash
# 这次实验的实际用法:
grep -ac 'BLEND_PATH=CacheBlend' server_C_purereuse.log   # C(纯复用)应有一堆
grep -ac 'BLEND_PATH=CONTROL'    server_C_purereuse.log   # 应 = 0(没误触发 control)
grep -ac 'BLEND_PATH'            server_B_vanilla.log     # B(vanilla)应 = 0(纯 prefill)

# 看具体复用/重算比例:
grep -a 'BLEND_PATH=CacheBlend' server_C_purereuse.log | head -1
# → BLEND_PATH=CacheBlend total=183 tok | recompute(top-k)=1 (0.5%) | reuse=182 (99.5%) | ratio_cfg=0.0 (layer1)
```
**判据**:
- 纯复用实验:server 日志**全是 `=CacheBlend`、`reuse≈99%`、`CONTROL=0`** → 复用真生效、没被 control 坑。
- vanilla 实验:**0 条 BLEND_PATH** → 纯 prefill。
- Exp A 诊断:`=CONTROL` 且 `reuse` 符合预期(truth 该 0%、swap 该是指定段)。

> 配合判定的还有 demo 端 `SEARCH_PATH=`(走缓存 vs live)——两套日志合起来,blend 路径 + 搜索路径都不靠猜。

---
*相关:[callchain_cdriver_and_control.md](callchain_cdriver_and_control.md)(blend/control 调用链)、[claude-docx/14](../../claude-docx/14-blend-control-truth-pitfall.md)(control 坑,本日志的起因)、[experiment_workflow.md](experiment_workflow.md);代码 [blender.py](../../lmcache/v1/compute/blend/blender.py)。*
