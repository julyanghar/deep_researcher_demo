# KV dump 实现:代码位置、调用链、流程

> blend 时把**两份 KV 落盘**:`kv_r`(复用的旧 KV^r)和 `kv_star`(现场重算的新 KV,truth 时 = full prefill KV\*),
> 用来离线验证 / 算几何偏差 / 算 M。**只在 Exp A 的 control-truth 抓取 pass 触发**,正常 blend / 生产 / 纯复用实验都不会 dump。
> 代码全在 [blender.py](../../lmcache/v1/compute/blend/blender.py)。

---

## 一、是什么 + 何时触发

每个 blend 请求 dump 一个 `.pt`,里面是这次命中段的:
- **`kv_r`** = 载入的**复用 KV^r**(production blend 真正会用的、旋转后的旧 KV);
- **`kv_star`** = 同一批 token**现场重算的新 KV**;control-truth(`reuse_token_ranges=[]` 全重算)时 `kv_star` 就是 **full prefill 的 KV\***。

**触发条件(四重 gate,[blender.py:144-149](../../lmcache/v1/compute/blend/blender.py#L144))**:
```python
dump_active = (
    self._dump_kv_dir                       # ① 设了 LMCACHE_BLEND_DUMP_KV=<dir>
    and self._control is not None           # ② control 生效(CONTROL_ENABLED=1 + 路径)
    and self._control.get("dump")           # ③ control 文件里 "dump": true
    and not self._control.get("reuse_token_ranges")  # ④ 是 truth pass(复用空 = 全重算)
)
```
→ **只有 Exp A 的"全重算 + 显式 dump"那个 pass 才落盘**;正常 CacheBlend / 纯复用实验(无 control、无 dump_dir)四个条件都不满足 → 永不 dump、零开销。

**怎么开启**(参考 [validate_32b.py](../../../tmp/validate_32b.py)):
```bash
export LMCACHE_BLEND_DUMP_KV=/path/to/dump_dir
export LMCACHE_BLEND_CONTROL_ENABLED=1
export LMCACHE_BLEND_CONTROL=/path/to/control.json   # 内容: {"reuse_token_ranges": [], "dump": true}
```

---

## 二、实现代码(4 处)

| 位置 | 行 | 作用 |
|---|---|---|
| `__init__` | [:79-87](../../lmcache/v1/compute/blend/blender.py#L79) | 读 `LMCACHE_BLEND_DUMP_KV`→`_dump_kv_dir`;init `_dump_layers={}`、`_dump_q_layers={}`(patch C)、`_dump_counter=0`;建目录 |
| `process_qkv` 抓取 | [:144-164](../../lmcache/v1/compute/blend/blender.py#L144) | `dump_active` 时把这层的 `(old_k, old_v, k, v)` 存进 `_dump_layers[layer_id]`;patch C 再抓 `q` |
| `blend_layer` 触发 | [:293-294](../../lmcache/v1/compute/blend/blender.py#L293) | 跑完所有层后:`if _dump_kv_dir and _dump_layers: _flush_dump(tokens)` |
| `_flush_dump` 落盘 | [:299-346](../../lmcache/v1/compute/blend/blender.py#L299) | 把逐层抓取 stack 成张量,`torch.save` 成 `.pt` |

### 2.1 抓取(process_qkv 内,[:150-164](../../lmcache/v1/compute/blend/blender.py#L150))
```python
if dump_active:
    self._dump_layers[int(layer_id)] = (
        old_k.detach().to("cpu", torch.float32),  # KV^r:载入的复用值(旋转后)
        old_v.detach().to("cpu", torch.float32),
        k.detach().to("cpu", torch.float32),       # KV*:现场重算的新值(旋转后)
        v.detach().to("cpu", torch.float32),
    )
    if self._control.get("dump_q"):                # patch C(算 M 用)
        self._dump_q_layers[int(layer_id)] = q.detach().to("cpu", torch.float32)
```
**抓取时机很关键**:在 `q, k = rotary_emb(...)`([:133](../../lmcache/v1/compute/blend/blender.py#L133))**之后**、control/CacheBlend **就地覆盖之前**。所以:
- `old_k` 此刻还是**复用的 KV^r**(旋转后),没被覆盖;
- `k` 是**现算的 KV\***(旋转后);
- 两者**同一 RoPE 帧** → 可直接比(cos/L2)。

### 2.2 落盘(`_flush_dump`,[:308-337](../../lmcache/v1/compute/blend/blender.py#L308))
```python
layer_ids = sorted(self._dump_layers)
kv_r    = torch.stack([stack(old_k, old_v) for l], dim=1)   # [2(K/V), L, T, hidden]
kv_star = torch.stack([stack(k,     v)     for l], dim=1)   # [2(K/V), L, T, hidden]
rank = dist.get_rank()                                       # TP worker 编号
path = f"{dump_dir}/blend_{counter:05d}_tp{rank}.pt"
save_obj = {"kv_r":kv_r, "kv_star":kv_star, "tokens":tokens, "recomp_ratios":...}
if self._dump_q_layers: save_obj["q"] = stack(...)          # patch C: [L, T, H*D]
torch.save(save_obj, path)
counter += 1; _dump_layers={}; _dump_q_layers={}            # 复位,下个 blend 重来
```

---

## 三、调用链 / 流程

```
vllm_v1_adapter.blend()  [adapter:902]      ← 命中存好的 KV,触发 blend
  └─ blender.blend()  [blender:370]
        ├─ _load_control()  [:383]          ← 读 control(决定 dump_active 的 ②③④)
        └─ blend_layer(tokens, mask)  [:239]
             │  逐层(for i in num_layers [:268]):
             │    ├─ next(layerwise_retriever)   载入这层 KV^r
             │    ├─ next(layerwise_model_executor) → compute_layer 现算这层
             │    │     └─ process_qkv(q,k,v,...) [:110]
             │    │           └─ if dump_active: 把 (old_k,old_v,k,v) 存进 _dump_layers [:150]
             │    └─ (control/CacheBlend 就地改 old_k —— 在抓取之后)
             └─ 所有层跑完 → if _dump_kv_dir and _dump_layers: _flush_dump(tokens) [:293]
                                └─ stack + torch.save → blend_XXXXX_tpR.pt [:337]
```
一句话:**逐层在 process_qkv 把"旧 KV^r + 新 KV\*"攒进 `_dump_layers`,所有层跑完一次性 stack 落盘成一个 `.pt`。**

---

## 四、dump 文件格式(`blend_{序号}_tp{rank}.pt`)

`torch.load` 得一个 dict:

| 键 | 形状 | 含义 |
|---|---|---|
| `kv_r` | `[2, L, T, hidden]` | 复用的旧 KV^r(dim0:0=K,1=V;L=层;T=命中token;hidden=本 rank 的 KV 头切片) |
| `kv_star` | `[2, L, T, hidden]` | 现算的新 KV;truth 时 = full prefill KV\* |
| `tokens` | `[T]` | 命中段的 token id(离线按分隔符切段用) |
| `recomp_ratios` | — | 当时的重算比例配置 |
| `q`(可选,patch C) | `[L, T, H*D]` | 决策 query;teacher-force 时末位 = 预测 status 的 query(算 M) |

**TP 切片**:每个 worker 只持有自己那份 KV 头,所以文件名带 `tp{rank}`;**离线要把各 rank 的 `hidden` 维拼回完整头维**(否则只有半截头)。

---

## 五、关键语义(别踩坑)

1. **`kv_star` 只有在"全重算"时才 = full prefill**:即 control-truth(`reuse_token_ranges=[]`)或 `recompute_ratio=1.0`。普通 CacheBlend(只重算 top-k)时 `kv_star` 不是完整 KV\*,dump 无意义 —— 所以 gate 第④条强制只在 truth pass dump。
2. **抓的是旋转后(post-RoPE)的 K**:`kv_r`/`kv_star` 同帧可比;别拿去和未旋转的 K 比。
3. **`old_k` 必须在覆盖前抓**:control/CacheBlend 分支会 `old_k[:] = k` 就地改,抓取代码在其之前(只读 detach 到 CPU),所以拿到的是真·复用值。
4. **一个 blend 一个 `.pt`**,`_dump_counter` 自增;每个 worker 各写各的 `tp{rank}` 文件。

---

## 六、谁在用这个 dump

- **[validate_32b.py](../../../tmp/validate_32b.py)**:同一上下文跑两遍,比 `kv_r`(第一遍存的干净 full-prefill KV)vs `kv_star`(第二遍 control-truth 现算)→ 证实 **control-truth 全重算 ≡ full prefill**(32B cos-dist 中位 0.0001)。
- **Exp A 几何**:`Δ = kv_r vs kv_star` 的逐 token 偏差(cos/L2)→ 看复用值偏离真值多少。
- **patch C(探针轨算 M)**:额外 dump 的 `q` + `kv_star` 的段 K → `M_i = softmax(q_末 · K*_段i)`。

---
*相关:[callchain_cdriver_and_control.md](callchain_cdriver_and_control.md)(blend/control 调用链)、[blend_path_logging.md](blend_path_logging.md)(BLEND_PATH 日志)、[experiment_workflow.md](experiment_workflow.md);代码 [blender.py](../../lmcache/v1/compute/blend/blender.py)、[validate_32b.py](../../../tmp/validate_32b.py)。*
