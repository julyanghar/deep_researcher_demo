# cascade attention 判决:vLLM 对"相同前缀并发"到底用不用(2026-07-07)

> **缘起**:"为什么并行更快"+"并发下 KV pruning"两条线留了个问题——decode 时批内每请求各读各的 KV(∝batch×上下文),但若它们**共享一段长前缀**,vLLM 会不会把那段共享前缀的 KV 只读一遍(**cascade attention**)?若会,则"相同前缀并发"decode 更快,agent 里同 researcher 的多个 sub-query 共享研究上下文就白拿一份提速。用户要:**相同前缀并发 vs 不同前缀并发,对比有无区别。**
> **一句话**:**vLLM 对我们的 workload 不用 cascade——①默认关(`disable_cascade_attn=True`);②强制开后要并发 ≥72(TP4 临界≈SM数/每卡KV头)才被选用,我们并发 ≤32(deep-researcher ~9)够不着 → 检测到共享前缀但性能模型判 FlashDecoding 更优、不选;③即便够到 ≥72、cascade 也只 ~3%(性能模型自己高估到 ~27%)。** 实测"相同前缀并发"省 ~18% decode,但**那不是 cascade**(我们体量下它全程没真跑),是 GPU **L2 缓存局部性** + 共享 KV 占用小——这份便宜"相同前缀"本身就白拿、不依赖 cascade。

## 一、什么是 cascade attention(30 秒)

批内多请求共享同一前缀时,标准注意力让**每个请求各自把那段共享 KV 从显存读一遍**(batch=B → 读 B 遍同一份)。cascade attention 把共享前缀的注意力**只算/读一遍**、再和每请求各自的后缀注意力合并——省的是"共享 KV 的 B−1 遍重复读"。它只影响 **decode**(prefill 那份共享 KV 由 prefix caching 管,是另一回事)。

## 二、三重不用(核心结论 + 证据)

| 层 | 结论 | 证据(源码/实测) |
|---|---|---|
| **① 默认关** | `disable_cascade_attn` 默认 **True** → `cascade_attn_enabled=False` → 计算 cascade 前缀的函数**根本不被调用** | `vllm/config/model.py:220` 默认值;`gpu_model_runner.py:464` `cascade_attn_enabled = not disable_cascade_attn`;`:3679` `if self.cascade_attn_enabled and not use_ubatching:` 才调用。默认跑插桩**零输出**。 |
| **② 我们并发下不选** | 传 `--no-disable-cascade-attn` 后:**共享前缀检测到了(8000),但并发 ≤32 时性能模型判 `use_cascade=False`**(临界 ≈72,见 §四)| 插桩实测(16 并发):`792× common_prefix_len=8000 num_reqs=16` 检测到;判决 `788× False / 8× True`(99% False)。批次扫描:64→False、**96/128→True**(§四)。 |
| **③ 开不开一样(我们体量)** | 并发 ≤32 时 ON 的 SHARED decode ≈ OFF(6.13 vs 6.16s)→ cascade 没真用、零净收益;**够到 ≥72 才被选用、也仅 ~3%** | 见 §三/§四表。 |

**只在 FLASH_ATTN 后端才有 cascade 代码**(我们这套用的就是它,server 日志 `Using FLASH_ATTN`);**FlashInfer 后端的 cascade 被明确禁用**(`flashinfer.py:1162` 返回 False,注释 "Cascade attention doesn't work")。

## 三、实验设计与数据

**Server**:vanilla vLLM(无 spec)、`--enforce-eager`、prefix caching 默认开、TP4、FLASH_ATTN、RTX 6000 Ada。
**Workload**:16 并发。**SHARED** = 同一段 8000-token 前缀 + 各自短尾(请求间共享前缀 8005 token);**UNIQUE** = 各自不同的 8000-token 上下文(共享前缀 0)。**两臂都先 gen=1 预热**(把各自前缀灌进 prefix cache)→ 两臂 prefill 都免费、TTFT 都 ~0.5s → **decode 对比干净、只差"前缀是否共享"**(消除"UNIQUE 未缓存 prefill 交错拖慢 decode"的混淆——初版没预热 UNIQUE,TTFT 16s、decode 20s,是假象)。
**判据**:① 插桩 `[CASCADE-DETECT]`(前缀检测到没)+ `[CASCADE] use_cascade`(选没选);② SHARED vs UNIQUE、ON vs OFF 的 decode_s。

**decode_s 中位(200 token/请求,16 并发):**

| | OFF(默认) | ON(`--no-disable-cascade-attn`) |
|---|--:|--:|
| **SHARED**(共享 8K 前缀) | 6.16s | **6.13s** |
| **UNIQUE**(各不同 8K) | 7.42s | 7.50s |
| SHARED 前缀检测 | —(cascade 关、函数不调) | **common_prefix_len=8000, num_reqs=16 ✓** |
| use_cascade | — | **False(788/796 ≈ 99%)** |

**读法**:
- **ON ≈ OFF**(SHARED 6.13≈6.16、UNIQUE 7.50≈7.42)→ 开 cascade 没改变任何东西;
- **SHARED 比 UNIQUE 快 ~18%**,且**两版都在** → 与 cascade 无关(cascade 全程没真跑)。

## 四、为什么性能模型否掉 cascade(机制)

cascade 的判决在 `flash_attn.py:987 use_cascade_attention`。前置门槛(共享前缀≥256、并发≥8、无 alibi/滑窗)我们**全过**;卡在最后的 **cascade vs FlashDecoding 性能模型**(`:1044-1062`,注释自承 "very rough")。

**关键:性能模型在 worker 里用【每卡】头数算,不是整模型。** `num_query_heads = get_num_attention_heads(parallel_config)`(`gpu_model_runner.py:458`,带了 parallel_config)→ **TP4 下每卡只有 16 Q 头 / 2 KV 头**(不是整模型的 64/8)。公式:
- `cascade_time = cdiv(每卡Q头16, SM) × 前缀瓦片`(共享前缀只算一份 → **与并发无关、恒定**);
- `flash_decoding_time = cdiv(并发 × 每卡KV头2 × 前缀瓦片, SM)`(每请求各复制一遍前缀 → **随并发线性涨**);
- `use_cascade = (cascade_time < flash_decoding_time)`。

**临界并发 ≈ SM数 / 每卡KV头 = 142/2 ≈ 72**(不是一度算错的 18——那误用了整模型的 8 KV 头)。**低于 72**:FlashDecoding 每请求 CTA 少、142 个 SM 装得下、一波打完更快;**高于 72**:每请求复制前缀撑爆 SM、cascade 的"只算一份"才划算。**TP 分片让每卡 KV 头变少(8→2)→ FlashDecoding 更不需要 cascade → 临界被推得很高。** 注意方向:**cascade 在高并发才赢,不是低并发**。

**批次扫描实测(cascade4:64/96/128 并发、2K 前缀、ON = `--no-disable-cascade-attn`):**

| 并发 | use_cascade | ON(cascade)/OFF(FlashDec) SHARED decode |
|--:|---|--:|
| 64 | False(临界下)| 1.00 |
| **96** | **True** | **0.97** |
| **128** | **True** | **0.97** |

→ **临界 72 坐实**(64 没翻、96/128 翻 True);**cascade 被选用后 SHARED decode 真提速、但只 ~3%**(性能模型预测 ~27%,实际只兑现 3%,印证"very rough")。UNIQUE 不变(无共享前缀、cascade 不适用)。

## 五、那 18% 是什么(不是 cascade)

cascade 全程 use_cascade=False,却仍有 SHARED<UNIQUE 18%。归因:
- **GPU L2 缓存局部性**:SHARED 的 16 个请求读的是**同一份物理 8K KV 块**(prefix caching 去重成一份)→ 反复读同块命中片上 L2,省 HBM 带宽;UNIQUE 是 16 份不同块、各读各的、L2 装不下 → 全走 HBM;
- **KV 总占用更小**:SHARED 的共享前缀只存一份,UNIQUE 存 16 份。

**关键**:这份便宜是"**相同前缀 + prefix caching**"本身带来的(共享物理块 → L2 复用),**不依赖 cascade**、不用开任何东西。这也补充了"[并发下 KV pruning](decode-step-compute-anatomy.md)"那条线:相同前缀并发**已经**通过 L2 局部性拿到一部分 decode 便宜了。

## 六、对部署/后续的含义

- **不用指望 cascade(对我们)**:默认关;强制开后**只有并发 ≥72 才被选用**(临界=SM数/每卡KV头,TP4≈72),而我们 workload 并发 ≤32(deep-researcher 才 ~9)→ **够不着、永不选用**;
- **即便够到 ≥72,cascade 也只 ~3%**(cascade4 实测),性能模型还高估到 ~27% → 收益本就薄;
- **相同前缀并发的真便宜是 L2 局部性(~18-20%)、不是 cascade**——白拿、不用开任何东西;agent 里同 researcher 的 sub-query 共享上下文天然吃到(靠 prefix caching + L2);
- **要真拿 cascade 收益,方向是"高并发(≥72)+ 长共享前缀"**(方向和直觉相反:高并发才需要 cascade),不是我们的 workload。

## 七、复现与物证

- 目录:`~/modify-code-runs/cascade-attn/`
  - `gen_workload.py`(生成 SHARED/UNIQUE,精确控前缀长度)、`concurrent_send.py`(N 并发、逐请求 ttft/decode_s、ignore_eos 强制等长、`--limit` 控批次);
  - `run_cascade.sh`(vanilla+spec 基线,16并发)、`run_cascade2.sh`(cascade ON/OFF,16)、`run_cascade3.sh`(批次 16/24/32)、`run_cascade4.sh`(**批次 64/96/128、2K前缀,证临界72 + cascade真收益~3%**);
  - `server_c{2,3,4}_*.log`(含插桩判决行)、`res_c{3,4}_{off,on}_{shared,unique}_n{N}.jsonl`(时间)、`cascade{,2,3,4}_status.txt`。
- **插桩说明(便于回滚)**:给 `vllm/v1/worker/gpu_model_runner.py` 加了 3 处(env `VLLM_CASCADE_LOG` 或哨兵文件 `/home/yilin/tmp/CASCADE_ON` 门控,**两者都不在时零行为变化**):flag 定义(~221)、`[CASCADE-DETECT]`(提前返回前,~2210)、`[CASCADE]`(use_cascade 后,~2282)。干净基线存于 `~/modify-code-runs/cascade-attn/gpu_model_runner.py.baseline_0707`,`cp` 回去即还原。

## 相关

- decode 每步成本解剖 + 并发下 KV 占比:[decode-step-compute-anatomy.md](decode-step-compute-anatomy.md)
- 访存瓶颈证明(为什么 decode 时间≈读权重):同上 §三
