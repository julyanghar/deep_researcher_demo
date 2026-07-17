# Exp_M · M 抓取卡点调试记录(大白话)

> 状态(2026-06-26):**✅ 已解决!** M 抓到了(`M_rowsum_mean=4212`、决策位 q abs-max=28.62)。
> 下面保留完整调试过程;**根因 + 解法**在本节。

## ✅ 解决了:根因 = store_generated 丢了末位 token,再喂时没跟着丢
**一句话**:`store_generated` 存的是 **`token_ids[:-1]`**——解码采样的**最后一个 token 从没喂回模型、没有 KV**,所以末位被丢、没存进去。我**再喂时却喂了完整决策**(多一个 token)→ 末段 token 数对不上(214 存 vs 215 喂)→ **content-hash 不匹配 → 决策段命不中 → 进不了 blend 区 → 抓不到决策 q → M=0**。

**解法**:再喂时也**丢掉决策末位**——喂 `[super_prompt + SEP + 决策[:-1]]`。一改,命中区从 8225 跳到 **8440**(决策全进来了),决策 q 抓到、M 非 0。

**正确配方**(已写进 [code_changes_report.md](code_changes_report.md) 的 step 4):
1. summary 全复用下解码决策,**开 `store_generated`**(决策段 KV 存缓存,`gen_start` 把它存成纯内容哈希段);
2. **丢决策末位** `gen_ids[:-1]`;
3. 等异步落盘(sleep);
4. 再喂 `[super_prompt + SEP + 决策[:-1]]` + dump → 决策段命中、进 blend 区 → `dump_q` 抓决策 q;
5. summary 仍走复用(`K^r`)→ 决策 q 打到 `K^r` → 离线 `softmax(q·K^r)` = M。

> 已在 LMCache 源码 [`vllm_v1_adapter.py`](../../lmcache/integration/vllm/vllm_v1_adapter.py)(store_generated 处)加了消费侧警告,防再踩。

---

## 以下是当时的调试过程(保留备查)
> 当时状态:整条管线就差**一步**没通——**抓不到 supervisor 决策的 query(q)**,导致 M 全 0。
> 别的全成了:demo 复用、决策段存进缓存、ΔK/ΔV、12 分支 control、transfer_M 对齐,都 OK。

---

## 一、我们要算的 M 是什么(先把目标说清)
M = **supervisor 做决策时,对每一段 summary 有多"上心"**(注意力分数)。算它需要两样东西:

1. **决策的 query(q)** —— supervisor 解码决策那一刻,每个决策 token 的"提问向量"。
2. **summary 的 key(K^r)** —— 被复用的那份 summary 的"被查向量"。

把这俩一拼(`softmax(q·K)`),就知道决策对哪段 summary 注意力高 → 这就是 M,用来指导 writer 该重算哪段。

打个比方:决策 token 是"考官",summary 是"考卷"。M = 考官批每张卷子时**眼睛在哪张卷子上停得久**。要测这个,得同时拿到**考官的视线(q)**和**卷子(K)**。

---

## 二、卡在哪:卷子(K)有了,考官的视线(q)抓不到
- **summary 的 K^r:✅ 拿得到**。summary 是"分隔符 `<|fim_pad|>` 切出来的段",会进 blender 的"复用区",blender 能把它的 K dump 下来。
- **决策的 q:❌ 抓不到**。这是卡点。

### 为什么决策 q 抓不到?
blender 这个组件**只能 dump"复用区"里的东西**,而"复用区" = **命中缓存的、按分隔符切的段**(也就是那些 summary)。

**决策是在所有 summary 之后才生成的、它前后没有分隔符包着、不是一个"段"** → 它进不了"复用区" → blender 根本看不到它的 q。

> 验证数据:dump 的覆盖长度 T 永远停在 **8225**(= super_prompt 的长度,summary+问题),决策位在 8226 往后,**从来没进过 dump**。

---

## 三、试图让"决策"也变成一个能命中的"段"(试了 5 种,全失败)
思路:用 `store_generated`(把生成的 token KV 存进缓存)把决策段存起来,再喂一遍 `[上下文 + 决策]`,指望决策段这次能"命中缓存、进复用区",blender 就能抓它的 q 了。

实测:**决策段每次都成功存进去了**(日志 `Final save of 8439 tokens, gen_start=8225`),但**再喂时 blend 死活不命中它**,复用区永远止于 8225。试的 5 种:

| # | 办法 | 结果(dump 覆盖 T) |
|---|---|---|
| 1 | 4a 存决策 + 再喂 `[上下文+决策]`(不加分隔符)| T=8103(只到 summary)|
| 2 | 先 prefill 存 `[上下文+SEP+决策]`,再喂同样的 | T=8225 |
| 3 | 让模型在 SEP 之后生成决策(结果决策变成 `'\n'` 垃圾)+ 再喂 | T=8225 |
| 4 | 正解:`store_generated` 靠 `gen_start` 把决策存成"纯内容哈希段" + 再喂带 SEP + 等异步落盘 | T=8225 |
| 5 | 在 #4 基础上,再喂时也带上相同的 `kv_transfer_params` | T=8225 |

**结论:决策位从来没进过 dump(T 卡在 8225)。** 决策段明明存了、哈希也应该对得上(纯内容哈希、不含前缀),但 blend lookup 就是不认它。

### 现在的理解(也是没看穿的地方)
- blend 的段是**纯内容哈希**(`_hash_tokens`,不含前缀)→ 理论上"存的决策段"和"再喂的决策段"哈希应该一样、应该能命中。
- summary 能命中,是因为它们在后续轮被**无额外配置地重新存过**;而决策段**只被 store_generated 存了一次**,刚存完就去 lookup,**就是匹配不上**。
- 怀疑方向:**异步落盘/索引没及时更新**,或 **retrieve 对 prompt 尾段那个 chunk 有特殊处理**。这一层(LMCache 的 store/lookup 内部)我没能从代码看穿。

---

## 四、已经跑通的部分(别误会成全错了)
| 环节 | 状态 |
|---|---|
| demo 里 supervisor/writer 真复用 summary(CacheBlend reuse 99.9%)| ✅ |
| store_generated 把决策段 KV 存进缓存 | ✅ |
| writer 的 ΔK/ΔV(从 writer dump 出)| ✅ |
| transfer_M 按内容对齐(supervisor 25 段 ↔ writer 24 段,对上 24)| ✅ |
| 12 个分支 control 全写出、复用比例都对 | ✅ |
| 离线单测(test_exp_writer)| ✅ 全过 |
| **M(决策注意力)** | ❌ 全 0(就差决策 q)|

**就差"抓到决策 q"这一环,接上就全通。**

---

## 五、两条路
- **A(继续现路)**:搞清"刚 store_generated 的决策段为啥 blend lookup 不命中"。要么异步落盘等久点/强制同步,要么 LMCache 内部有个我没找到的开关。需要对 LMCache store/lookup 更深的了解。
- **B(换路,更稳·推荐)**:teacher-force `[上下文 + 决策]`(summary 照常走 blend 复用)时,**给模型 attention 挂一个 forward hook,直接抓决策位的 q**——**绕开 blend 那套"必须是命中段"的限制**。summary 的 K^r 已经在 dump 里了,离线把 q 和 K 一拼就出 M。

**一句话**:卡点是"决策这段东西塞不进 blender 能看见的复用区"。A 是想方设法把它塞进去(试了 5 次没成);B 是干脆不靠 blender、直接在模型前向里把决策的 q 抓出来。

> 相关:dump 还留着(`/home/yilin/tmp/exp_m_test/dump/`),A/B 都能马上接着试。设计见 [code_change_plan.md](code_change_plan.md),改动见 [code_changes_report.md](code_changes_report.md)。
