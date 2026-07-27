# 从零讲清:`--trim-loss-positions` 为什么在 USP 序列并行下会坏

> **缘起**:PR #705 给 SpecServe 的 eagle3 训练加了 `--trim-loss-positions`(A 级 loss 位置裁剪,省显存)。核心 maintainer 回复:"长上下文 eagle3 训练现在要用序列并行(sequence parallel),请验证这个方法和序列并行兼不兼容"。一查——**不但不兼容,而且没设防、会静默走进坏路**。这篇把"坏在哪、为什么坏"从零讲清。
>
> **读法**:§1 TL;DR 一句话结论;§2 概念梯子(USP 到底怎么切序列)是理解前提,别跳;§3 手把手小例子是本文核心,三处硬伤都在这一个例子上看得见;§4 追问区;§5 术语表;§6 进阶(ring/Ulysses 注意力内部,跳过不影响理解)。
>
> **假设起点**:你已经知道 eagle3 草稿训练会做 k 步 TTT 展开、每步跑草稿 backbone 再算 teacher/logits/loss;也知道 `--trim-loss-positions` 的意图是"backbone 照样跑全长,但 teacher/logits/loss 只在监督位算"。你**不需要**预先懂序列并行——那正是本文要从零补的。
>
> **代码坐标**:全部基于分支 `pr-trim-a-v2`(commit `ba20730`,即 #705 的 head)。链接是相对路径,点开需仓库切在该分支。

---

## 1. TL;DR

`--trim-loss-positions` 在**单卡**上被证明和全长路径逐位等价(有测试)。但在 **USP 序列并行**下,它会走进一条**没人拦的坏路**,同时踩三个雷:

1. **backbone 喂错长度** —— trim 分支把"本地整段(含重叠尾)"直接喂给 backbone,而序列并行要求先裁回"自己那段";喂进去的 hidden 长度和 position_ids 长度对不上 → RoPE/注意力形状错或算错。
2. **重叠尾跨卡双算** —— trim 统计监督位时把"重叠尾"也算进来,可那截位置是**下一张卡的开头**,于是同一个监督位被两张卡各算一遍。
3. **loss 分母用错** —— trim 用"本地整段长度"当归一化分母,可正确的分母是"自己那段长度";差一截,而且没有按全局监督数跨卡归并。

而 `_trim_ok` 这个开关**只看 `trim_loss_positions` 有没有开、不看用的是不是 USP**,schema 里也没有任何 validator 禁止 `usp + trim` 组合。所以只要你 `attention_backend=usp` 且 `trim_loss_positions=true`,就静默走进上面这条坏路——不会报"不支持",而是默默算错。

一句话钉住:**trim 的设计前提是"一张卡拿着整条序列",而序列并行的前提是"每张卡只拿一段";这两个前提直接冲突,trim 却没检查自己在不在序列并行下。**

---

## 2. 概念梯子:USP 是怎么把序列切到各张卡的

### 2.1 为什么要"序列并行":一张卡装不下 64k 的激活

草稿训练里,每个位置都要过一遍 `[隐藏维]`,还要投影到词表算 `[序列长 × 词表]` 这么大的激活。序列一长(64k),这些激活单卡就爆显存。

**办法:把"序列"这根轴切开,分给多张卡。** 卡 0 管前一段位置、卡 1 管中间一段……每张卡只存自己那段的激活,显存就下来了。这就是**序列并行(sequence parallel)**。SpecServe 用的具体方案叫 **USP**(Unified Sequence Parallel,底层是 Ulysses + Ring attention 两种机制拼起来,细节见 §6,现在不用管)。

> 桥句(记住这一句,后文反复回指):**序列并行 = 把序列切段,每张卡只拿一段。**

### 2.2 三个角色:发料员、对齐工、拼图工

序列并行要正确,靠三个"工人"配合。先认人,再看它们怎么协作:

| 角色(人话) | 源码里是谁 | 职责 |
|---|---|---|
| **发料员** | collator `process_data_usp` | 把整条序列切成每张卡的本地一段,还多塞一小截"重叠尾"给它当草稿纸 |
| **对齐工** | `UspAdapter.step_view` | 每一步:先用重叠尾做完 TTT 移位,再把这段**裁回"自己那段"**、丢掉重叠尾、摆好位置号 |
| **拼图工** | backbone 里的 ring/Ulysses 注意力 | 靠跨卡通信,把各卡的段拼回"全局注意力",让每个位置还能看到全局 |

> **源码钉子**:发料员 [`preprocessing.py:432-499`](../../../SpecForge/specforge/data/preprocessing.py#L432-L499);对齐工 [`eagle3_adapters.py:106-145`](../../../SpecForge/specforge/core/eagle3_adapters.py#L106-L145);拼图工里决定注意力全局长度的那行 [`llama3_eagle.py:1415`](../../../SpecForge/specforge/modeling/draft/llama3_eagle.py#L1415)。

### 2.3 发料员切出来的三个长度(本文最重要的三个词)

发料员的切法([`preprocessing.py:442-447`](../../../SpecForge/specforge/data/preprocessing.py#L442-L447)):

```python
global_len = min(max_len, input_ids.shape[1])      # 整条序列长
chunk_size = (global_len + sp_size - 1) // sp_size  # 每张卡"自己那段"的长度 = ceil(global/卡数)
start = sp_rank * chunk_size                         # 这张卡从哪个全局位置开始
local_len = chunk_size + ttt_length                 # 这张卡实际拿到的长度 = 自己那段 + 重叠尾
end   = min(start + local_len, global_len)           # 拿 [start, start+local_len)
```

对齐工那边再定义([`eagle3_adapters.py:126`](../../../SpecForge/specforge/core/eagle3_adapters.py#L126)):

```python
usp_chunk_size = seq_length - ttt_length   # = chunk_size,即"自己那段"
```

**人话概念 ↔ 源码名字**(这张表是本文的桥,忘了就回来查):

| 人话 | 源码名 | 玩具例子值 | 真实值 |
|---|---|---|---|
| 整条序列长 | `global_len` | 12 | 64k |
| 并行卡数 | `sp_size` | 2(=ring2) | 4(=ring4) |
| **每张卡"自己那段"** | `chunk_size` = `usp_chunk_size` | 6 | 16k |
| 重叠尾长度 | `ttt_length` | 2 | k(TTT 步数) |
| **每张卡实拿(自己那段+重叠尾)** | `local_len` | 8 | 16k+k |
| 监督位集合 | `sup`(loss_mask 非零) | 见下 | —— |

> 桥句 2:**`local_len = usp_chunk_size + ttt_length`。多出来的 `ttt_length` 那截叫"重叠尾",是草稿纸,不是这张卡该负责的位置。**

### 2.4 关键:重叠尾是"草稿纸",不是"自己的地盘"

为什么每张卡要多拿 `ttt_length` 个位置?因为 TTT 展开第 j 步,位置 p 的老师(teacher)在位置 `p+j`。一张卡"自己那段"最后几个位置,它们的老师落在**下一段**里。为了不用每步都跨卡去取,发料员干脆多塞 `ttt_length` 个"下一段开头的位置"进来当草稿纸——这就是重叠尾。

**用完草稿纸就得擦掉**:对齐工 `step_view` 做完移位后,把这段**裁回 `usp_chunk_size`、丢掉重叠尾**([`eagle3_adapters.py:135-144`](../../../SpecForge/specforge/core/eagle3_adapters.py#L135-L144)):

```python
usp_chunk_size = seq_length - ttt_length
...
return StepState(
    input_ids     = global_input_ids[:, :usp_chunk_size],   # 裁回自己那段
    hidden_states = hidden_states[:, :usp_chunk_size, :],    # 裁回自己那段
    position_ids  = position_ids[:, : usp_chunk_size * self.sp_ulysses_degree],
    ...
    loss_mask     = loss_mask[:, :usp_chunk_size, :],        # loss 只在自己那段算
)
```

> 桥句 3:**重叠尾 = 下一张卡的开头。谁都不许拿重叠尾去算 loss,否则同一个位置被算两遍。对齐工负责擦掉它。**

这就是全部前提。记住三句桥:①序列并行=切段,每卡一段;②`local_len = 自己那段 + 重叠尾`;③重叠尾是下一卡的头,必须擦掉。下面看 trim 是怎么把这三条全违反的。

---

## 3. 手把手:一个 12 token、2 张卡的小例子,三处硬伤全看得见

> 教学缩参:玩具用 `global_len=12`、`sp_size=2`(即 ring=2)、`ttt_length=2`。**真实是 64k、ring=4**;缩小只为手算,机制一模一样。

### 3.1 发料员切完:两张卡各拿到什么

`chunk_size = ceil(12/2) = 6`,`usp_chunk_size = 6`,`local_len = 6+2 = 8`。

| | 全局位置区间 | 自己那段(own) | 重叠尾(下一卡的头) |
|---|---|---|---|
| **卡 0** | `[0..7]`(len 8) | `0,1,2,3,4,5` | `6,7` ← 卡1 的头 |
| **卡 1** | `[6..11]`+2 padding(len 8) | `6,7,8,9,10,11` | `12,13`(越界→padding) |

设监督位(assistant token,loss_mask=1)在**全局 `{5, 6, 7}`**。翻译到每张卡的本地下标:

| | 本地下标 → 全局位置 | 本地看到的监督位(loss_mask=1) |
|---|---|---|
| **卡 0** | 本地 0→g0 … 本地 5→g5,本地 6→g6,本地 7→g7 | 本地 `5(=g5), 6(=g6), 7(=g7)` |
| **卡 1** | 本地 0→g6,本地 1→g7 … 本地 5→g11 | 本地 `0(=g6), 1(=g7)` |

注意 **g6、g7 出现了两次**:在卡 0 是"重叠尾",在卡 1 是"自己的头"。这就是雷区。

### 3.2 正确路径(全长路径,经过对齐工)怎么算

对齐工 `step_view` 把每张卡裁回 `usp_chunk_size=6`(本地下标 `0..5`),重叠尾丢掉:

| | 裁完保留(本地 0..5) | 其中的监督位 |
|---|---|---|
| 卡 0 | g0..g5 | **g5** (本地 6,7=g6,g7 被丢) |
| 卡 1 | g6..g11 | **g6, g7** (本地 0,1) |

跨卡合起来 = `{g5, g6, g7}`,**每个恰好一次**。每张卡的 loss 分母 = `usp_chunk_size = 6`,两卡经 DDP 平均 + `accumulation_steps *= sp_size` 校正,等效于对全局 12 个位置归一。✔ 全对。

### 3.3 trim 路径怎么算(三处硬伤逐一现形)

trim 分支的关键:`state = None`,直接把**本地整段**喂给 backbone,**完全绕过对齐工**([`model.py:399-407`](../../../SpecForge/specforge/algorithms/eagle3/model.py#L399-L407)):

```python
for idx in range(self.length):
    if trim_pack is not None:
        # A 级:backbone 照样跑全长,只有监督行过 logits/loss
        state          = None
        step_input_ids = global_input_ids
        step_hidden    = hidden_states     # ← 本地整段 local_len=8,没裁!
        step_attn      = attention_mask
        step_pos       = position_ids      # ← 长度是 usp_chunk_size=6
    else:
        state = adapter.step_view(...)     # ← 只有这条 else 分支经过对齐工
        step_hidden = state.hidden_states  #   (裁回 usp_chunk_size=6)
        ...
```

而监督位 `sup` 是在**本地整段 local_len=8** 上数的([`model.py:653-655`](../../../SpecForge/specforge/algorithms/eagle3/model.py#L653-L655)):

```python
B, L = loss_mask.shape[0], loss_mask.shape[1]   # L = local_len = 8
sup = loss_mask.view(-1).nonzero(...).squeeze(-1)  # 含重叠尾里的监督位!
```

分母也用的是这个 `L`([`model.py:679`](../../../SpecForge/specforge/algorithms/eagle3/model.py#L679) 里 `full_len=L`,[`model.py:475-476`](../../../SpecForge/specforge/algorithms/eagle3/model.py#L475-L476) 里 `loss_scale=rows_j.numel()/full_len`)。

#### 硬伤 ①:backbone 喂错长度(hidden 长 8,position_ids 长 6)

trim 喂给 backbone 的 `step_hidden` 长 `local_len=8`,但 `step_pos`(position_ids)发料员只给了 `usp_chunk_size=6` 长([`preprocessing.py:488-494`](../../../SpecForge/specforge/data/preprocessing.py#L488-L494))。backbone 里注意力用**本地 hidden 长度**去算全局长度([`llama3_eagle.py:1415`](../../../SpecForge/specforge/modeling/draft/llama3_eagle.py#L1415)):

```python
global_q_len = q_len * self.sp_ring_degree * self.sp_ulysses_degree
# q_len 被 trim 塞成 8,本该是 6 → 全局长度算成 8×2=16,本该是 12
cos, sin = self.rotary_emb(query_states, seq_len=global_q_len + lck)  # 行 1421,RoPE 也跟着错
```

结果:轻则 hidden(8) 和 position_ids(6) 形状对不上直接报错,重则侥幸跑过去、但注意力按错误的全局长度(16≠12)去拼,算出一堆错的。**桥句 3 被违反:trim 动了 backbone 的输入长度,而"trim 绝不该碰 backbone"正是 A 级设计的红线。**

#### 硬伤 ②:重叠尾跨卡双算(g6、g7 被算两遍)

trim 的 `sup` 在 local_len=8 上数,把重叠尾里的监督位也收进来:

| | trim 数到的监督位(全局) |
|---|---|
| 卡 0 | g5, **g6, g7**(重叠尾也算了) |
| 卡 1 | **g6, g7**(自己的头) |
| **合计(带重数)** | g5 ×1,**g6 ×2,g7 ×2** ← 双算 |

正确应是每个 ×1(见 §3.2)。**桥句 3 再次被违反:重叠尾是下一卡的头,trim 没擦掉就拿去算 loss。**

#### 硬伤 ③:loss 分母用错(用了 8,该用 6)

trim 按 `full_len = local_len = 8` 归一;正确的全长路径按 `usp_chunk_size = 6` 归一(对齐工把 loss_mask 裁到 6,损失核对 6 个位置求平均)。差一个 `6/8` 的系数。更糟的是:跨卡靠 `reduce_loss` 归并——可它是**空操作**([`eagle3_adapters.py:158-159`](../../../SpecForge/specforge/core/eagle3_adapters.py#L158-L159) `def reduce_loss: return loss`),真正的跨卡合并是靠"每张卡用**相同的本地分母**+ DDP 平均"这个约定凑出来的。trim 用了 8 这个不一样的分母,约定被打破,**每张卡的 loss 量级和梯度都偏了,且没有按全局监督数归并**。

> 顺带说明为什么"metric 的 acc 反而是对的":`_acc_and_loss` 里 acc 那部分调了 `adapter.reduce_metrics` 做跨卡 all-reduce([`model.py:179-181`](../../../SpecForge/specforge/algorithms/eagle3/model.py#L179-L181)、[`eagle3_adapters.py:147-156`](../../../SpecForge/specforge/core/eagle3_adapters.py#L147-L156)),所以 acc 指标能跨卡加对;但 **loss 本身没走 reduce**,坏的是 loss/梯度,不是 acc 指标。这也解释了为什么坏得隐蔽——训练不一定立刻崩,acc 看着还行,loss 却在错的尺度上优化。

### 3.4 门为什么没拦住:`_trim_ok` 不看后端

按理说,`usp + trim` 这种坏组合应该被拦下。但开关 `_trim_ok` 只查四件事,**没有一件是 `attention_backend`**([`model.py:313-318`](../../../SpecForge/specforge/algorithms/eagle3/model.py#L313-L318)):

```python
_trim_ok = (
    self.trim_loss_positions
    and self.lk_loss_type is None
    and loss_mask.shape[0] == 1          # USP 本来就强制 batch==1,这条永远满足
    and int(loss_mask.sum().item()) > 0
)
```

schema 里 `trim_loss_positions` 只出现一次([`schema.py:490`](../../../SpecForge/specforge/config/schema.py#L490)),**没有任何 validator 把它和 `usp`/`sp>1` 互斥**。于是 `attention_backend=usp` + `trim_loss_positions=true` 静默走进 §3.3 的坏路——不报错,只算错。

---

## 4. 追问区(预判的坑)

**Q1:单卡明明证明了等价,为什么换成 USP 就不行?**
单卡(SDPA)路径下没有"切段"这回事——对齐工在单卡只做 TTT 移位、不做 `usp_chunk_size` 裁剪,而 trim 用自己的行选择(`rows = sup - j`)复现了那个移位,所以两边逐位相等,测试也通过([`test_equiv_trim_loss_positions.py`](../../../SpecForge/tests/test_runtime/test_equiv_trim_loss_positions.py),单进程 SDPA,seq=16)。**USP 下对齐工多干了一件事——把段裁回 `usp_chunk_size`——trim 没复现这一件,分歧就出来了。** 测试从没跑过任何 ≥2 卡的 USP,所以抓不到。

**Q2:硬伤 ① 到底是报错还是静默算错?**
取决于具体尺寸能不能整除:hidden 长 `local_len` 和 position_ids 长 `usp_chunk_size` 对不上,Ulysses 的 all-to-all reshape 若维度除不尽会直接抛形状错;若侥幸能 reshape,就按错误的全局长度算注意力,静默出垃圾。两种都不对,只是崩得响不响的区别。

**Q3:"重叠尾"和"TTT 移位"是一回事吗?**
不是。TTT 移位是"第 j 步位置 p 的老师在 p+j"这个逻辑,trim 和对齐工都各自处理了(trim 用 `rows=sup-j`)。**重叠尾**是序列并行**额外**引入的:因为段的最后几个位置的老师落在下一段,才多塞一截草稿纸。trim 处理了移位、却没处理"用完草稿纸要裁掉"——这才是 USP 独有的坑。

**Q4:g6、g7 双算,会不会正好被 DDP 平均抵消?**
不会。DDP 是对各卡梯度求平均,前提是各卡在算"同一个全局量的不同部分、且不重叠"。g6、g7 被卡0 和卡1 各算一遍,等于这两个位置的贡献被放大近 2 倍,平均之后依然偏大;再叠加硬伤 ③ 的分母错,整体 loss 尺度是错的。

**Q5:那 A 级设计本身错了吗?**
没错——A 级"backbone 跑全长、只在监督位算 teacher/logits/loss"在**单卡**是对的、且省显存。错的是**没意识到"backbone 跑全长"这句话在序列并行下含义变了**:序列并行里,"全长"对每张卡而言是"本地整段(含重叠尾)",而正确做法是"自己那段(裁掉重叠尾)"。trim 把"本地整段"当成了"全长",于是碰了本不该碰的 backbone 输入。

---

## 5. 术语表(忘了回来查)

- **序列并行 / sequence parallel**:把序列这根轴切段,分给多张卡,每卡只存一段激活。
- **USP**:SpecServe 用的序列并行方案,底层 = Ulysses(all-to-all)+ Ring attention;开关 `attention_backend=usp` + `sp_ring_size`/`sp_ulysses_size`。
- **ring size / `sp_ring_size`**:Ring attention 的组大小;maintainer 说的 "ring size 4" = `sp_ring_size=4`。
- **`global_len`**:整条序列长度(玩具 12,真实 64k)。
- **`chunk_size` / `usp_chunk_size`**:每张卡"自己那段"的长度 = `ceil(global_len/sp_size)`(玩具 6,真实 16k)。
- **`local_len`**:每张卡实际拿到的长度 = `usp_chunk_size + ttt_length`,多出的是重叠尾。
- **重叠尾 / overlap tail**:每张卡多拿的 `ttt_length` 个位置,是下一张卡的开头,当 TTT 移位的草稿纸;算 loss 前必须裁掉。
- **`step_view`(对齐工)**:USP 每步把本地段裁回 `usp_chunk_size`、丢重叠尾、摆位置号的函数。
- **`sup`**:监督位集合(loss_mask 非零处);trim 错在它是在 `local_len` 上数的。
- **`_trim_ok`**:决定走不走 trim 路径的开关;错在不看 `attention_backend`。
- **`reduce_loss` / `reduce_metrics`**:跨卡归并;前者在 USP 是空操作(靠 DDP+accumulation 凑),后者做 all-reduce。

---

## 6. 进阶:Ulysses 与 Ring 各干什么(跳过不影响理解)

§2 把"拼图工"当黑箱就够懂本文了。这里补一句它内部两层:
- **Ulysses(all-to-all)**:把"按序列切"临时换成"按注意力头切",让每张卡拿到全部序列、但只算部分头;算完再换回来。位置号要按 `sp_ulysses_degree` 展开,这就是 `position_ids` 长度是 `usp_chunk_size * sp_ulysses_degree` 的原因([`preprocessing.py:489-494`](../../../SpecForge/specforge/data/preprocessing.py#L489-L494))。
- **Ring attention**:把 KV 分块在一圈卡之间轮转传递,每张卡只持有一块 KV、轮流看到全局。`sp_ring_degree>1` 时走这条([`llama3_eagle.py:1440`](../../../SpecForge/specforge/modeling/draft/llama3_eagle.py#L1440))。

对本文的意义只有一点:**这两层都靠"本地段长度 = `usp_chunk_size`"这个假设来算全局长度([`llama3_eagle.py:1415`](../../../SpecForge/specforge/modeling/draft/llama3_eagle.py#L1415)`global_q_len = q_len × ring × ulysses`)。trim 把 `q_len` 塞成 `local_len`,这个假设就崩了——硬伤 ① 的根。**

---

## 7. 相关

- 本文管"原理:坏在哪"。**修法**(档1 设防回退 / 档2 SP-native 真修)+ 收益估算另出一份评估文档(方向定了再写,暂列在本目录 `pr-705-sp-compat-assessment.md`)。
- #705 那条 rebase+bugfix 回复:[`pr-705-reply-rebase-and-bugfix.md`](./pr-705-reply-rebase-and-bugfix.md)。
- A 级裁剪本身的 per-step mask 语义(rows = `sup - j`、teacher 钉在 `sup`)见 [`_build_trim_pack`](../../../SpecForge/specforge/algorithms/eagle3/model.py#L626-L680) 的 docstring。
