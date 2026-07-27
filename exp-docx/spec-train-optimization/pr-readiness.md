# PR 就绪度评估(2026-07-16)

回答"现在能否提交一个 PR"。结论:**能——ropebuf bugfix 现在就能提**;A 级要清理才行、B 级先 RFC。

- 依据:fork `julyanghar/specforge-yilin` @ 78a6432(含 A+B-i+B-ii+MCMEM);上游 `sgl-project/SpecForge` main @ 40d8fef(#648)。
- 定位见 [contribution-positioning.md](contribution-positioning.md)。

## 一、上游兼容:意外地好(不用怕 rebase)

- 落后上游 **130 commit**,但 `merge-base == 我们的基线 357a97e` → **干净 3-way merge,历史没缠**;
- **eagle3.py(A 级靶文件):上游 130 commit 里 0 次改动** → A 级 rebase 零干扰;
- **train_eagle3.py(ropebuf 靶文件)**:上游只 1 次改(grad-norm 去重,在训练循环、不碰 build_draft_model)→ **自动合并,ropebuf 幸存**;
- **llama3_eagle.py(B 级靶文件)**:上游只 1 次改(draft registry `@register_draft`,加 2 行、不重叠)→ **自动合并**;
- **唯一真冲突** = `eagle3_target_model.py`(上游把它重构成 import shim,我们那 5 行 chunked-prefill 补丁撞上)——**但它不属于任何一个 PR**,机械改到新家即可。

## 二、三个候选 PR 的就绪度

| PR | 判定 | 工作量 | 触及文件 |
|---|---|---|---|
| **ropebuf bugfix** | ✅ **现在能提** | ~30-60min | 1 文件(train_eagle3.py) |
| A 级(--trim-loss-positions) | ⚠️ 清理后能提 | 半天 | 2 文件(eagle3.py + flag) |
| B 级(--trim-prompt-rows/step1) | 先 RFC | — | 5 文件 |

### ropebuf(现在能提)
- **真 bug**:meta-device `from_pretrained` 不回填 `persistent=False` 的 RoPE buffer(inv_freq/cos_cached/sin_cached)→ warm-start(`--ckpt-dir`)loss=nan,影响所有 `--ckpt-dir` 用户;
- 修复全在 [train_eagle3.py](../../../SpecForge/scripts/train_eagle3.py) build_draft_model(~583-595):from_pretrained 后遍历 modules 调 `_init_rope()` 再 `.cuda()`;
- **`_init_rope` 上游已存在** → **PR 零改 llama3_eagle.py**,是个单文件 diff;
- **待办**(提前):① 从 origin/main 拉干净分支,手工只贴 ropebuf 这一段(不能 cherry-pick,它混在 commit 77f4a0f 里)② print 挪出 per-module 循环(打一次)③ 中文注释改英文、讲清 persistent=False 根因 ④ 加回归测试(save→reload→断言 rotary buffer finite 且 == 新 _init_rope 参考;CPU CI 未必稳定复现 nan,用相等性更稳)⑤ 单文件 diff 复核。

### A 级(清理后能提,半天)
- 功能完整、SC4 自检已证等价(最差 2.6e-3)、与 B 级可分;但要:
  - eagle3.py 剥 ~72-126 行 debug 脚手架(MCTRIM_SELFCHECK/MCDBG/SC4/SCB/_sc_ref)、删所有 B 级代码;
  - **revert llama3_eagle.py + flex_attention.py 到上游**(A 级不需要它们——A 走全长 backbone、trim_ctx=None);
  - 把 4 个无关 bugfix(ropebuf/vocab-mapping/chunk-acc/chunked-prefill)拆成各自的 commit;**nocompile/ckpt-norm 不进 A(是 B 的前置)**;
  - 加等价性测试(把 SC4 oracle 产品化)+ fallback 断言;
  - **诚实措辞**:标"port TorchSpec 的损失末端裁剪",不吹原创。

### B 级(先 RFC)
- 5 文件、含 backbone 内部改动,按计划先 RFC——主打算量/吞吐 + agent 数据动机 + 与 #669 正交,别主打省峰值(见定位文档 §四)。

## 三、动手前的前置(非技术)

- **实验室开源署名政策**:能否以个人/机构名义向上游贡献?(计划 §3.6 已列为前置,提 PR 前须确认)
- ropebuf 提 PR = 对上游仓库的**外向、不可逆**动作,用你的 GitHub 身份——准备分支我可以做(本地),**开 PR 由你点**。
