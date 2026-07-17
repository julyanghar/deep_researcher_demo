# control 约定统一为「重算 recompute」—— 改动报告

> 背景:之前 control 文件混着两套相反约定——选择性分支用 `recompute_*`(列**要重算**的),但 branch3 还用老的 `reuse_token_ranges`(列**要复用**的),且 blender 日志先打 reuse 再打 recompute,容易看反。
> 统一原则:**control 文件一律列"要重算(recompute)的"位/层;blender 内部取补集得到"要复用的"集合做 clone-revert。** 日志也 recompute 在前。
> ⚠ 全部改动**行为等价**(`test_exp_writer.py` 18 分支全过)、且**不影响正在跑的实验**(运行中的 driver/server 用内存里的旧代码,只有重启才加载新代码;即便 resume 也因等价而结果一致)。

## 改了哪些位置

### 1. `server/vllm/Exp_writer_isolation/exp_writer_select.py`(driver 端,写 control)
| 位置 | 改动 | 为什么 |
|---|---|---|
| 新增 [`_non_summary()`](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L171) | 抽出"非 summary 位(P0/分隔符),恒重算"的公共函数 | branch3 与 select_token 都要它;DRY + 统一语义 |
| [`select_token()`](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L177) | 形参 `prompt_len`→`blend_len`;改用 `_non_summary`;注释写明"返回要**重算**的位、blender 复用补集" | 命名/注释统一到 recompute 语义 |
| [branch3 :239](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L239) | `{"reuse_token_ranges": segments}` → `{"recompute_token_idx": _non_summary(...)}` | **核心**:branch3 也用 recompute 约定(只重算非 summary = summary 全复用),不再用 reuse_token_ranges |
| 顶部 docstring [:33-37](../../server/vllm/Exp_writer_isolation/exp_writer_select.py#L33) | control 三模式说明统一成"一律列要重算的";branch3 行改为 recompute_token_idx | 文档与代码一致 |

> 等价性:老 branch3 `reuse_token_ranges=[summary 段]` → 复用=summary → 复用 summary;新 branch3 `recompute_token_idx=[非summary]` → blender 取补集 复用=summary → **同样复用 summary**。测试 `branch 3 (full_reuse): 复用==全 summary 位 ✓`。

### 2. `lmcache/v1/compute/blend/blender.py`(server 端,日志)
| 位置 | 改动 | 为什么 |
|---|---|---|
| [BLEND_PATH 日志 :188-192](../../lmcache/v1/compute/blend/blender.py#L188) | 字段顺序 `reuse=.. \| recompute=..` → **`recompute=.. \| reuse=..`**(recompute 在前);判语 `TRUTH=full-prefill`/`partial-reuse` → `TRUTH=recompute-all`/`partial-recompute` | 日志与"重算"约定一致、recompute 优先 |
| layer 模式日志 [:180](../../lmcache/v1/compute/blend/blender.py#L180) | 本就是 `recompute K/L layers fully (rest fully reused)`,已 recompute 在前 | 无需改,已一致 |
| [`_controlled_reuse_indices()` docstring](../../lmcache/v1/compute/blend/blender.py#L383) | 已写明:mode1/2 = 列要**重算**的(recompute_layers/recompute_token_idx)、mode3 = 老 reuse_token_ranges(列复用)、空=重算全部 | 解析逻辑不变,只确认 docstring 把三模式约定讲清 |

> **保留 `reuse_token_ranges` 两种用途**(不删,向后兼容):① Exp A 探针轨仍用它(列复用区间);② 本实验 truth/baseline 的"全重算"由 driver 写空 `{"reuse_token_ranges":[]}` 表示(=复用 0=重算全部)。
> **内部复用集变量(`reuse`/`reuse_idx`)**:这是 clone-revert 机制本身需要的(`old_k[reuse_idx]=KV^r`),它是"control 重算集的补集",不是命名不统一——control 对外一律 recompute,内部补集自然是 reuse。

### 3. `server/vllm/Exp_writer_isolation/test_exp_writer.py`
- 无需改:branch3 断言 `复用==summary` 对新 control(recompute_token_idx)仍成立(blender 取补集)→ 测试原样通过,反而验证了等价性。

## 影响 / 注意
- **监控脚本**:正在跑的 server 日志仍是旧格式(`reuse=.. | recompute=..`),我现有的 reuse 率核对脚本对**本轮**照常有效;**下次重启 server 后**日志变新格式(`recompute=.. | reuse=..`),届时核对脚本按新字段名解析即可。
- 已 `py_compile` + `test_exp_writer.py` 全过;改动与运行中实验隔离。

*相关:[code_changes_report.md](code_changes_report.md)、[part1_code_and_test.md](part1_code_and_test.md);代码 [exp_writer_select.py](../../server/vllm/Exp_writer_isolation/exp_writer_select.py)、[blender.py](../../lmcache/v1/compute/blend/blender.py)。*
