# 本次实验的输入↔输出重叠分析(replay 口径,逐条关联实测加速)

> 目的:v2 的 overlap 统计基于**当时录制的输出**(采样温度)、且是**字符级**;suffix 实验重放用 **temp0**、吃的是 **token 级连续段**。本分析用**本实验实际的输入输出**回答:抄了多少、抄的段多长、和每条请求实测加速比对不对得上——把「碎片照搬→不加速,长段照搬→加速」从推断变成**逐条实证**。
> 数据:workload 80 条(60 summary + 20 report),replay 输出文本补捕获自现役 server(temp0);加速比 = 同 index 配对的 baseline/suffix 实测。

## 结论(四句话)
1. **temp0 重放的照搬量和录制输出几乎一模一样**(字符级 cov8:54.6% vs 54.5%)——排除"重放抄得少所以不加速"的假说,v2 的统计可直接迁移到本实验。
2. **token 级比字符级碎得多**:summary 字符级 cov8 有 54.6%,**token 级 cov8 只剩 12.9%、cov24 只剩 0.4%**(report 分别 65.6% / 44.7%)。suffix 真正能吃的,比字符级 coverage 显示的少一个量级。
3. **"可收割占比"与实测加速单调对应**(剂量-反应):harvest 6.8%→0.82x,14.7%→0.98x,28.3%→1.10x,62.7%→1.93x。**打平点在 harvest≈15-20%**。
4. **决定因素是连续可收割量,不是任务类型**:Q3 桶(1.10x)里 20 条有 17 条是 summary——**summary 里 harvest>25% 的 11 条中位 1.106x,照样加速**;而几乎无照搬的请求(harvest<5%)0.81x,**暴露 suffix 固定税 ≈18%**。

## 表1 字符级:replay 输出 vs 录制输出(同 80 条,均 vs prompt)

| | replay cov8 | 录制 cov8 | replay cov30 | 录制 cov30 | replay 最长段(中位) | 录制 最长段(中位) |
|---|--:|--:|--:|--:|--:|--:|
| SUMMARY (60) | 54.6% | 54.5% | 9.8% | 8.8% | 41 字 | 39 字 |
| REPORT (20) | 87.3% | 86.2% | 65.0% | 63.1% | 172 字 | 164 字 |

→ 两列几乎相同:temp0 重放的照搬行为 = 录制时的照搬行为(量和碎度都一致)。

## 表2 token 级(Qwen tokenizer,suffix 的真实口径)

| | tok cov4 | tok cov8 | tok cov16 | tok cov24 | 最长段(tok,中位) | 可收割占比(中位) |
|---|--:|--:|--:|--:|--:|--:|
| SUMMARY | 30.8% | 12.9% | 3.2% | **0.4%** | **14** | 14.3% |
| REPORT | 73.3% | 65.6% | 57.6% | **44.7%** | **73.5** | 62.6% |

- **可收割占比** = 已锁定(≥3 token 已匹配)后还能延续的 token 数 ÷ 输出 token 数——模拟 suffix "对上暗号后每步白捡"的机制,是比 coverage 更贴机制的指标。
- summary 的最长 token 段中位只有 **14 个 token**:锁定要吃掉 ~3 个,factor=1.0 下草稿从短养起,一段里真正白捡的所剩无几——9.2 节"三笔钱"的逐条版。

## 表3 可收割占比 × 实测加速(四分位桶,长度归一 char/s 比)

| 桶 | n(sum/rep) | harvest 均值 | 实测加速(均/中位) |
|---|---|--:|--:|
| Q1 | 21(21/0) | 6.8% | **0.82x / 0.81x** |
| Q2 | 20(20/0) | 14.7% | 0.98x / 0.89x |
| Q3 | 20(**17**/3) | 28.3% | **1.10x / 1.07x** |
| Q4 | 19(2/**17**) | 62.7% | **1.93x / 1.91x** |

三个读法:
- **单调**:harvest 越高加速越大,无反例桶——重叠(按对口径量)确实预测加速。
- **打平点 harvest≈15-20%**(Q2 附近):低于它开 suffix 纯亏,高于它开始赚。
- **固定税直读**:harvest<5% 的 6 条(几乎无肉)= 0.81x → **开 suffix 的每步固定开销 ≈18%**(异步调度被关+查树),和"调参救不了 summary"完全自洽:summary 整体 harvest 中位才 14.3%,在打平点之下。

## 代表性说明(诚实小字)
- 捕获文本与当次计时 run 的 hash 命中率低(baseline 5/80、suffix 1/80)——**temp0 在 vLLM 下本就非逐位确定**(batch 组合/浮点顺序),属预期;与 suffix run 的长度差**中位仅 4 字符**(P90 |Δ| 1098,少数长尾)。
- 所以逐条 hash 对不上,但**统计上高度代表**(表1 replay≈录制也旁证)。分桶结论(表3)用长度归一比值,对长尾稳健。

## 方法/产物
- 捕获:`capture_outputs.py`(现役 prob0.5 server,non-stream temp0)→ `replay_outputs.jsonl`(80/80 无错)。
- 分析:`analyze_replay_overlap.py`(复用 v2 的 SAM/build_matchlen;token 级用同一 SAM 跑 token id 序列)→ [per_call_overlap.csv](per_call_overlap.csv)(逐条全指标+speedup)。
- 脚本与日志在 `/home/yilin/modify-code-runs/suffix-spec-decode/`;本目录存 csv + 输出文本。
- 相关:[summary.md](summary.md)(主实验)| [questions-log.md](questions-log.md) Q9(机制详解)| [../prompt-overlap-analysis-v2/summary.md](../prompt-overlap-analysis-v2/summary.md)(录制口径全量统计)。

## 对部署判断的增量
主实验说"report 开、summary 略亏";本分析给出更细的判据:**加速与否由请求的"连续可收割占比"决定(打平点 ~15-20%),不由任务类型决定**。若未来某类调用的输出会大段连续引用输入(如引用式 QA、模板化生成),即使不是 report 也会受益;反之纯创作类调用固定亏 ~18%。
