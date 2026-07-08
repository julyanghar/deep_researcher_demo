# online100_v2 分析1：同题内 summary(researcher) 之间输入/输出重复

- 单元 = 每条 RESEARCH_SUMMARY_TEXT（1812 条）；同题两两配对，共 18975 对。
- 单元 = 单个 sub-query 的摘要；同 research_question 的多个单元 = 同一 researcher。

## 逐字覆盖 vs n-gram Jaccard（两种输出重复口径的区别）

- **逐字覆盖 out_contain8**：短的一方 summary 里，有多少字落在'与另一方完全相同的 ≥8 连续字符'块内（占比）。要求**连续、完全一致**，抓**硬照抄/复制粘贴**，保守精确。
- **n-gram Jaccard out_jac5/3**：把两条 summary 各切成字符 n-gram **集合**，算 |交|/|并|。**无序、看集合**，容忍改写/语序，抓**'讲同一件事'**；n 越小越宽松（也越易被套话抬高）。
- 互补：逐字漏掉的'换了说法但同义'由 Jaccard 补上。

## 全体配对分布

| 指标 | mean | median | p90 | max |
|---|---|---|---|---|
| url_jac | 2.8% | 0.0% | 12.5% | 100.0% |
| url_shared | 0.19 | 0.0 | 1.0 | 6 |
| snip_contain8 | 27.6% | 27.0% | 59.1% | 100.0% |
| out_contain8 | 22.9% | 20.5% | 48.3% | 90.6% |
| out_jac5 | 7.7% | 6.1% | 16.5% | 48.6% |
| out_jac3 | 18.6% | 12.2% | 38.2% | 66.4% |

### 分组：same_rq=同一researcher(不同sub-query)  (n=1812)

| 指标 | mean | median | p90 |
|---|---|---|---|
| url_jac | 5.0% | 0.0% | 16.7% |
| url_shared | 0.34 | 0.0 | 1.0 |
| snip_contain8 | 32.9% | 38.5% | 65.2% |
| out_contain8 | 34.6% | 35.7% | 60.5% |
| out_jac5 | 12.1% | 11.1% | 22.7% |
| out_jac3 | 24.2% | 23.6% | 44.0% |

### 分组：diff_rq=不同researcher  (n=17163)

| 指标 | mean | median | p90 |
|---|---|---|---|
| url_jac | 2.6% | 0.0% | 12.5% |
| url_shared | 0.17 | 0.0 | 1.0 |
| snip_contain8 | 27.1% | 25.3% | 58.2% |
| out_contain8 | 21.7% | 18.9% | 46.1% |
| out_jac5 | 7.2% | 5.2% | 15.6% |
| out_jac3 | 18.0% | 10.2% | 37.4% |

### 分组：same_round=同一轮  (n=7176)

| 指标 | mean | median | p90 |
|---|---|---|---|
| url_jac | 3.2% | 0.0% | 14.3% |
| out_contain8 | 25.2% | 23.2% | 52.1% |
| out_jac5 | 8.5% | 7.5% | 18.1% |

### 分组：cross_round=跨轮  (n=11799)

| 指标 | mean | median | p90 |
|---|---|---|---|
| url_jac | 2.6% | 0.0% | 12.5% |
| out_contain8 | 21.6% | 18.6% | 45.8% |
| out_jac5 | 7.2% | 5.0% | 15.5% |

## 高重复配对占比

| 阈值 | 占比 |
|---|---|
| out_contain8 > 30% (硬照抄) | 37.3% |
| out_jac5 > 30% | 0.4% |
| out_jac5 > 50% | 0.0% |
| url_shared >= 1 (搜到同一页面) | 14.6% |
| url_jac > 20% | 3.7% |

## 每题检索冗余（被 ≥2 个单元共享的 URL 占比）

- 跨题 mean=16.7%, median=15.4%, p90=28.6%, max=44.4%

> 明细见 per_pair_metrics.csv / per_summary_meta.csv；典型冗余对见 redundancy_examples.md