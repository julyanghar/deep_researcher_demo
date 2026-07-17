# 诚实边界(caveat)

← 返回 [experiment-data-summary.md](../experiment-data-summary.md)

1. **两变量同变(最重要)**:DRBench(中文,11K 字符)与 DRGym(英文,40K 字符)语言和长度都不同,**不能断言单一变量导致 [03](03-drgym-long-prompt.md) 的反转**,只能说"在英文超长检索 prompt 的 agent summary 上 EAGLE 碾压 suffix"。机制分析支持长度是主因(suffix 草稿税 ∝ 输入长度,扫更长输入找匹配),但**未做控制变量实验**隔离语言效应——这是后续该补的实验(如中文长 prompt 或英文短 prompt);

2. **接受率 41.2% 为混合值**(含训练集内 546 条偏高);加速比已分层,干净 126 条的 1.303× 为公平数;过拟合 gap 极小(+0.036×,见 [04](04-gain-decomposition.md))佐证干净数可信;

3. **DRGym 干净样本仅来自 7 题**(126 条 summary),题目多样性有限;加速比稳定但**题级泛化待更多干净题验证**(受限于"有 search 缓存 + 未进训练"的题只有 7 个,补更多需重新 online 建池);

4. **加速比可能被超长 prefill 稀释**:40K token prefill 占墙钟大头,spec decode 只加速 decode 段,故报告的绝对加速比是**保守下界**,真实 decode 段加速更高——但对 EAGLE-vs-suffix 的**相对结论无影响**(同题两 server 同 prefill);

5. **DRBench 与 DRGym 是不同 run**(不同时间、机器状态),各自内部同 run 可比,跨数据集比较以"vs vanilla 相对值"为准,不比绝对墙钟。
