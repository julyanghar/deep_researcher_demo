# eagle3 推理成本解剖:每步多付的钱花在哪(对标 suffix 固定税)

> **缘起**:加速实测发现现成 eagle3 head 净减速(summary 0.97×、report 0.88×,[../eagle-idea.md](../eagle-idea.md) §加速实测)。本文用 suffix 固定税同款方法,把 eagle3 每步比 vanilla 多付的 **+10ms(summary)/+19ms(report)** 拆到成分,给出税表 + break-even 公式。
> **口径**:RedHat head(AL 较优的那个)、TP4/GPU 0-3、同批真实 prompt(20 summary + 10 report,c2_spec40 harvest)、temp=0 串行回放;每步成本 = decode 总时长 ÷ 步数(步数 = draft_tokens/k,/metrics 差分)。**AL(Acceptance Length)= 每个 decode 步(一次目标模型前向)产出的 token 数 = 1(必得的 bonus token)+ 本步接受的草稿数**;AL=1 即没投机,固定 k 时 AL = 1+接受率×k;**加速比 = AL ÷ (每步成本/vanilla每步成本)**——AL 是收入、每步成本是支出。物证 `~/modify-code-runs/eagle3-test/`(sweep_status.txt、speedup_k*.{log,json}、server_k*.log、anatomy_analysis.txt)。

## 一、五臂扫描原始数据

| 臂 | k | 执行 | sum ms/步 | sum AL | rep ms/步 | rep AL |
|---|--:|---|--:|--:|--:|--:|
| k1 | 1 | cudagraph | 32.6 | 1.29 | 41.7 | 1.37 |
| k2 | 2 | cudagraph | 33.2 | 1.36 | 42.5 | 1.51 |
| k3 | 3 | cudagraph | 34.0 | 1.41 | 43.3 | 1.58 |
| k5 | 5 | cudagraph | 35.7 | 1.42 | 45.3 | 1.64 |
| k3eager | 3 | **enforce-eager** | 35.5 | 1.42 | 45.2 | 1.58 |
| k3timing | 3 | cudagraph+分段计时 | 37.4* | 1.41 | 46.5* | 1.58 |

(*timing 臂含每 propose 5 次 cuda sync 的测量膨胀 ~+3ms/步,绝对值不作数、只用分段**占比**。vanilla 基线:summary 23.5 / report 24.1 ms/步。)

## 二、斜率-截距拟合(k=1,2,3,5 四点,残差 ±0.1ms —— 模型几乎完美)

```
summary: 每步 = 31.7ms + 0.80ms × k
report:  每步 = 40.7ms + 0.90ms × k
```

## 三、eagle3 税表(对照 suffix 五常数)

| 成分 | summary | report | 是什么 |
|---|--:|--:|---|
| ① 基线(vanilla) | 23.5 | 24.1 | 读一遍 32B 权重 |
| ② **k 无关固定费** | **+8.2** | **+16.6** | 截距−基线:hidden-state 抽取/融合(combine)+ prepare + **首次 drafter 前向** + spec 管线/rejection 采样。**强 ctx 敏感**(report 的长上下文让 drafter 注意力/gather 变贵一倍)——这是 eagle3 最大的一笔 |
| ③ 每草稿位置边际费 | +0.80×k | +0.90×k | 每加一个草稿位置 = **一次额外 drafter 串行前向 + 一个 verify 位置**。对照:suffix 的每位置只要 **0.62ms**(CPU 树查找免费,只付 verify)|
| ④ 混合边界税 | **0** | **0** | **eagle3 不交这笔**:k 固定 → verify 形状稳定 → graph 正常工作;实测 eager 反而慢 1.6-1.9ms(graph 在帮它)。**与 suffix 恰好相反** |

**账本自洽核对**:k=3 预测 summary 31.7+2.4=34.1(实测 34.0 ✓)、report 40.7+2.7=43.4(实测 43.3 ✓)。

**分段账本(timing 臂,propose 内部占比)**:combine(hidden-state 融合)15% / prepare 15% / 首次 drafter 前向 24% / 后续 k−1 次前向 46%——**drafter 前向合计 ~70% 的 propose 时间**;propose 之外的固定费(rejection 采样、prepare_next_token_ids、目标前向里的 hidden-state 抽取)占②的另一半。

## 四、break-even 公式:什么样的 head 才够格

加速 = AL ÷ (每步成本/基线成本)。盈亏线(加速=1)需要:

记基线每步 $T_0$(23.5/24.1ms)、k 无关固定费 $C_{\text{fix}}$(8.2/16.6ms)、每位置边际 $c$(0.80/0.90ms),则每步成本 $T(k)=T_0+C_{\text{fix}}+ck$;恒等式 $\mathrm{AL}=1+rk$($r$=接受率)。加速比

$$S(k,r)=\frac{\mathrm{AL}}{T(k)/T_0}=\frac{(1+rk)\,T_0}{T_0+C_{\text{fix}}+ck}$$

盈亏线 $S=1$ 反解:

$$r_{\text{need}}(k)=\frac{C_{\text{fix}}+ck}{k\,T_0}$$

**算例(summary,$k{=}3$)**:$r_{\text{need}}=\dfrac{8.2+0.8\times3}{3\times23.5}=15.1\%$;实测 $r=13.6\%$ → $S=\dfrac{1+0.136\times3}{34.1/23.5}=0.97\times$ ✓(与实测吻合)。(下表数值以逐点实测每步成本代入,与拟合式差 ≤0.4pp。)

| k | summary 需接受率 | 实测(RedHat) | report 需接受率 | 实测 |
|--:|--:|--:|--:|--:|
| 1 | 38.7% | 29% | 73.0% | 37% |
| 3 | **15.1%** | **13.6%**(差 1.5pp) | **26.7%** | **19.3%** |
| 5 | 10.4% | 8.4% | 17.6% | 12.8% |

- **k=3 是甜点**(每 token 成本最低:summary 24.1ms、report 27.4ms),但**所有 k 都过不了盈亏线**——AL 随 k 饱和(summary 1.41→1.42,k5 白加成本),成本却线性涨。
- **要 2× 加速**(k=3,summary):AL ≥ 2.89 → 接受率 ≥ **63%**——chat 域自训 head 论文水平 AL 2.5-4,**本域自训理论可及但要求极高**;现成 head(AL 1.3-1.6)差得远。

## 四b、对照 suffix:每步成本打平,输在收入(追问回填)

**"为什么比 suffix 还慢——drafter 前向应该很便宜?"** 把两边每步账并排(同批 prompt):

| | 每步成本 | AL(每步收入) | 每 token |
|---|--:|--:|--:|
| suffix(Ceager)summary | 34.4ms | **2.77** | 12.4ms |
| eagle3(k3)summary | 34.0ms | 1.41 | 24.1ms |
| suffix report | 46.0ms | **4.61** | 10.0ms |
| eagle3 report | 43.3ms | 1.58 | 27.4ms |

**数据来源(suffix 行)**:同一天受控回放的 suffix 臂(`speedup_suffix.{json,log}`,同批 prompt、同 TP4/0-3,server=Ceager+SUFFIX_TRAJ)。**每步是反推的**:suffix 每步吐 1+acc_i 个 token → **步数 = gen − acc**(gen 来自逐调用 usage、acc 来自 /metrics 差分);每步成本 = Σdecode_s ÷ 步数(decode_s 为 server 侧 timing)。复算:summary steps=11830−7553=4277、146.9s→34.4ms、AL=2.77;report steps=25000−19582=5418、249.5s→46.0ms、AL=4.61。推导物证:`~/modify-code-runs/eagle3-test/per_step_suffix_derivation.txt`。两点口径注意:①suffix 臂带 traj 写盘微开销(方向保守);②suffix 是**部署配置**(树深 24 自动 k、变长草稿),不是与 eagle 同 k——本表是"部署态 vs 部署态"的比较。

**每步成本两者几乎打平**(suffix 交 eager 附加+激活 5.8+0.62×位置;eagle3 交 drafter 前向费——凑巧差不多),**差距全在 AL:收入差 2-3 倍**。suffix 靠上下文树做"逐字记忆"在照抄段命中 60%,一层小模型从分布里猜不出 2500 token 的逐字引用——**败因是现成 head 猜得差,不是 drafter 太贵**。

**"权重加载"假设澄清**:没有每步"加载"(权重启动即常驻 HBM);每次前向的真实成本是 HBM 权重**读**——head 3.1GB,每前向实读 ~1.3GB(embed 查表不全读)÷4 卡 ≈ **~0.5ms/前向带宽地板**(估算)。这解释了边际费 0.80-0.90 比纯 verify 的 0.62 多出的那截;真实但小,**不是主因**。"参数量小=免费"的直觉错在:1.5B ≈ 基座每步读量的 5%,×(k+1) 次串行前向,再叠 k 无关固定费(hidden-state 融合/prepare/rejection),堆起来正好吃光 AL 1.4-1.6 的收入。

## 五、判决

1. **eagle3 的钱 70% 花在 drafter 自己身上**(k 次串行小模型前向 + hidden-state 抽取/融合),这是**结构性成本**,不是配置能调掉的;
2. **固定费强 ctx 敏感**(summary 8.2 → report 16.6):上下文越长 drafter 越贵——恰好我们 report 上下文长,雪上加霜;
3. **eagle3 没有混合边界税**(与 suffix 相反):graph 正常发挥,`--enforce-eager` 对它是**负优化**(慢 4%);
4. **每位置边际 0.80-0.90ms vs suffix 0.62ms**:suffix 草稿免费(CPU 树)、eagle 草稿要 GPU 前向——**同样的接受率下 suffix 永远更赚**;
5. 结合接受率实测:现成 head 在本 workload **无解**(k 扫遍也过不了盈亏线);自训本域 head 需把接受率从 14%/19% 拉到 **>15%/27% 才保本、>63% 才 2×**。

## 六、诚实边界

- timing 臂绝对值含 sync 膨胀(+3ms/步),只用占比;拟合与核对全用无埋点臂。
- AL(k) 与接受率来自 RedHat head;AngelSlim 更低(见 [../eagle-idea.md](../eagle-idea.md)),结论只会更差。
- 每位置 0.80/0.90 是"drafter 前向+verify"合并斜率,两者未再分拆(要分需在 eager 臂扫 k,未做——对判决不承重)。
- EAGLE_TIMING 埋点保留在 [eagle.py](../../../../../anaconda3/envs/lmcache/lib/python3.12/site-packages/vllm/v1/spec_decode/eagle.py)(env 门控,默认零行为变化;备份 `eagle.py.bak_timing`)。

## 相关

- 接受率+加速实测(四臂对照表):[../eagle-idea.md](../eagle-idea.md)
- suffix 固定税(方法论出处):[fixed-tax-conclusions.md](fixed-tax-conclusions.md) · [mixed-mode-tax-explained.md](mixed-mode-tax-explained.md)
- 为什么 eagle3 不交混合税(桶/锁原理):[cuda-graph-lock-from-zero.md](cuda-graph-lock-from-zero.md) 追问区 Q6
