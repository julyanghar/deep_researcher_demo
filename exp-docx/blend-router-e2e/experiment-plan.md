# 实验方案:CacheBlend KV 复用 × spec router 的 TTFT / e2e / 质量三口径测量

> 状态:方案 v3。v2=A 臂改纯 vanilla(无投机、report 单次生成);v3=**B 臂 blend 改用 attention-guided 方法**(用户拍板"attn-guided 没问题了"=交接闸终审放行)。摸底证据:五份并行调查 + 差距核查,关键行号实核于 2026-07-20 代码现状。

## 1. 目的与指标

在真实 DRAgent(deep_researcher_demo)工作负载下,测 CacheBlend KV 复用:

| # | 问题 | 指标 | 数据源 |
|---|---|---|---|
| 1 | KV 复用省多少 TTFT | supervisor decide 调用的 ttft_s/prefill_s,A vs B 同题同轮配对;辅以 BLEND_PATH reuse% | 每题 `llm_calls.jsonl`(需 server 带 VLLM_RESP_TIMING=1)+ server 日志时间窗对齐 |
| 2 | 逐题 end-to-end 效率 | 逐题 wallclock 配对差 + bootstrap CI;另报两臂总时长 | reports.jsonl 时延字段 / harvest 逐题记录 |
| 3 | 最终 report 质量 | DRBench RACE 分,配对差,须超过评委自漂移 | `run_drbench.py --mode score --metrics race`,评委 gemini-3.5-flash |

**归因口径(重要)**:A、B 相差三件事(blend KV 复用 / router 投机 / report 分节),所以——
- TTFT 差 → **干净归因 blend**(投机只影响 decode 不影响 prefill;eager 对 prefill 影响很小);
- e2e 差 → 三项叠加的**系统级收益**,不做单项归因(要归因需消融臂,列为可选后续);
- 质量差 → 投机是无损采样;三步分节有 RACE 无损先例(e2e-3config-40q.md:99-108),但 B 臂实际用的是**四步(+审阅+补节,无先例)**——故质量差指向 **attn-guided blend 与新审阅机制的混合**,无法进一步拆分(用户知情拍板"实现并直接进 B 臂";如需拆分归因,后续单独消融)。

## 2. 两臂定义

| | A 臂(vanilla) | B 臂(cacheblend) |
|---|---|---|
| server | 裸 `vllm serve`:无 LMCACHE env、无 kv-transfer-config、**无投机、不 enforce-eager**(vanilla 默认 cudagraph) | LMCache blend(LMCacheConnectorV1)+ APC 常态开 + **router 投机**(锁 enforce-eager,算 B 的系统代价) |
| prompt 布局 | 原始(不设任何 SEPARATOR) | blend 布局:`KV_REUSE_SEPARATOR='<|fim_pad|>'` + `FINAL_KV_REUSE_SEPARATOR=`(空) |
| blend 作用面 | 无 | ① researcher summary 生成 KV 入库(`lmcache.blend_store_generated`,agents.py:527);② supervisor prefill 复用(复用布局 agents.py:306-319 + warmup workflow.py:63-76);**writer 全量 prefill**(空 sep → 布局问题在前 + 无 warmup → blend 从 0 起连续命中必 miss) |
| blend 修正方式 | — | **attention-guided**(`LMCACHE_BLEND_ATTN_GUIDED=true`):prefill 恒走 gaps-only+floor1(不做 top-k 抢名额),真·首 decode 步影子前向取注意力 → score=a×‖Δv‖ 选 top-B → 因果精确修正回写 paged;修正落第 2 个 decode 步 pre-forward |
| report 生成 | **单次调用**(REPORT_MODE=detailed_cited,不设 REPORT_PARALLEL) | **分节+审阅+补节**:REPORT_PARALLEL=1 + REPORT_REVIEW(**默认开**,终审拍板)——outline→各节并行→审阅(全文入,出终稿大纲)→补节并行→代码按序拼;实现经 modify-code 全验收+终审 DONE(review-report.md) |
| 客户端路由 | 不设 PROPOSER_ROUTING(payload 原样) | PROPOSER_ROUTING=1(summary→eagle 域训头,其余→suffix) |

共同设置:题间并行 1(`--concurrency 1`)、题内 3 迭代 × 3 researchers × 3 subqueries × 题内并发 3(`MAX_ITERATIONS/MAX_FOLLOWUPS/MAX_QUERIES_PER_RESEARCHER/MAX_CONCURRENCY=3`,全是默认值但显式钉死)、`RESEARCH_MODE=local`、drbench 前 40 题(决策点 1)、`REPORT_MODE=detailed_cited`(A 臂生效;B 臂被 REPORT_PARALLEL 优先覆盖,agents.py:744-748)。

## 3. 为什么这样设计(取舍说明)

- **A 臂 = 用户当前会部署的裸基线,B 臂 = 全优化栈**:这是"系统 vs 系统"对比,回答"整套上了能省多少";不是单变量消融。TTFT 仍可干净归因(见 §1)。
- **两臂都开 APC**(vLLM 默认):A 臂常量前缀走 APC;B 臂"从 0 连续命中区"由 APC 接管、blend 只管中段 summary 段(a8baa3de 五改动 + 盐补丁,B 臂必须 `VLLM_DEFAULT_CACHE_SALT=cacheblend`)。B 的 TTFT 增益因此干净地 = "中段 summary 段复用"。
- **writer 不用 blend 的保证机制**:不是"prompt 里没分隔符"——现行代码 researcher 用分隔符连接 sub-summaries(agents.py:633-634),writer prompt **必然含** `<|fim_pad|>` 字面文本。真正保证是 FINAL sep 为空 → writer 用原始布局(问题在前,agents.py:712-715)且不做 writer warmup(agents.py:719-720)→ segment 0 逐题不同 → blend lookup 必 miss → 全量 prefill。验证见 §7 断言 6。
- **researcher 侧不显式关 blend**:跨题 summary 内容哈希天然 miss;题内 researcher 常量指令前缀两臂都由 APC 接住,公平。分析时 researcher 调用单列核实,若出现非预期 BLEND_PATH 命中,如实记录进结果而非忽略。
- **驱动必须走 `run_drbench.py --harvest`**:角色级 SEPARATOR env 只在 cli.py 路径生效(cli.py:106-111);非 harvest 的 generate() 走 build_workflow 只认全局 sep(run_drbench.py:58-61 → run_deepsearchqa.py:379-384)。harvest 模式每题起 `python -m deep_researcher_demo` 子进程、继承全部 env(harvest_gen.py:52-56),且每题独立 `q<i>/llm_calls.jsonl`。两臂同用 harvest 路径保口径一致。
- **SUFFIX_TRAJ 只在 B 臂开**(A 臂无投机,无轨迹可记;纪律"凡 suffix 投机实验必开"只约束 B)。其写盘开销只在 B 侧,方向是让 B 略吃亏,不会虚报 B 的收益——可接受。MCPROUTE_DBG 只在冒烟开,正式跑必关。
- **report 生成 = 四步新机制(用户拍板推翻"本次不动"的原建议)**:审阅+补节已实现并进 B 臂(REPORT_REVIEW 默认开,显式 0 才关;全验收+终审 DONE,物证 [review-report.md](../../../modify-code-runs/report-review-supplement/review-report.md),机制讲解 [risks-explained-report-review.md](risks-explained-report-review.md))。代价已知情:与 c3 三步分节口径不再完全可比、质量差含新机制变量(见 §1 归因口径)。SUMMARY_PARALLEL 同日标废弃,分节只用于 report。

## 4. Server 起法

统一:`/home/yilin/anaconda3/envs/lmcache/bin/vllm`(conda run 是坏的);`CUDA_VISIBLE_DEVICES=1,3,4,5`(**严禁 GPU 6**,start.sh 默认含 6 不能直接用);port 30000;TP4;Qwen3-32B;`PYTHONHASHSEED=0`;`VLLM_RESP_TIMING=1`。

**A 臂**:直接用现成 `/home/yilin/LMCache/server/vllm/config_no_lmcache.yaml` **原样**(无投机、无 eager、无 kv-transfer-config),不设任何 LMCACHE env、不设盐。

**B 臂**:新建 `config-b-blend-router.yaml`(放本目录)= `server/vllm/config.yaml`(含 kv-transfer-config LMCacheConnectorV1,**不加** no-enable-prefix-caching)+ 两行:
```yaml
enforce-eager: true
speculative-config: '{"method":"router","model":"/home/yilin/deep_researcher_demo/phase2-12k-3ep","num_speculative_tokens":3}'
```
B 臂 env 全套(start.sh:15-26 为底,两处覆盖),外加 `SUFFIX_TRAJ=/home/yilin/tmp/blend_e2e/traj_b`:
```bash
export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=true LMCACHE_MAX_LOCAL_CPU_SIZE=16
export LMCACHE_ENABLE_BLENDING=true LMCACHE_USE_LAYERWISE=true
export LMCACHE_SAVE_UNFULL_CHUNK=true LMCACHE_SAVE_DECODE_CACHE=false
export LMCACHE_BLEND_SPECIAL_STR='<|fim_pad|>'      # 覆盖 start.sh 的 "# #";必须与客户端逐字节一致
export LMCACHE_BLEND_CHECK_LAYERS=1
export LMCACHE_BLEND_ATTN_GUIDED=true                # v3:attention-guided 修正
export LMCACHE_BLEND_RECOMPUTE_RATIOS=0.2            # 决策点 2:attn-guided 语义下 = decode 修正预算 B(prefill 恒 gaps-only);0=影子专用不修正,别用
export VLLM_DEFAULT_CACHE_SALT=cacheblend            # APC 兼容硬前提
# 不设:CONTROL 两 env(原生模式);AG 调试旋钮(AG_DEBUG_LOGITS/DUMP/POS_OFFSET/FORCE_TAIL)全关
```
无 GPU config 冒烟(起 server 前先跑,应打印 `True attn_dv`):HANDOFF §1b 的一行命令(`LMCACHE_ENABLE_BLENDING=... python -c "from lmcache.v1.config import LMCacheEngineConfig; ..."`)。

## 5. Driver 命令(每臂)

```bash
cd /home/yilin/deep_researcher_demo
# 共同 env
env RESEARCH_MODE=local SEARCH_BENCHMARK=drbench \
  MAX_ITERATIONS=3 MAX_FOLLOWUPS=3 MAX_QUERIES_PER_RESEARCHER=3 MAX_CONCURRENCY=3 \
  REPORT_MODE=detailed_cited \
  OPENAI_BASE_URL=http://localhost:30000/v1 MODEL=Qwen3-32B \
  <B臂追加: REPORT_PARALLEL=1 PROPOSER_ROUTING=1 \
            KV_REUSE_SEPARATOR='<|fim_pad|>' FINAL_KV_REUSE_SEPARATOR= \
            KV_REUSE_TOKENIZER=/data/yilin/huggingface/Qwen3-32B> \
  /home/yilin/anaconda3/envs/gpt-deep/bin/python eval/DeepResearchBench/run_drbench.py \
    --tag <a_van40|b_blend40> --mode generate --harvest --n 40 --concurrency 1 \
    --base-url http://localhost:30000/v1 \
  > /home/yilin/tmp/logs/drbench_<臂>.log 2>&1
```
- A 臂**不设** REPORT_PARALLEL / PROPOSER_ROUTING / 任何 SEPARATOR。
- `KV_REUSE_TOKENIZER`:summary 触 max_tokens 截断时客户端按 token_ids[:-1] 重解码,对齐服务端只存 M-1 token 的段(llm.py:216-219),B 臂必带。
- 结果自动落 `eval/results/drbench/<tag>/q<i>/{report.md,llm_calls.jsonl,harvest.jsonl}`。

## 6. 打分(gemini-3.5-flash)

- key 已存 `/home/yilin/tmp/blend_e2e/judge.env`(**不进 git 仓库**,打分前 `source` 它;judge_adapter 读 os.environ 优先于 .env,judge_adapter.py:35-38)。零代码改动:打分链路是裸 OpenAI 兼容 POST,走 Google OpenAI 兼容层。
- 命令:`source /home/yilin/tmp/blend_e2e/judge.env && python eval/DeepResearchBench/run_drbench.py --tag <tag> --mode score --metrics race`(gpt-deep 绝对路径 python)。
- **评委自一致性(纪律,qwen-flash 自漂 0.236 教训)**:同一臂报告用 gemini 打两遍,算 mean|Δ|;A−B 效应必须显著超过该自漂移才可下结论。
- 冒烟(拿分前必做):① `curl {base}/models` 核实模型名(按 `gemini-3.5-flash` 试,以 /models 为准);② 1-2 题试打,盯两个兼容风险:RACE payload 的 `reasoning_effort`+`max_completion_tokens=64000`(api.py:169-174,必要时 `MAX_OUTPUT_TOKENS` 调小)与 thinking 吃 token 导致空响应。
- B 臂报告正文可能被模型抄进 `<|fim_pad|>` 字面文本(writer findings 里带):打分前 `grep -l` 清查,若出现,两臂同规则 strip 字面分隔符后再打分,并记录发生率。

## 7. 冒烟与断言清单(正式跑前逐条过)

B 臂是**三重栈首次同开**:blend + attn-guided 修正 + router 投机。两两之间各有未实测面——blend+router 零运行记录(文件级不冲突有证据);**attn-guided × 投机解码交互零实测且有真实风险**:attn-guided 依赖"真·首 decode 步"影子前向取 t1、修正落第 2 个 decode 步 pre-forward,而投机解码的 decode 步是 draft+verify 多 token 步,时序假设是否还成立没验过。另外 attn-guided 此前验收只到 TP2(rerun-5-32b-ag),TP4 首次。所以 B 臂冒烟不可省、且要盯细:

1. 预检补丁在位(重装 vllm 会全丢):router 4 文件 sha256 对 README:85-88;盐补丁 `input_processor.py:326`;`VLLM_RESP_TIMING` 在 serving.py;config 冒烟打印 `True attn_dv`(§4)。
2. B server 冒烟(带 `MCPROUTE_DBG=1`)起后跑 1 题,断言:`MCPROUTE_init suffix=1 eagle=1 aux_hidden=True`、`[SUFFIX_CFG]` 行、`BLEND_PATH=AttnGuidedPrefill` 行(attn-guided 开启后标签不再是 CacheBlend,blender.py:296)、`ATTN_GUIDED job ... sel_check=ok` 出现、**`grep "ATTN_GUIDED job failed"` 计数=0**(修正失败会被静默吞掉降级纯复用——"没崩"不算过)、**`MCPROUTE_pick` 逐请求核实路由零错配:仅 RESEARCH_SUMMARY_TEXT→eagle,其余全部→suffix**(用户钉死的策略;server 对未标记请求默认 eagle,PROPOSER_ROUTING=1 下所有调用都带标记,不应出现未标记请求)、无崩溃。**若修正任务在投机解码下全部 failed 或时序断言异常:停下向用户报告,不硬跑**。过了之后**去掉 MCPROUTE_DBG 重启**再正式跑。
3. A 臂生效断言:server 日志 `speculative_config=None`、无任何 LMCACHE/BLEND 行;llm_calls 里 `FINAL_REPORT_OUTLINE_JSON` 出现数 = 0(单次生成)。
4. B 臂分节+审阅断言:`FINAL_REPORT_OUTLINE_JSON` >0(repOL>0)、summary 无 outline(sumOL=0,SUMMARY_PARALLEL 已废弃防误开)、每题 `FINAL_REPORT_REVIEW_JSON` 恰 1 条、`FINAL_REPORT_ADDITION_MARKDOWN` ≥0 条(计数入结果);spec 断言 `method='router'`。冒烟题**人工目检**真实审阅 JSON(顺序合理性/漏列率)与补节正文质量(风险⑤),不过关即停。
5. 每臂跑完:40/40 报告齐;reports.jsonl 的 `search_cache_miss` 全 0(local 模式没走网)。
6. B 臂 writer 不走 blend:harvest 里 writer user prompt 以 `<original_question>` 开头、findings 外层连接符是 `\n\n---\n\n`;server 日志按 FINAL_REPORT_* 调用时间窗对齐,窗口内**无** BLEND_PATH 行(BLEND_PATH 无 req_id,用邻行 `Blend ... req <id>`(vllm_v1_adapter.py:965-971)交叉核;TP4 一次 blend 打 4 行去重)。
7. B 臂 supervisor 走了 blend:SUPERVISOR_DECISION_JSON 调用窗口内有 `BLEND_PATH=AttnGuidedPrefill` 行且 reuse 覆盖 summary 段量级合理;正式跑全程 `ATTN_GUIDED job failed` 占比统计进结果(理想为 0,非 0 时报告降级率)。

## 8. 分析口径

- TTFT:排除 warmup 调用(max_tokens=8);supervisor decide 按 (题, 轮) 配对 A−B;报 ttft_s 均值差、分位数、reuse% 分布;省算 token 量 = Σ(BLEND_PATH reuse tokens)。
- e2e:逐题配对差 + bootstrap CI;B 臂额外成本单列(blend fuse 时间 `Blend (KV load + fuse) took X ms`、final-save、attn-guided 影子/修正开销——32B TP2 warm 参考值 shadow=41ms/corr=57ms,TP4 实测为准);结论措辞 = "全优化栈系统收益",不单项归因。
- 质量:RACE overall + 分维度,配对差 vs 评委自漂移;归因按 §1 口径(主要指向 blend,标注推断)。
- 日志:`/home/yilin/tmp/logs/`(server_{a,b}.log、drbench_{a,b}.log);数据:`/home/yilin/tmp/blend_e2e/`;每轮独立命名防追加污染。

## 9. 决策点(全部已定,2026-07-20 用户拍板)

1. **题库 = drbench 前 40 题**(已定;local 缓存全、c1/c2/c3 先例同口径)。
2. **LMCACHE_BLEND_RECOMPUTE_RATIOS = 0.2**(已定;attn-guided 语义 = decode 修正预算 B,prefill 恒 gaps-only+floor1)。
3. ~~LMCache 脏工作区~~ **已完成(2026-07-20)**:attn-guided 终审放行并已提交——代码 `accc5097`、资料+HANDOFF `8b01948b`(基线 6466b0d0 之上),`git status --short` 已核干净。B 臂实验代码状态 = `8b01948b`。
4. **评委模型名**:按 `gemini-3.5-flash` 走,冒烟以 `/models` 实测为准。

## 10. 执行顺序与预算

第 0 步(已完成):LMCache 改动全部提交(`accc5097`+`8b01948b`)、`git status --short` 干净 → 预检+冒烟(≈1h)→ A 臂 generate → 换 server 跑 B 臂 generate(每臂粗估 3-5h,40 题串行,以实测为准;A 臂无投机会比先例 vanilla 慢口径略近 c1)→ RACE 打分 A、B + 自一致性第二遍 → 分析出报告。杀 server 按坑 4:nvidia-smi 拿 worker PID、`ps -o user=` 验属主再 kill -9,别 pkill -f。GPU 现状(2026-07-20):1/3/4/5/7 空闲,2/6 忙(6 训练严禁触碰)。

## 11. 附:report 分节机制现状 + "排序/补节后处理"提议评估

### 现状(REPORT_PARALLEL=1 的完整链路)

入口 `FinalWriter.write`(agents.py:744-745)→ `_write_parallel`(agents.py:797-827)→ 通用引擎 `outline_then_parallel`(agents.py:146-233):

1. **备料**(agents.py:803-808):每条 finding(= researcher 的 combined summary)拼成 `"sources: <URLs>\n<正文>"`,编号 [1]..[N] 进池。
2. **outline 调用**(1 次串行,tag=FINAL_REPORT_OUTLINE_JSON,agents.py:201-213):prompt 要求"清晰、不重叠的分节;每节给 title + 该节要用的 finding ids"(_OUTLINE_SYSTEM,agents.py:688-693);返回 JSON `{sections:[{title, source_ids}]}`(schemas.py:28-40);截最多 8 节(agents.py:158,213)。**节的集合和先后顺序在这一步一锤定音**。解析失败/空 → 回退单次全量生成(agents.py:217-220)。
3. **各节并行生成**(agents.py:224-226 gather,无并发上限):每节独立一次调用,只喂分给它的 findings(agents.py:173-175;id 空/越界则喂全部);system = "只写本节正文" + sibling-aware 约束(_SECTION_FOCUS,agents.py:248-253:只覆盖本节、不重复他节、不写 intro/conclusion、不加标题、不加 references);user prompt 里带**完整 outline 标题列表** + 本节名(agents.py:181-185),让每节知道全局结构以避免越界。每节 max_tokens=5000(agents.py:820)。
4. **拼接 = 确定性代码,零 LLM**(agents.py:227-233):按 outline 顺序 `"## {title}\n\n{body}"` 硬拼,空节跳过;References 也是确定性生成——全部 finding URL 去重罗列(agents.py:823-827),不经模型。

### 对"生成后再一次调用:排序 + 补节 + 按序拼"的评估

- **排序**:现状已有——顺序由 outline 前置定死。事后重排只能移动已写好的块,不能改内容分配,预期增益小;若嫌 outline 排序差,改 _OUTLINE_SYSTEM 加顺序约束(如"必须以引言开头、结论收尾、按逻辑递进排列")是零成本路径。
- **补节**:是现状的真实缺口——_SECTION_FOCUS 明令各节不写 intro/conclusion,若 outline 没规划"结论"节,报告就没有全局收束;跨节衔接是硬拼,无过渡。其余已知弱点:跨节重复只靠 prompt 约束(各节看不到彼此正文)、outline 一锤定音无补救。
- **但要先过证据关**:"光分节+拼凑效果不好"目前是推断——40 题 3 配置实验实测 RACE **无损**(e2e-3config-40q.md:99-108)。RACE 对衔接/连贯的敏感度可能有限,担忧不能排除,但按主张证据纪律,应先定向验证(抽 10 篇分节报告人工查衔接/重复/缺结论,或加连贯性单项评审)证实缺口存在,再上机制。
- **若做,定案形态(用户已拍板两点:审阅喂全量正文;整体流程认可)**:各节生成后加一次**审阅调用**——输入 = 问题 + 全部节的全文(prefill 快,全量喂),输出 = JSON{已有节的最终顺序, 补节清单:[{标题, 插入位置}]}。**不再跑第二次 outline**:审阅调用本身兼任补节的规划者。随后每个补节各发一次生成调用,补节之间并发(gather,与现有各节生成同套路;常见只补引言/结论 1-2 个,并发与否差别不大);补节调用同样喂全文(引言/结论是全局性内容,只看标题写不出收束),system 约束"只写本节、不重复已有内容"。最后代码按最终顺序确定性拼接。成本形状:审阅 prefill=全报告、decode=短 JSON;每个补节 prefill=全报告、decode=一节;关键路径从 2 跳变 4 跳(outline→节→审阅→补节)。更省的替代仍是只改 outline prompt(必含节+顺序约束,零额外调用);全文重写式后处理不做(把分节收益吃回去)。
- **与本实验的关系**:本实验 B 臂用现状机制(有无损先例背书,口径与 c3 可比);"排序+补节"若立项,单独做消融(现状分节 vs 新机制,RACE+连贯性口径),不混进本实验。

## 评委模型统一口径（2026-07-25 定，用户指示）

**所有 benchmark 打分一律用 `gemini-2.5-flash`**（`judge.env` 的 `JUDGE_MODEL`，`.env` 兜底值同步）。已用 **3.5-Flash 打过的旧结果不重打、不改动**，作为历史口径保留。

换模型的原因（三条实测）：
1. 短暂用过的 `gemini-3-flash-preview` 是**思考模型**——`AUTORATER_MAX_TOKENS=1500` 被内部推理吃光，DRQA 的 JSON 在 Explanation 中间截断，报错率 68–70%（135/200、140/200）；
2. Gemini 每日配额**按模型**计（`generate_requests_per_model_per_day`，10k/天）：并发跑六个打分任务当天把 3-flash 打满，换模型即得全新额度；
3. 2.5-flash 实测非思考模型：438 字符输出仅耗 89 completion tokens、1 秒返回，JSON 可直接解析。

文件口径分层（互不覆盖）：
- `summary_oldjudge_3.5flash.json` = 旧尺（3.5-Flash），不动；
- `*_BAD_truncated_1500tok.jsonl` / `summary_BAD_429subset.json` = 作废件（3-flash 截断 / 429 子集偏差）；
- `summary.json` + `evaluation_results_*_gemini-2.5-flash.json` = 当前 2.5-Flash 口径。

配套修的三个真 bug（否则换模型也会重蹈覆辙）：`AUTORATER_MAX_TOKENS` 1500→8000；gym 打分器改增量落盘 + 连续失败 15 次早停（原来只在全部完成后返回，中途挂掉全丢）；`run_drgym.score()` 逐指标落盘（原来从不写文件，内建续跑机制形同虚设）。
