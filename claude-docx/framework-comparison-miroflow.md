# 三个 Deep Research Agent 框架对比：deep_researcher_demo vs. Open Deep Research vs. MiroFlow

> 本文对比三个 deep research agent 框架，各画一条调用链，帮助从 high-level 理解各自的 workflow。
> - **deep_researcher_demo**（我们的）：源码在 `/home/yilin/deep_researcher_demo`
> - **Open Deep Research**（LangChain 官方）：源码在 `/home/yilin/open_deep_research`
> - **MiroFlow**（别人的）：源码在 `/home/yilin/MiroFlow`
>
> 所有调用链、行为、默认值均来自通读源码，链接指向具体文件。

---

## 一、一句话概览：三者是一条谱系

**核心轴 = 控制权在"代码"还是"模型"手里。**

```
代码全控 ◄────────────────────────────────────────────────► 模型全控
   我们的                     Open Deep Research                 MiroFlow
（固定流水线）            （外层 graph 骨架 + 内层 ReAct）        （纯 ReAct）
 代码写死每步            LangGraph 定死骨架，研究阶段自主        主/子 agent 全自主
```

| | deep_researcher_demo | Open Deep Research | MiroFlow |
|---|---|---|---|
| 本质 | **固定流水线**：代码写死每步，LLM 单点填 JSON | **混合**：外层 graph 骨架固定 + 内层 ReAct 自主 | **Agentic ReAct**：LLM 自己决定调什么、何时停 |
| 控制权在谁手里 | 代码 | 外层代码 + 内层模型 | 模型 |
| 工具 | 只有网页搜索 | 搜索 + MCP + think_tool | 多模态 MCP 工具群 |

一句话：**我们的**最死板可控，**MiroFlow**最放飞自主，**ODR** 卡在中间——LangGraph 定死了"澄清→写 brief→研究→报告"的外层骨架，但研究阶段内部的 supervisor 和每个 researcher 都是自主 ReAct 循环。

---

## 二、框架一：deep_researcher_demo（固定流水线）

入口是普通 CLI，核心是一个**写死步骤的 async 编排循环**，模型只在每个节点被喊一次做单一 JSON 任务。

### 调用链

```
python -m deep_researcher_demo  →  cli.py : main → async_main()
 ├─ OpenAICompatibleClient        一个 OpenAI 兼容客户端
 ├─ create_search_provider        duckduckgo/tavily (+ 缓存/相关性包装)
 ├─ 建 3 个角色: Supervisor / Researcher / FinalWriter
 └─ DeepResearchWorkflow.run(question)                      [workflow.py] ★核心
     │
     ├─ (可选) warmup_kv_prefix()   KV 复用实验才用:预热常量前缀
     │
     ├─ 1. supervisor.initialize_question()                 [agents.py]
     │       └─ LLM → JSON: 把大问题拆成若干"独立子问题"
     │
     ├─ 2. ★ for iteration in range(max_iterations):        固定轮数循环(代码控制)
     │      │
     │      ├─ _run_researchers(current_questions)           ← 多个 researcher 并行(gather+信号量)
     │      │    └─ 每个 researcher.research(q):
     │      │         ├─ plan_queries()     LLM → JSON: 子问题 → 几条搜索 query
     │      │         └─ 各 query 并行 _process_sub_query():
     │      │              ├─ search_provider.search([query])     真·联网搜索
     │      │              └─ summarize_results()  LLM → 文本: 把搜索结果压成摘要
     │      │
     │      ├─ supervisor.decide(summaries)                  LLM → JSON:
     │      │      { status: continue|complete, followup_questions, reason }
     │      │
     │      └─ status==complete → break;  否则把 followup 塞进下一轮 pending
     │
     └─ 3. final_writer.write(summaries)                     LLM → Markdown 最终报告
```

### 关键点

- **控制流全在 Python 里**：拆题 → 研究 → 评审 → 写报告，顺序写死。模型从不决定"下一步干嘛"，只在每个格子里填内容。见 [workflow.py `run`](../deep_researcher_demo/workflow.py#L54)。
- **三个角色都是"prompt + 严格 JSON 契约"**，带 JSON 解析失败自动 repair。见 [agents.py](../deep_researcher_demo/agents.py)。
- **两层并行扇出**：多个 researcher 并行 + 每个 researcher 内多条 query 也并行。
- **循环几时停由代码定**：`max_iterations` 轮数上限 + supervisor 判 `complete`。
- **一堆 KV 复用埋点是实验用的，跟 workflow 逻辑正交**（`kv_reuse_separator` / `store_generated_kv` / `warmup` / `interleave`），默认关掉时逐字节等价。参见同目录 [workflow.md](./workflow.md)。

---

## 三、框架二：Open Deep Research（混合：LangGraph 骨架 + 内层 ReAct）

交付形态跟另两个都不一样——它是个 **LangGraph 服务**（`langgraph dev` 起 server + Studio UI 可视化调试），核心是一张**状态图**（节点 + 边 + 状态在其间流转 + reducer 合并并行分支），不是普通函数调用栈。

### 调用链

```
langgraph dev  →  graph "deep_researcher"                    [deep_researcher.py]
                  (StateGraph: 状态在节点间流转 + reducer 合并并行分支)

 START
  ├─ clarify_with_user          LLM(结构化输出): 需求模糊?→ 需要就 END 反问用户  ★人在环
  ├─ write_research_brief        LLM(结构化输出): 把对话历史 → 一段 research_brief，初始化 supervisor
  ├─ research_supervisor  ═══════ 子图 supervisor_subgraph ═══════
  │    START → supervisor  ⇄  supervisor_tools     ★ReAct 循环
  │      supervisor:       LLM.bind_tools([ConductResearch, ResearchComplete, think_tool]) 出招
  │      supervisor_tools: 执行工具:
  │         ├ think_tool       → 记录一段反思，回 supervisor            (显式"思考"工具)
  │         ├ ConductResearch  → 并行 spawn researcher 子图 × N (≤max_concurrent=5) ⤵
  │         │                     收集各自 compressed_research 当 ToolMessage，回 supervisor
  │         └ ResearchComplete / 无工具 / 超轮数(>6) → END (把 notes 汇出)
  │
  │      researcher 子图 (每个并行研究员，本身又是一个 ReAct 子图):
  │         START → researcher ⇄ researcher_tools → compress_research → END
  │           researcher:       LLM.bind_tools(搜索 + MCP + think_tool) 出招
  │           researcher_tools: 并行执行工具; ResearchComplete/超轮数(≥10) → 去压缩
  │           compress_research: LLM 把这个研究员的所有发现压成一段 summary  ★先自压缩再交回
  │
  └─ final_report_generation    LLM: 把所有 notes 汇成最终报告 (超 token 渐进截断重试)
 END
```

### 关键点

- **外层是固定 graph，内层是自主 ReAct**：`clarify → write_brief → supervisor → final_report` 这条主干由 LangGraph 边写死；但 supervisor 和 researcher **各自是 `节点 ⇄ 工具节点` 的 ReAct 循环**，调几次工具、搜什么、何时停，由模型自己定（`max_researcher_iterations=6`、`max_react_tool_calls=10` 只是兜底）。见 [deep_researcher.py](../../open_deep_research/src/open_deep_research/deep_researcher.py)。
- **两级委派**：supervisor 通过 `ConductResearch`（带 schema 的结构化工具）一轮可**并行派多个** researcher（≤`max_concurrent_research_units=5`）；每个 researcher 是独立 ReAct 子图，跑完**先自己压缩成摘要**才交回 supervisor——控制上下文爆炸。思路同 MiroFlow"主→子 agent"，只是用 LangGraph 子图 + 结构化工具表达，而非命名约定。
- **think_tool**：把"反思"做成一个显式工具，逼模型在搜索之间停下来想，而不是闷头搜。ODR 一个有意思的设计。
- **人在环**：`clarify_with_user` 可在开跑前反问用户澄清需求——**另两个框架都没有**。见 [deep_researcher.py:60](../../open_deep_research/src/open_deep_research/deep_researcher.py#L60)。
- **显式 State + reducer**：并行 researcher 各自往 `notes`/`raw_notes` 里写，靠 `override_reducer` / `operator.add` 自动合并。见 [state.py](../../open_deep_research/src/open_deep_research/state.py#L55)。
- **LangGraph 白拿的东西**：断点续跑（checkpointing）、streaming、Studio 可视化调试、状态合并。代价是绑死 LangChain/LangGraph 全家桶。

---

## 四、框架三：MiroFlow（纯 ReAct，主/子 Agent 递归）

入口是 benchmark 跑分脚本，核心是一个 **ReAct 工具循环**：主 Agent 循环里能把子 Agent 当成"工具"来递归调用。

### 调用链

```
main.py  (fire CLI)
 └─ common_benchmark.py : main → entrypoint(cfg)            [Hydra 配置]
     ├─ create_pipeline_components(cfg)                     构建 MCP ToolManager(主+子) + OutputFormatter
     └─ 遍历 benchmark 任务 (asyncio.Semaphore 限并发, pass@k 重试)
         └─ execute_task_pipeline(...)                      [pipeline.py]
             ├─ TaskTracer         建日志追踪器(全程 save)
             ├─ LLMClient × 2      主 Agent / 子 Agent 各一个
             └─ Orchestrator.run_main_agent(task)           [orchestrator.py] ★核心
                 │
                 ├─ 1. process_input + 注入 task_guidance(要求"穷尽所有候选答案")
                 ├─ 2. extract_hints()   (可选) 先让 LLM 预判题目里的坑
                 ├─ 3. 取工具定义 = MCP工具 + expose_sub_agents_as_tools(子Agent伪装成工具)
                 ├─ 4. 生成 system prompt (把工具描述塞进去)
                 │
                 ├─ 5. ★ ReAct while 循环 (max_turns):
                 │      ├─ _handle_llm_call_with_logging → llm_client.create_message() → LLM 出招
                 │      ├─ 解析 LLM 想调哪些工具 (tool_calls)
                 │      ├─ 逐个执行工具调用:
                 │      │    ├─ server 名以 "agent-" 开头 ─→ run_sub_agent()  ⤵ 递归进子 Agent
                 │      │    └─ 否则 ─────────────────────→ tool_manager.execute_tool_call()
                 │      │                                     → MCP server (stdio/SSE 子进程)
                 │      ├─ 工具结果塞回 message_history
                 │      └─ 回到循环顶(LLM 看到结果继续决策，直到它不再调工具)
                 │
                 ├─ 6. _handle_summary_with_context_limit_retry()  生成最终总结(超长会砍历史重试)
                 └─ 7. extract_gaia_final_answer()   再喊一次 LLM 把答案抠成评测要的格式
```

`run_sub_agent()` 和主循环**几乎是同一套代码**：它有自己那套真正干活的工具（搜索/阅读/代码/音频/图像），跑完把一段总结 `return` 给主 Agent——在主 Agent 眼里，这就是一次"工具调用的返回值"。

### 关键点

- **主 Agent 其实不亲自干活**。GAIA 配置里主 Agent 只挂 `tool-reasoning` + 子 Agent；真正搜网/跑代码的是 `agent-worker` 子 Agent。主 Agent 负责思考和派活。见 [agent_gaia-validation_claude37sonnet.yaml](../../MiroFlow/config/agent_gaia-validation_claude37sonnet.yaml)。
- **工具 = MCP server 进程**，stdio/SSE 通信，现连现断。见 [manager.py](../../MiroFlow/src/tool/manager.py)。
- **循环几时停由模型定**：LLM 某轮不再请求工具就结束，`max_turns` 只是兜底。
- **子 Agent 靠命名约定被识别**：`server_name.startswith("agent-")` 就走递归。见 [orchestrator.py:884](../../MiroFlow/src/core/orchestrator.py#L884)。
- **工程配套很重**：Hydra 配置、TaskTracer 全程日志、pass@k、hint 生成、答案抽取、context-limit 砍历史重试、工具找不到自动纠错。

主循环代码：[orchestrator.py `run_main_agent`](../../MiroFlow/src/core/orchestrator.py#L689)。

---

## 五、三方逐维度对照

| 维度 | deep_researcher_demo | Open Deep Research | MiroFlow |
|---|---|---|---|
| **编排技术** | 纯 Python async 循环 | LangGraph 状态机（节点/边/reducer） | 手写 async while 循环 |
| **控制范式** | 全固定流水线（代码控） | 混合：外层 graph 固定 + 内层 ReAct 自主 | 全 ReAct（模型控） |
| **谁决定研究几步** | 代码（max_iterations） | 骨架固定，但 supervisor/researcher 自己决定调几次工具 | 模型 |
| **多 Agent 关系** | 3 平级角色，无递归 | supervisor → 并行 researcher 子图（两级） | 主 agent → 子 agent（伪装成工具，递归） |
| **工具** | 只搜索 | 搜索(Tavily/native/DDG/Exa) + MCP + think_tool | MCP 多模态(搜索/浏览器/代码/音频/图像) |
| **反思机制** | 无（supervisor 的 decide 隐含） | 显式 think_tool | 隐含在 ReAct 推理里 |
| **人在环** | 无 | 有（clarify 可反问用户） | 无 |
| **交付形态** | CLI 脚本 | LangGraph 服务（Studio UI + API + auth） | benchmark 跑分脚本 |
| **状态管理** | 普通局部变量 | 显式 State + reducer（处理并行合并） | message_history 列表 |
| **模型输出约束** | 强制 JSON schema（pydantic） | 结构化输出 + bind_tools（pydantic） | 自由 tool-calling 文本解析 |
| **失败处理** | JSON 一次 repair、抓不到页回退 snippet | 结构化输出重试、token 超限渐进截断 | context-limit 砍历史、工具自动纠错、超时包装 |
| **典型场景** | DeepSearchQA 问答 + KV 复用系统实验 | Deep Research Bench，通用研究报告 | GAIA / BrowseComp / HLE 硬核 agent 跑分 |
| **代码复杂度** | 低（~1000 行，零框架依赖） | 中（核心 ~700 行，重度依赖 LangGraph） | 高（orchestrator ~1100 行，多 provider） |

---

## 六、取舍总结（why）

- **deep_researcher_demo（我们的）** 赌"流程可控"：适合做**系统层实验**（KV 复用要求同样检索、同样步骤才能对照）。零框架依赖、好埋点、好复现，代价是不灵活——遇到需要临场决定"该搜还是该跑代码"的任务就没辙。
- **Open Deep Research** 赌"骨架固定 + 局部自主"：拿 LangGraph 白嫖工程能力（可视化/续跑/streaming/服务化），研究阶段又保留模型的临场判断。**平衡点做得最好**，代价是绑死 LangChain 生态、调试要理解 graph/state/reducer 这套抽象。
- **MiroFlow** 赌"模型足够聪明"：多模态、开放式硬题（GAIA）能打，代价是最不可控、最难复现、最烧 token。

一句话：**我们的优化"能不能量准、复现、做实验"；ODR 优化"好不好用、好不好扩展"；MiroFlow 优化"能不能啃下开放式硬题"。** 三者不是谁取代谁，是三个目标。

### 对 KV 复用实验的额外视角

**ODR 和 MiroFlow 的 ReAct 循环里，上下文是"滚雪球"式增长的**（message_history 一直累加），复用的段边界不稳定、难对齐；**我们的固定流水线每个节点的 prompt 结构是确定的**，才好切段、埋 separator、做对照——这是当初选固定流水线作为实验载体的合理性所在。

---

## 七、谱系图

```
           代码控制  ◄──────────────────────────────────────────►  模型控制

  deep_researcher_demo          Open Deep Research                 MiroFlow
  ┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
  │ Supervisor 拆题   │          │ clarify(可反问)   │          │ 主 Agent(思考+派活)│
  │       ↓          │          │       ↓          │          │       ↓          │
  │ Researcher×N 并行  │          │ write_brief      │          │ while 未收敛:     │
  │ (plan→搜→压缩)    │          │       ↓          │          │  LLM 自己决定调:  │
  │       ↓          │          │ supervisor ⇄tools │          │   ├ 真工具(MCP)   │
  │ Supervisor 评审    │          │  ├think_tool     │          │   └ 子Agent(递归) │
  │ continue/complete │          │  └ConductResearch │          │  结果回填, 继续... │
  │       ↓          │          │     ↓并行         │          │       ↓          │
  │ FinalWriter 报告   │          │  researcher×N ⇄   │          │ 总结 + 答案抽取   │
  │                  │          │   (ReAct→压缩)    │          │                  │
  │                  │          │       ↓          │          │                  │
  │                  │          │ final_report     │          │                  │
  └──────────────────┘          └──────────────────┘          └──────────────────┘
   控制权: 代码                   外层代码 + 内层模型              控制权: 模型
   停止: max_iterations           骨架固定, 局部模型定             停止: 模型说"不调了"
```

---

## 参考源码

**deep_researcher_demo**（`/home/yilin/deep_researcher_demo`）：
- 入口 [cli.py](../deep_researcher_demo/cli.py)
- 主循环 [workflow.py](../deep_researcher_demo/workflow.py)（`DeepResearchWorkflow.run`）
- 三个角色 [agents.py](../deep_researcher_demo/agents.py)（Supervisor / Researcher / FinalWriter）
- 搭配阅读：同目录 [workflow.md](./workflow.md)、[search-call-chain.md](./search-call-chain.md)

**Open Deep Research**（`/home/yilin/open_deep_research`）：
- 核心 graph [deep_researcher.py](../../open_deep_research/src/open_deep_research/deep_researcher.py)（主图 + supervisor 子图 + researcher 子图）
- 状态定义 [state.py](../../open_deep_research/src/open_deep_research/state.py)
- 配置旋钮 [configuration.py](../../open_deep_research/src/open_deep_research/configuration.py)
- graph 入口声明 [langgraph.json](../../open_deep_research/langgraph.json)

**MiroFlow**（`/home/yilin/MiroFlow`）：
- 顶层入口 [main.py](../../MiroFlow/main.py) → [common_benchmark.py](../../MiroFlow/common_benchmark.py)
- 单任务流水线 [pipeline.py](../../MiroFlow/src/core/pipeline.py)
- 核心编排器 [orchestrator.py](../../MiroFlow/src/core/orchestrator.py)（`run_main_agent` / `run_sub_agent`）
- MCP 工具管理 [manager.py](../../MiroFlow/src/tool/manager.py)
- 代表性配置 [agent_gaia-validation_claude37sonnet.yaml](../../MiroFlow/config/agent_gaia-validation_claude37sonnet.yaml)
