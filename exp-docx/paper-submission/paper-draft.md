# Once Generated, Many Times Served: Characterizing and Exploiting Content Recycling in Deep Research Agent Inference

<!--
Draft v0.1 (title + abstract + introduction; later sections stubbed).
Conventions:
  - [@key] = citation placeholder (key map at bottom of file).
  - [TBD] = pre-registered experiment, number not yet obtained. Performance-section absolute numbers stay as placeholders per author instruction; speedup RATIOS already measured are written in.
  - All numbers trace to once-generated-paper-narrative.md (Chinese narrative, same directory) and its source docs; the "first characterization" wording follows the 2026-07-11 adversarial novelty check (four qualifiers: object=deep research, layer=LLM-call, metrics=content-level, novelty=the intersection).
  - NUMBER PROVENANCE FLAG: headline count is 800 tasks / 15,296 calls (per author). The detailed per-metric numbers below (90.7%, 8-88%, 22.9% vs 7.7%, 61.2% / 28.8%, r values, 47.5%) are still from the original 100-task / 1,912-call run (online100_v2). Recompute them on the 800-task run before submission; do not mix silently.
-->

## Abstract

Deep research agents—multi-agent systems that search, read, and write long, citation-grounded reports—are bottlenecked by LLM inference. We present the first content-level characterization of deep research inference at the LLM-call layer (800 tasks, 15,296 calls). We found a defining property: **content recycling**. Content is generated once, yet later calls serve it again and again. They re-read it during prefill and re-generate it during decode. However, recycling is pervasive but heterogeneous: some calls copy their inputs almost verbatim; others rewrite and condense (copy rates 8–88%, bimodal). **Once generated, many times served**. Existing techniques cannot collect this waste: (1) cross-request KV reuse perturbs downstream decisions in multi-agent pipelines and degrades report quality; (2) uniform speculative decoding wins on copy-heavy calls but loses on rewrite-heavy ones. We exploit recycling on both sides of inference. On the prefill side, we reuse upstream KV caches and preserve downstream decisions via attention-guided selective recomputation. On the decode side, we route speculative decoding by predicted copy rate: the prediction selects the drafter and sets the draft depth. On mainstream deep research benchmarks, our system cuts end-to-end latency by up to 47.5% and TTFT by [TBD]% versus vLLM. Report quality is statistically unchanged.

## 1 Introduction

Deep research agents turn a question into an investigation. Given a query, a system such as OpenAI Deep Research [@openai-deepresearch] plans sub-questions, searches and reads sources, and writes a long report with citations. Inside, such a system works like a newsroom. A supervisor plans and assigns work. Several researchers each search, read, and write a *summary*. A writer then reads all summaries and produces the final *report*. Each role is a separate LLM call, and one query issues tens of them. The cost is dominated by inference, not retrieval. On the pipeline we study, summary and report generation take about 60% of end-to-end wall clock. Within inference, decode dominates, accounting for about 95% of GPU compute. A serving-layer speedup therefore turns directly into user-visible latency.

We study this workload at the LLM-call layer. We collect a trace of 800 tasks and 15,296 generation calls from a production-shaped pipeline. For every call, we record its input, its output, and the overlap between them. To our knowledge, this is the first *content-level* characterization of deep research inference. Prior agent traces stop at the systems level. They record token counts and timings but discard the text [@tracelab], so copy rates cannot be recovered from them. Prior content measurements sit at other layers: search-query logs [@agentic-search-wild], final-report quality [@reporteval], or single-turn RAG prompts [@byte-exact-dedup]. None reach the LLM-call layer of a multi-agent pipeline.

The trace reveals one property that shapes the rest of the paper: **content recycling**. A researcher copies retrieved passages into its summary. The writer copies those summaries into the report. A span of text is thus authored once but used many times. The serving stack does not exploit this. It treats each call as independent and each output token as new. It pays once to generate the span, again to read it in a downstream prefill, and a third time to regenerate it in a downstream decode. *Once generated, many times served.*

Recycling is pervasive, but it is not uniform, and the non-uniformity has two faces. First, the *amount* of copying varies across calls. Report-writing calls copy their input almost verbatim. Summary calls instead rewrite and condense it. Across the trace, per-task copy rates span 8% to 88%, and the distribution is bimodal. No single accelerator fits both ends, so the system must choose per call. Second, when content does recur, it recurs *word for word*. Among summaries of the same task, exact-match overlap averages 22.9%, while a paraphrase-tolerant measure yields only 7.7%. Recurrence is copying, not restatement. Exact-match machinery is therefore enough, and approximate matching is unnecessary.

Two families of technique already target repeated content, and both fail here. The first is cross-request KV reuse. Prefix caching reuses the KV of an identical prompt prefix [@radixattention]. But a writer reads its summaries in arbitrary order and subset, so a shared prefix rarely exists; the closest agent-aware system concedes that reordered reuse forces full recomputation [@plato]. Position-independent reuse removes that constraint by blending precomputed KV into the new prompt [@cacheblend; @kvcomm; @droidspeak]. The blended KV is approximate, however. In a multi-agent pipeline it shifts what a downstream agent attends to. That shift changes the agent's decisions and degrades report quality. The second family is speculative decoding. Suffix-based speculation drafts the next tokens directly from repeated content [@suffixdecoding], and it runs identically on every call. On copy-heavy report calls it wins: 61.2% of drafts are accepted. On rewrite-heavy summary calls the draft rarely matches (28.8% accepted), and the wasted drafting is pure overhead; under concurrency it becomes a net slowdown. Adaptive variants tune the draft length or the drafter [@specdecpp; @banditspec], but they react to acceptance measured at runtime, not to the copy structure that is knowable before generation.

We exploit recycling on both sides of inference, and we let the copy structure drive both. On the **prefill side**, we reuse the KV that an upstream call already computed while decoding the content. To keep this reuse from perturbing decisions, we recompute selectively. We find the few tokens that downstream attention weights most heavily and recompute only those; the rest are reused unchanged. This preserves the downstream decision while still skipping most of the prefill. On the **decode side**, we predict each call's copy rate from the model's hidden state, and we route speculation on that prediction. The prediction selects the drafter — suffix matching for copy-heavy calls, a specialized draft head for rewrite-heavy calls. It also sets the draft depth: deep where copying is high, shallow or off where it is low. The two sides reinforce each other. Our decode-side fix lowers per-step cost but raises TTFT, and the prefill-side reuse pays that TTFT back.

We evaluate on mainstream deep research benchmarks against vLLM. The system cuts end-to-end latency by up to 47.5% and TTFT by [TBD]%. Report quality is statistically unchanged under blind LLM-judge evaluation. This paper makes four contributions:

1. **Characterization.** The first content-level characterization of deep research inference at the LLM-call layer. We identify content recycling and its two axes of heterogeneity, in amount and in form, and we show that copy rate is the signal an accelerator must route on.
2. **Prefill side.** Decision-preserving KV reuse. We reuse upstream KV across a multi-agent pipeline and recompute only the tokens that downstream attention weights most, which avoids the quality loss of blind blending.
3. **Decode side.** Copy-rate-routed speculative decoding. A predictor drives per-call drafter selection and draft depth, so speculation helps copy-heavy calls without taxing rewrite-heavy ones.
4. **System and evaluation.** An integrated system that couples the two sides, evaluated end-to-end on deep research benchmarks, with a quality check and honest negative results that bound where recycling-based acceleration applies.

<!-- ============================================================
SECTION STUBS (source material: once-generated-paper-narrative.md, same dir)

## 2 Background: The Inference Cost Structure of Deep Research   (narrative §2)
## 3 Characterizing Verbatim Content Recycling                   (narrative §3)
## 4 System Design                                               (narrative §4)
## 5 Evaluation                                                  (narrative §5)
     PERFORMANCE PLACEHOLDERS (per author instruction: ratios already measured
     may be cited in text; absolute tables to be filled at submission):
     - Main table: {vanilla | +spec | +spec+sectioned} × {e2e, report-segment,
       TTFT, quality}  -> e2e −29.6% / −47.5% (ratios locked), absolutes [TBD-final-run]
     - Ablation ledger: suffix / enforce-eager(+28%) / sectioning(−25.4%, concurrency
       not acceptance) / KV-blend / routing  -> per-row nature annotations
     - TTFT tax-hedge figure: vanilla / +eager(+14%) / +eager+blend(−51~67% large-prompt)
     - Concurrency (batch>1) curves [TBD]
     - Specialized head vs 15/20/35% thresholds [TBD]
     - Second framework (GPT Researcher) replication of characterization core table [TBD]
## 6 Related Work    (narrative §6; per-work expanded notes in related-work-overlap-explained.md)
## 7 Limitations and Discussion                                  (narrative §7)
## 8 Conclusion                                                  (narrative §8)

CITATION KEY MAP (placeholder -> work):
@openai-deepresearch   OpenAI Deep Research (system card / product page, 2025)
@anthropic-multiagent  Anthropic, "How we built our multi-agent research system" (blog, 2025-06)
@tracelab              TraceLab, arXiv:2606.30560 (coding-agent trace characterization)
@agentic-workload-char Agentic AI Workload Characteristics, arXiv:2605.26297
@agent-memory          Agent Memory characterization, arXiv:2606.06448
@agentic-search-wild   Agentic Search in the Wild, arXiv:2601.17617 (CTAR)
@reporteval            Understanding DeepResearch via Reports, arXiv:2510.07861
@byte-exact-dedup      Byte-Exact Deduplication in RAG, arXiv:2605.09611
@agentinfer            AgentInfer, arXiv:2512.18337
@grusky2018            Grusky et al., Newsroom (coverage/density), NAACL 2018
@copy-paste-rag        Copy-Paste to Mitigate LLM Hallucinations, arXiv:2510.00508
@suffixdecoding        SuffixDecoding, arXiv:2411.04975
@llma                  LLMA, arXiv:2304.04487
@specdecpp             SpecDec++, arXiv:2405.19715
@banditspec            BanditSpec, arXiv:2505.15141
@plato                 Plato, COLM 2025, arXiv:2402.12280
@relaycaching          RelayCaching, arXiv:2603.13289
@kvcomm                KVCOMM, NeurIPS 2025, arXiv:2510.12872
@cacheblend            CacheBlend, EuroSys 2025, arXiv:2405.16444
@droidspeak            DroidSpeak, NSDI 2026, arXiv:2411.02820
@radixattention        SGLang/RadixAttention, arXiv:2312.07104
@parrot                Parrot, OSDI 2024
@autellix              Autellix, NSDI 2026, arXiv:2502.13965
@eagle3                EAGLE-3
@paypal-eagle          PayPal commerce-agent EAGLE-3 deployment, arXiv:2604.19767
@distillspec           DistillSpec, ICLR 2024
@osd                   Online Speculative Decoding, ICML 2024
@aurora                Aurora, arXiv:2602.06932
============================================================ -->
