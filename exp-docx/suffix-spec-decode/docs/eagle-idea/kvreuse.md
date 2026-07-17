## ground:kv-reuse-status
decode-to-prefill KV reuse 盘点。①实现完成度：blend_store_generated 已完成（/home/yilin/LMCache/claude-docx/16、04）。方案A=请求结束时借 request_finished/延迟释放协议一次性存 all_token_ids[:-1]，gen_start=prompt_len 把生成段切成纯内容哈希的独立 chunk，取/查侧零改动自然命中；开关为请求级 kv_transfer_params={"lmcache.blend_store_generated":true}（全局 env 已删）；改动量 adapter ~270行+token_database ~58行+cache_engine ~20行（含空段防御）。已知坑：末位 token 无 KV，复用必须喂 generated[:-1] 否则内容哈希对不上静默 miss。②质量结论：exp-test-supervisor（40题，评委 kimi-k2.5）判 KV 复用无害——SW(只supervisor复用)30.3%/C(全复用)28.2% 不低于不复用 B 24.8%，配对效应≈0、CI 全跨0、在噪声底内；唯一崩题 q27 归 writer。评委教训：qwen-flash 自漂移 0.236 不可靠（kimi 0.018，稳13倍），旧"C 低8分"是评委漂移假象。Exp_M 80题：M×ΔV 相对纯 ΔV/ΔK 为 null（差≤0.03 全不显著），但新 run full-prefill 上界塌至31.1（旧35.6），复用伤害仅+0.033不显著（旧 run 同题+11分）=欠功效，测不出≠证伪。KVCOMM 引擎同管线153题 MMLU：CacheBlend 0.732(−2.6分) 优于 KVCOMM 0.647(−11.1分)，两者都非无损。另有 LMCACHE_BLEND_CONTROL 残留坑曾把复用实验静默变全重算，早先"复用有害"结论是 artifact（doc14）。③性能实测：大档(5-12k token) blend prefill −51~67%（writer 0.998s vs native 3.044s），小档(0.7-1.1k) 反而+32~60%净亏，crossover≈4-6k token；decode 占 GPU 计算95% → 端到端仅−8~10%。短 blend 266ms 中 load 134ms 的97.7%是逐层搬运/launch-bound 非计算。KVCOMM 复现中 CacheBlend TTFT 恒定66ms、100% agent 覆盖 vs KVCOMM 61ms 但仅60%覆盖；论文"CacheBlend 崩溃"与3.1×加速均未复现（后者靠 HF-eager 慢基线）。④定位：LMCache-blend=生产服务路径（CPU 分层+分页KV+多请求），慢在管道税非算法；跑 KVCOMM 负载三要点=warmup 预存段0/原子 sep token/复用 output 需 store_generated。⑤欠的实验：验证"supervisor 强制全复用压塌上界"假设（重跑普通 CacheBlend 拉回~35再判 M×ΔV，或接受 null）；blend vs native 质量裁决需 replay 固定搜索结果；load 134ms 的 stage A/B/C 分相实测与批量化 kernel（①-full/②/③）优化均未做。
关键数字:
- blend_store_generated 已实现：adapter ~270行/token_database ~58行/cache_engine ~20行，请求级 opt-in，末位 token 坑=复用须喂 generated[:-1]（LMCache/claude-docx/04、16）
- 40题四臂：SW 30.28%/C 28.18%/B 24.82%，supervisor 复用效应 −0.007 CI[−0.073,0.055]，复用无害（exp-docx/TRASH/Exp_test_supervisor/Exp_test_supervisor_result.md）
- 评委自漂移：qwen-flash 0.236 vs kimi-k2.5 0.018，稳约13倍，打分改用 kimi（exp-docx/experiment_notes.md §11）
- Exp_M 80题：M×ΔV−ΔV 差仅+0.003~+0.029 全不显著=null；新 run 上界 31.1 vs 旧 35.6 塌4.5分、复用伤害+0.033不显著=欠功效（exp-docx/Exp_M/results_expm_80q_final.md）
- 大档 blend prefill −51~67%（writer 0.998s vs 3.044s）、小档 +32~60% 净亏，crossover≈4-6k token（claude-docx/11）
- decode 占 GPU 计算 95%，端到端 blend vs native 仅 −10%(大档)/−4%(小档)（claude-docx/11）
- 短 prompt blend 总 266ms：load 134ms 的 97.7% 是逐层搬运 launch-bound，asyncio 派发仅 3ms(2.3%)（claude-docx/12）
- KVCOMM 同管线 153题：full 0.7582/CacheBlend 0.7320/KVCOMM 0.6471；TTFT 114/66/61ms，CacheBlend 100% agent 覆盖 vs KVCOMM 60%（KVCOMM-lmcache/claude-docs/archived/00-comparison-summary.md）
- 论文 CacheBlend 崩溃未复现：HumanEval 实测 85.7-87.0% vs 论文 21-33%（claude-docs/03-table1-reproduction.md）
- 手动 GPU 常驻 CacheBlend agent5 TTFT 108ms vs LMCache 版 144ms vs 论文 159ms，快1.3-3.6×=服务架构税（archived/00-comparison-summary.md）

