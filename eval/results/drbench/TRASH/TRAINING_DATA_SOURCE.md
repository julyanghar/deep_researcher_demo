# ⚠️ 本目录不是可弃数据:phase2 EAGLE-3 训练数据的主要来源

"TRASH"命名是历史遗留(作为 DRBench 评测批次它们确实作废),但
`train/Eagle3/convert_harvest_to_eagle3.py` 第 14 行写死 `RUNS=['eagletraj_local_r1','TRASH/*']`
从这里收 harvest,且脚本内 `assert not _missing` ——**删了本目录,训练数据重建直接断言退出**。

实测谱系(按 `phase2-12k-3ep/data/train_main.jsonl` 的 id 前缀统计,2026-08-16):

- 训练集 7,681 条中 **3,805 条(49.5%)** 来自本目录 30 个 run;
- heldout 46 条中 **43 条**来自本目录;
- 其余主力:drgym_local232_r1(2,937 条,在 eval/benchmarks/results/)、eagletraj_local_r1(939 条)。

各 run 贡献条数(降序):

```
690 c1_van40         535 census_A         487 census_Ceager    220 c2_spec_nonpar
158 c1_vanilla_nonpar 121 e2e5_Ceager     108 e2e5_C           105 e2e5_Deager
102 tax_E_e2e        101 e2e5_D            96 e2e5_A            96 tax_D_e2e
 94 e2e5_CeagerK8pad  91 e2e5_G            88 l3v2_quote        88 tax_C_e2e
 87 tax_A_e2e         83 tax_B_e2e         79 l0v2_baseline     76 l0v2_suffix
 43 p4rep             42 p4offN            37 p3off             36 p3rep
 33 p4off             32 p4repT            27 p4offT            24 l3smoke
 20 p4repN             6 sanity1
```

备份:HF dataset `julyanghar/Efficient-DRAgent-data` 的 `tars/drbench_trash.tar.zst`。
完整谱系与其他复现指针见仓库根 `PROVENANCE.md`。
