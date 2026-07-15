# PhyD / SAR-JEPA 下游多 Seed 实验台账

更新日期：2026-07-13

本目录保存当前已确认的下游结果、逐 seed 数值、权重路径和可比性说明。数值单位均为百分比，标准差为 population standard deviation（`statistics.pstdev`）。

## 1. 关键结论

- 当前最佳模型是 `phyd_best_ckpt300`。
- 当前正式多 seed 结果均未使用 SFAFM，严格下游为 `USE_SFAFM=0`。
- `phyd_best_ckpt300` 和 `phyd_warmstart_ckpt299` 使用严格官方风格下游，可直接比较。
- 历史 `checkpoint-250`、原始 `checkpoint-299`、旧 SFAFM 结果使用旧 pipeline 或不完整协议，只能用于历史诊断。
- warm-start `checkpoint-299` 在 18 个数据集/协议/shot 组合中有 15 个低于最佳 `checkpoint-300`，不能替代当前最佳模型。

## 2. 模型登记

| ID | Checkpoint | 训练设置 | SFAFM | 下游状态 |
|---|---|---|---|---|
| `sarjepa_official_ckpt200` | `runs/sarjepa_pretrain_2xh100/checkpoint-200.pth` | 官方 SAR-JEPA | 否 | MSTAR 10-shot finetune，5 seeds |
| `phyd_ckpt250_historical` | `runs/sarjepa_official_phyd_2xh100/checkpoint-250.pth` | grad=1.0, LFST=1.0, norm=patch | 否 | 旧 pipeline，261/360 |
| `phyd_original_ckpt299` | `runs/sarjepa_official_phyd_2xh100/checkpoint-299.pth` | 原始训练 epoch 0-299 | 否 | 旧快速测试，不完整 |
| `phyd_best_ckpt300` | `runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth` | 从 ckpt250 续训；grad=1.0, LFST=0.1, norm=image | 否 | 严格官方下游，357/360；当前最佳 |
| `phyd_lfst0p2_ckpt321` | `runs/sarjepa_official_phyd_ft299_bs1024_lfst0p2_2xh200/checkpoint-321.pth` | LFST=0.2 续训分支 | 否 | 仅少量 seed，不是当前最佳 |
| `phyd_warmstart_ckpt220` | `runs/sarjepa_official_phyd_warmstart_bestckpt300_bs1088_lfst0p1_image_20260709_174032/checkpoint-220.pth` | 加载最佳 ckpt300 的 model-only 权重，重启 schedule | 否 | 3 数据集、2 协议、10-shot、3 seeds |
| `phyd_warmstart_ckpt299` | `runs/sarjepa_official_phyd_warmstart_bestckpt300_bs1088_lfst0p1_image_20260709_174032/checkpoint-299.pth` | 同上，epoch 0-299 | 否 | 严格官方下游，360/360 |
| `old_ph_ckpt99_sfafm` | `runs/pretrain_2xh100_rerun_bs256_lr1e-4/checkpoint-99.pth` | 旧 Ph 权重 + 下游 SFAFM | 是 | MSTAR 10-shot seed0，仅历史参考 |

## 3. 共同项目：MSTAR 10-shot Finetune

| 模型 | Accuracy | Macro-F1 | n | Pipeline |
|---|---:|---:|---:|---|
| 旧 Ph checkpoint-99 + SFAFM | 54.80 | 53.60 | 1 | 旧 pipeline |
| 旧 Ph checkpoint-99，无 SFAFM | 43.80 | 39.30 | 1 | 旧 pipeline |
| 官方 SAR-JEPA checkpoint-200 | 58.74 +- 3.05 | 57.50 +- 2.62 | 5 | 官方风格下游 |
| 原始 PhyD checkpoint-250 | 53.56 +- 3.82 | 50.72 +- 4.14 | 20 | 旧 pipeline |
| 原始 PhyD checkpoint-299 | 54.68 +- 5.70 | 52.56 +- 5.86 | 5 | 旧快速测试 |
| **PhyD 最佳 checkpoint-300** | **70.22 +- 3.00** | **67.78 +- 3.00** | 20 | 严格官方下游 |
| LFST=0.2 checkpoint-321 | 51.83 +- 0.75 | 49.43 +- 0.49 | 3 | 旧测试 |
| Warm-start checkpoint-220 | 63.50 +- 0.99 | 61.20 +- 1.06 | 3 | 严格官方下游 |
| Warm-start checkpoint-299 | 63.80 +- 2.39 | 61.23 +- 2.39 | 20 | 严格官方下游 |

## 4. 最佳 checkpoint-300：严格官方下游

| Dataset | Protocol | Shot | Accuracy | Macro-F1 | n |
|---|---|---:|---:|---:|---:|
| MSTAR_SOC | finetune | 10 | 70.22 +- 3.00 | 67.78 +- 3.00 | 20 |
| MSTAR_SOC | finetune | 20 | 83.34 +- 2.33 | 81.91 +- 2.53 | 18 |
| MSTAR_SOC | finetune | 40 | 89.89 +- 2.16 | 89.24 +- 2.28 | 20 |
| MSTAR_SOC | linear | 10 | 63.47 +- 0.83 | 61.06 +- 1.02 | 20 |
| MSTAR_SOC | linear | 20 | 72.25 +- 1.06 | 70.35 +- 1.07 | 20 |
| MSTAR_SOC | linear | 40 | 77.31 +- 1.16 | 75.81 +- 1.18 | 20 |
| New_FUSAR | finetune | 10 | 80.36 +- 1.95 | 67.83 +- 2.41 | 19 |
| New_FUSAR | finetune | 20 | 82.94 +- 1.19 | 71.22 +- 1.28 | 20 |
| New_FUSAR | finetune | 40 | 85.88 +- 0.86 | 75.34 +- 0.92 | 20 |
| New_FUSAR | linear | 10 | 80.44 +- 1.88 | 68.06 +- 2.37 | 20 |
| New_FUSAR | linear | 20 | 83.26 +- 1.02 | 71.68 +- 1.63 | 20 |
| New_FUSAR | linear | 40 | 85.98 +- 0.68 | 75.59 +- 0.85 | 20 |
| SAR_ACD | finetune | 10 | 53.82 +- 1.81 | 53.44 +- 1.90 | 20 |
| SAR_ACD | finetune | 20 | 61.75 +- 2.08 | 61.69 +- 2.05 | 20 |
| SAR_ACD | finetune | 40 | 71.88 +- 1.98 | 71.62 +- 2.02 | 20 |
| SAR_ACD | linear | 10 | 53.51 +- 1.54 | 53.20 +- 1.68 | 20 |
| SAR_ACD | linear | 20 | 59.89 +- 1.27 | 59.84 +- 1.28 | 20 |
| SAR_ACD | linear | 40 | 65.52 +- 1.67 | 65.54 +- 1.63 | 20 |

缺失或无效日志：MSTAR finetune 20-shot seed 9、11；New_FUSAR finetune 10-shot seed 4。它们的日志没有匹配到预期 checkpoint，因此未计入均值。

## 5. Warm-start checkpoint-299：严格官方下游

| Dataset | Protocol | Shot | Accuracy | Macro-F1 | n |
|---|---|---:|---:|---:|---:|
| MSTAR_SOC | finetune | 10 | 63.80 +- 2.39 | 61.23 +- 2.39 | 20 |
| MSTAR_SOC | finetune | 20 | 77.61 +- 1.74 | 75.42 +- 2.03 | 20 |
| MSTAR_SOC | finetune | 40 | 87.87 +- 1.21 | 86.67 +- 1.41 | 20 |
| MSTAR_SOC | linear | 10 | 47.62 +- 2.83 | 43.98 +- 3.28 | 20 |
| MSTAR_SOC | linear | 20 | 56.27 +- 1.74 | 52.47 +- 2.39 | 20 |
| MSTAR_SOC | linear | 40 | 64.72 +- 1.85 | 62.01 +- 2.06 | 20 |
| New_FUSAR | finetune | 10 | 81.11 +- 1.82 | 68.31 +- 2.19 | 20 |
| New_FUSAR | finetune | 20 | 84.38 +- 0.86 | 72.71 +- 1.19 | 20 |
| New_FUSAR | finetune | 40 | 87.34 +- 0.62 | 77.17 +- 0.88 | 20 |
| New_FUSAR | linear | 10 | 70.46 +- 3.36 | 57.58 +- 3.06 | 20 |
| New_FUSAR | linear | 20 | 78.39 +- 1.15 | 65.69 +- 1.71 | 20 |
| New_FUSAR | linear | 40 | 82.34 +- 1.03 | 70.72 +- 1.06 | 20 |
| SAR_ACD | finetune | 10 | 51.60 +- 2.16 | 51.14 +- 2.04 | 20 |
| SAR_ACD | finetune | 20 | 58.91 +- 2.26 | 58.73 +- 2.38 | 20 |
| SAR_ACD | finetune | 40 | 69.49 +- 2.59 | 69.42 +- 2.51 | 20 |
| SAR_ACD | linear | 10 | 44.91 +- 2.61 | 43.02 +- 3.16 | 20 |
| SAR_ACD | linear | 20 | 49.69 +- 2.36 | 48.65 +- 2.72 | 20 |
| SAR_ACD | linear | 40 | 55.92 +- 1.86 | 55.69 +- 1.95 | 20 |

## 6. 历史 checkpoint-250 聚合结果

这些结果使用旧下游 pipeline，不得作为严格 epoch-only 对照。

| Dataset | Protocol | Shot | Accuracy | Macro-F1 | n |
|---|---|---:|---:|---:|---:|
| MSTAR_SOC | finetune | 10 | 53.56 +- 3.82 | 50.72 +- 4.14 | 20 |
| MSTAR_SOC | finetune | 20 | 72.77 +- 2.33 | 70.27 +- 2.36 | 20 |
| MSTAR_SOC | finetune | 40 | 81.56 +- 1.55 | 79.57 +- 1.77 | 20 |
| MSTAR_SOC | linear | 10 | 45.70 +- 2.38 | 44.30 +- 2.60 | 20 |
| MSTAR_SOC | linear | 20 | 55.41 +- 1.99 | 53.66 +- 2.24 | 20 |
| MSTAR_SOC | linear | 40 | 61.59 +- 1.42 | 59.67 +- 1.53 | 20 |
| New_FUSAR | finetune | 10 | 71.03 +- 2.99 | 59.23 +- 2.41 | 20 |
| New_FUSAR | finetune | 20 | 78.47 +- 1.53 | 67.61 +- 1.87 | 20 |
| New_FUSAR | finetune | 40 | 82.95 +- 1.31 | 73.25 +- 1.49 | 20 |
| New_FUSAR | linear | 10 | 71.15 +- 2.14 | 58.43 +- 2.05 | 20 |
| New_FUSAR | linear | 20 | 77.60 | 66.40 | 1 |
| SAR_ACD | finetune | 10 | 49.70 +- 2.88 | 49.02 +- 3.06 | 20 |
| SAR_ACD | finetune | 20 | 58.17 +- 3.20 | 56.96 +- 3.42 | 20 |
| SAR_ACD | finetune | 40 | 72.13 +- 1.38 | 71.75 +- 1.30 | 20 |

## 7. 逐 Seed 文件

- `best_ckpt300_seed_results.csv`：最佳 checkpoint-300 的357条有效 Accuracy/Macro-F1。
- `warmstart_ckpt299_seed_results.csv`：warm-start checkpoint-299 的360条完整 Accuracy/Macro-F1。
- `ckpt250_historical_seed_accuracy.csv`：历史 checkpoint-250 的261条逐 seed Accuracy。原统计输出未打印逐 seed Macro-F1，因此该列为空，不能反推。
- `raw_best_ckpt300_official_stats.txt`：最佳 checkpoint-300 原始统计输出。
- `raw_warmstart_ckpt299_official_stats.txt`：warm-start checkpoint-299 原始统计输出。
- `raw_ckpt250_historical_stats.txt`：历史 checkpoint-250 原始统计输出。
- `supplementary_selected_checkpoint_results.csv`：官方 SAR-JEPA checkpoint-200、warm-start checkpoint-220、LFST=0.2 checkpoint-321 和 SFAFM7 checkpoint-20 的补充逐 seed 结果。
- `sfafm7_checkpoint_sweep.csv`：7 个 SFAFM 模块实验从 epoch 0 到 180 的训练损失和 MSTAR 10-shot seed0 checkpoint 扫描；epoch 20 最好，epoch 40 后发生明显退化。

## 8. 可比性与论文使用

1. 论文主表优先使用 `phyd_best_ckpt300` 的严格官方下游结果。
2. `phyd_warmstart_ckpt299` 可用于说明长时间继续优化导致 representation drift。
3. `checkpoint-250` 与原始 `checkpoint-299` 的旧 pipeline 结果只能作为历史记录，不能直接写成与最佳300的 epoch 消融。
4. 旧 SFAFM 结果只有单 seed，且 pipeline 已不同，不应宣称当前性能来自 SFAFM。
5. 当前最佳 checkpoint 的预训练发生在官方预训练 loader 完全恢复之前；发表前仍需在恢复后的严格官方预训练 pipeline 上复现最终配置。
