# MAT3300 Problem C Phase 1

本阶段完成数据基础及 R1、R2 初步模型。首先打开 **report/phase1_report.html**，可离线阅读并打印为 PDF。

## 运行

本文件夹放在原项目根目录下，与 `Problems/`、`References/` 同级。输入固定为 `../Problems/ProblemC_Music_data/` 的四个官方 CSV；不下载任何音乐数据，不修改原始文件。

```bash
python -m pip install -r requirements.txt
python run_phase1.py
```

本次实际使用 `/opt/anaconda3/bin/python`（Python 3.14）及 requirements.txt 中的版本。已有依赖时无需重新安装。可在任意工作目录执行 `python /完整路径/phase1/run_phase1.py`。随机种子为 3300，置换 499 次；设为单线程 BLAS，避免这些小矩阵反复启动线程。

## 代码与产物

| 脚本 | 工作 | 主要产物 |
|---|---|---|
| code/01_prepare.py | 四表审计与清洗 | artist_nodes、influence_edges、artist_features、artist_decade_features；原始行号与标记 |
| code/02_network.py | R1 网络及案例 | 网络指标、Vocal 完整子网指标、网络验证；图 01–03 |
| code/03_similarity.py | R2、PCA、置换与针对性敏感性 | 固定标准化参数、PCA、流派结果、置换分布；图 04–10 |
| code/04_checks.py | 独立数值核对与覆盖诊断 | verification、全部流派探索表、时间覆盖；不拟合后续模型 |
| code/05_report.py | 用已保存的结果生成中文阶段报告 | report/phase1_report.html |

`settings.py` 只集中路径、特征顺序、绘图设置。分析按顺序执行，没有 class 或模型框架。`run_phase1.py` 是顺序运行入口。

## 目录

- `data/`：四张核心处理表；可追溯的歌曲—音乐人长表（压缩 CSV）；固定尺度下的特征与 PCA 得分。
- `results/`：数据审计、数值结果和验证。表中 ID 是连接标识，不是可用于距离的数值变量。
- `figures/`：10 张有明确问题的图，每张提供 PNG 和 SVG。
- `report/`：内嵌图像的 HTML 阶段报告。原文件链接依赖上述相对目录。

## 关键口径

1. 三个冲突 ID 从主网络及主特征样本排除；不按同名自动合并。保留争议 ID 仅作敏感性诊断。
2. 全行重复歌曲去重，不按歌名去重。主口径每位署名者计一条；分摊权重和独立署名是替代分析。
3. tempo=0 的记录不进入十维特征。duration 先按秒取自然对数，再求音乐人均值。其余在合理定义范围内的极端音乐特征保留。
4. 年份早于出道年代、日期年份不一致的记录不进入 Artist–Decade 表；静态特征保留有效风格值，并记录时间标记。出道年代不是影响边时间。
5. 标准化参考样本：网络内、无身份冲突、已知流派、至少 5 条有效歌曲记录的 3,072 位音乐人。参考均值和标准差见 feature_scaler.csv，所有替代口径复用。
6. R2 主比较：先取至少 20 位参考音乐人的流派，再限制流派×出道年代单元格至少 5 人。流派间、流派内音乐人分别等权，排除自配对。最终 2,933 人、14 个流派。
7. 置换单位为音乐人的流派标签，限制在同一出道年代内，保留流派人数。不把配对当独立观测；p=0.002 是 499 次置换的分辨率下限。
8. PCA 不 whitening。保留全部 PC 的 Euclidean 距离与原始标准化距离相同；截断 PCA 距离仍除以 sqrt(10)。

## 重要结果文件

- `data_audit.json`：输入 SHA-256、原始问题与清洗流转。
- `network_artist_metrics.csv`：全部节点的出入度、反向 PR、跨流派输出、年代内 PR 百分位。
- `subnetwork_metrics.csv`：Vocal 完整子网络指标。图只画 14 个核心节点，不能拿显示图代替指标计算图。
- `r2_artist_contrasts.csv`、`r2_genre_results.csv`：正式的年代内比较。
- `genre_similarity.csv`：未分年代的描述性热力图，排除自配对。
- `pca_components.csv`：单位长度的主成分方向系数；`pca_correlation_loadings.csv`：特征与 PC 的相关系数，两者不要混称。
- `similarity_sensitivity.csv`、`genre_sensitivity.csv`：总体与分流派替代结果。Cosine 差值尺度与 Euclidean 不同。
- `r4_support_only.csv`：仅统计后续匹配可用的候选人数，未进行 R4 影响检验。
- `temporal_coverage.csv`：后续时间研究的样本量，不是音乐演化结果。

所有 CSV 均保留全精度。报告展示值通常四舍五入到四位小数；图、表和数值可从脚本重建。没有制作最终报告、Summary Sheet、ICM Society Letter 或 Slides。
