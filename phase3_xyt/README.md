# Phase 3 R5–R7 正式分析

本目录只保留一套正式代码和输出。旧 notebook、分散的 evidence 文件夹、旧中英文草稿、旧 HTML 与 ZIP 已由这里的统一版本替代。

## 阅读入口

- [研究报告 HTML](report/Phase3_Report.html)：图表内嵌，可直接在浏览器阅读、放大。
- [研究报告 PDF](report/Phase3_Report.pdf)：同一内容的28页排版稿；不是另一个研究版本。
- [验证结果](output/validation.json)：48项具名检查，另逐条复算所有存储的追随者匹配。

报告覆盖数据、定义、图表、结论与限制，是 Phase 3 的完整研究材料，不是已经合并 R1–R4 的课程终稿。

## 目录

```text
phase3_xyt/
  README.md
  requirements.txt
  code/
    common.py          输入、画像、时间限定匹配
    r5.py              转折、新颖性、候选人、两类匹配
    r6.py              Country动态指标、跨年代分解、声音比较
    r6_predict.py      前期十特征预测下一代覆盖率、时间留出和重要性
    r7.py              文化来源、世代间隔、声音方差、机会基线
    validate.py        独立的数值与匹配资格检查
    build_report.py    根据正式CSV生成统一报告内容与HTML
    render_report.py   将同一报告内容排版为PDF
    run_all.py         唯一完整运行入口
  output/
    r5/                R5数据表与PNG图
    r6/                R6数据表与PNG图
    r7/                R7数据表与PNG图
    input_hashes.json  Phase 2输入文件摘要
    validation.json   检查结果
    report_layout.json PDF页面布局检查数据
    manifest.json     正式代码、输出与报告的哈希清单
    run.log           最近一次完整执行日志
  report/
    report_content.json  HTML/PDF共同使用的结构化内容
    Phase3_Report.html
    Phase3_Report.pdf
```

PNG是正式分析的单一图形文件（含描述性诊断图），CSV是可复核的数值输出。没有保留多套图形PDF或单独的补充报告。

## 一次复现全部结果

从项目根目录运行：

```bash
python phase3_xyt/code/run_all.py
```

输入依赖既有 Phase 2 正式产物：`phase2/analysis/output/phase2/shared/`，R1、R3、R4结果，以及 `phase2/code/output/full_music_data_clean.csv`。本阶段不重新拟合 Phase 2 尺度，也不修改原始数据或 Phase 2。

Python依赖见 `requirements.txt`。若分析环境没有 ReportLab，可指定另一个安装了它的解释器：

```bash
python phase3_xyt/code/run_all.py --pdf-python /path/to/python-with-reportlab
```

macOS默认使用 `/System/Library/Fonts/Supplemental/Arial Unicode.ttf`。其他系统请设置 `PHASE3_CJK_FONT` 指向支持中文的TrueType字体。Codex桌面环境自动识别已配置的PDF运行时。

本机验证命令：

```bash
/opt/anaconda3/bin/python phase3_xyt/code/run_all.py
```

每次执行覆盖同名正式产物，不产生时间戳副本。模型代码默认使用非交互Matplotlib后端；条件重采样种子固定为33006。PDF元数据创建时间可能使文件哈希在重跑后改变，数值结果可复现。

## 数据表索引与统计含义

| 主题 | 正式表 | 说明 |
|---|---|---|
| R5时间 | `r5/annual.csv`, `five_year.csv`, `segments.csv`, `bin_origin_stability.csv` | 年度、五年、分段准则及全部五种分箱起点 |
| R5早期样本 | `r5/artist_flow.csv`, `artist_evidence.csv`, `early_cases.csv` | 首次有效观测后的五年新颖性与完整20年后续窗 |
| R5时期候选 | `r5/transition_candidates.csv`, `transition_cases.csv`, `historical_neighbors.csv` | 全部局部窗口候选、转折对应量筛选及每人的3个历史近邻；legacy_selected保留旧筛选标记 |
| R5新颖性敏感性 | `r5/early_novelty_sensitivity.csv` | 去除liveness后的百分位；三近邻距离及百分位 |
| R5来源侧对照 | `r5/alignment.csv`, `alignment_support.csv`, `matched_feature_evidence.csv`, `matched_feature_summary.csv`, `source_matched_direction.csv` | 固定追随者、更换来源；保留原Metallica结果和无法估计者 |
| R5追随者侧对照 | `r5/follower_matched_pairs.csv`, `follower_matched_features.csv`, `follower_matched_support.csv`, `follower_matched_summary.csv` | 固定来源、更换目标；每行保存control_ids，可逐条复算 |
| R6世代指标 | `r6/dynamic_indicators.csv`, `cohort_leaders.csv`, `cohort_support.csv`, `leader_persistence.csv` | 边份额、节点覆盖、排名、持续性；尚不符合时间资格时留空 |
| R6预测面板 | `r6/prediction_panel.csv`, `prediction_cohort_support.csv`, `prediction_leader_eligibility.csv` | 前一年代十维画像、下一代覆盖率、既有覆盖与缺失资格 |
| R6预测标准 | `r6/prediction_feature_importance.csv`, `prediction_feature_fold_gains.csv`, `prediction_importance_sensitivity.csv`, `prediction_summary.json` | 十项增益排序、逐代改善、五种设定与明确筛选规则 |
| R6预测复核 | `r6/prediction_fold_losses.csv`, `prediction_coefficients.csv`, `prediction_artist_scores.csv`, `prediction_contributions.csv`, `prediction_performance.csv`, `prediction_leader_examples.csv` | 每折误差、训练尺度、实际/预测覆盖与逐特征贡献 |
| R6声音轨迹 | `r6/genre_trajectories.csv`, `leader_style_profiles.csv` | 五位领袖的全部十特征；对照排除本人的同期Country均值 |
| R6全部分解 | `r6/all_decade_decomposition.csv`, `all_continuing_changes.csv` | 六次年代比较、十特征、3/5/10条门槛；共同者不足10人不分解 |
| R6声音传承 | `r6/leader_follower_pairs.csv`, `leader_follower_features.csv`, `leader_follower_support.csv`, `leader_follower_summary.csv`, `leader_follower_feature_summary.csv` | 严格作品先后、目标侧匹配；每个来源×目标世代不足5人不在主图报告均值 |
| R6日历匹配敏感性 | `r6/leader_follower_calendar_sensitivity.csv` | 同一研究中保留/去掉目标首次观测作品年代匹配的结果，不是两个版本 |
| R7全网 | `r7/cohort_signatures.csv`, `follower_signatures.csv`, `fixed_genre_weights.csv`, `genre_cohort_support.csv` | 跨流派联系、多样性、集中度与目标构成 |
| R7 Country | `r7/country_source_counts.csv`, `country_source_shares.csv`, `country_follower_equal_sources.csv`, `country_generation_gaps.csv`, `country_cultural_support.csv` | 具体来源传统、按边/按人权重、出道世代间隔与覆盖 |
| R7声音分解 | `r7/sonic_variance_decomposition.csv`, `sonic_genre_components.csv`, `audio_fixed_weights.csv`, `audio_genre_period_support.csv` | 平均平方距离精确分解为流派内部W与流派间B |
| R7系统比较 | `r7/genre_pair_comparison.csv`, `summary.json` | 在来源世代×目标世代内计算期望边数；所有流派对均保留，reliable标注主图资格 |

其余R5表用于图形逐坐标重建，包括 `period_comparisons`、`artist_feature_comparisons`、`historical_novelty`、`recorded_followers`、`candidate_directions`、`coverage` 和 `fixed_weights`。R7保留 `nonlater_source_sensitivity` 与 `r4_cohort_context`，后者只是全生涯匹配结果按目标世代汇总，不是新增事件效应估计。

## 关键定义与边界

- `start`在时期候选中是作品窗口起点；`active_start`/`target_cohort`是出道年代。首次有效作品年份可能晚于真实出道，也可能对应再发行。
- R5 `artist_evidence.csv` 的 `z_*` 为早期画像减历史流派中心的差；年度/时期表的 `z_*` 为固定标准化后的画像。
- R5 `reach` 的机会分母为完整落入后续窗口的全网出道世代节点；不是实际听众数。局部候选在同一作品窗口内计算后续覆盖百分位。
- `source_matched_direction`的方向由原全部署名早期偏离确定；`follower_matched_pairs`的方向由solo来源画像减历史全部署名同流派均值确定。两种统计量的对照对象不同，不能混成一个传播效应。
- 追随者侧对照匹配目标流派、出道世代、首次观测作品年代、solo作品数档和网络入度档。对照不是已记录追随者；未记录的真实影响仍可能存在。目标作品严格晚于来源窗口。
- R6组成分解以持续组为参照，分量之和严格等于总体变化；它不是入样/离样的因果效应。重采样区间只描述持续组的条件波动，不处理整个目录选择机制。
- R7平均平方距离的单位为参考SD²；T=W+B。它不等于原平均RMS半径，也不是社会文化多样性指标。
- 网络只有回溯关系，没有形成日期；出道世代差不是传播速度。所有新颖性、相似性和文化解释均保留描述性边界。

## R5：重大变化与相关艺术家（2026-11-23修订）

报告第3–8页按“重大变化→具体声音方向→艺术家时间位置→后续网络证据”组织。采纳反馈中的十特征与时间阶段建议；一般相似性匹配降为补充诊断，不再以“新颖性不等于影响力”为主线。R6/R7的研究内容保持原样。

- 总体转折仍由十维位移、分箱起点和分段结果定位。新增`transition_persistence.csv`检查1975–84沿转折方向是否回到1960–64水平；这是状态延续，不能排除旧趋势延续。
- 每人从1960–64及1970–74总体端点均值中排除本人，定义单位转折方向u。窗口声音相对前五年同流派中心的偏离d，投影`P=d·u/√10`衡量与本次转折的对应量。P不是个人造成的总体变化。
- 每窗按后续覆盖百分位≥75%、后继数>0、P>0筛选，再取P最高两例。取消历史距离≥75%的硬门槛。覆盖50%/75%/90%及去liveness的六种设定全部写入`transition_selection_sensitivity.csv`。这是回顾性案例选择，不能称为预注册或独立检验。
- 主名单：1960–64 Herbie Hancock / The Kinks；1965–69 Deep Purple / Bo Diddley；1970–74 Miles Davis / Charlie Daniels。旧名单仍可由`transition_candidates.csv`中的`legacy_selected`核查，旧匹配案例亦完整保留。
- `transition_feature_links.csv`保存全部候选×十特征：历史均值、个人画像、偏离、排除本人的总体方向、有符号投影分量及新颖性平方占比。分量之和等于P；两种“贡献”不能混用。五个主案例的最大正投影项是时长，说明筛选偏重此方向，不代表全面创新排名。
- `transition_timing.csv`使用固定1955–59同流派基线和排除本人的同期同行轨迹；不足样本留空，不把出道年代、发行年和首次创新时间等同。主角色为转折早段/形成期/后段，没有自动认定先驱。
- `transition_followers.csv`复用原匹配目标与control_ids，将距离改为总体转折方向上的距离。`transition_evidence.csv`汇总全部主案例；仅The Kinks（24人）与Deep Purple（36人）达到5人报告门槛。负值、缺失和弱结果保留；不作为革命存在与否的判据。
- 正文新图：`06_transition_candidates.png`、`08_transition_feature_links.png`、`09_transition_timing.png`。两类一般匹配的完整表图仍留在output/r5，作为R4式补充诊断。

## R6的新标准：预测下一代网络覆盖率

更新于2026-11-22。报告第10–13页完整呈现“目标定义→时间连接→基准模型→十特征增益→稳定性筛选→领袖实例”。全部十个特征保留，特征重要性由未来世代的预测表现衡量。

1. **目标**：来源i在Country目标出道世代b的不同追随者数F，除以该世代全部Country节点数N，即`y=F/N`。覆盖率能比较规模不同的世代，并不等于实际听众规模。
2. **匹配**：按artist_id连接b之前一个年代`[b−10,b−1]`的音频画像，每窗口至少5条有效署名；来源出道世代必须更早。共340条来源—世代记录、206位来源。未有合格音频者不补零，Vince Gill→1990s仅3条前期作品，保留真实领袖地位但不给预测。
3. **基准**：既有Country追随者数、出道世代间隔、前期作品数、线性世代时间。二项logistic岭模型使预测处于0–1；非截距斜率全部惩罚，主λ=10，敏感性λ=1/100。训练标准化只使用更早世代。
4. **留出**：先用1960s–1970s训练预测1980s，随后扩展训练窗预测1990s、2000s、2010s。每代计算覆盖率MSE，再对四代等权，避免样本较多的世代支配比较。MSE用于惩罚较大的覆盖率误差；这是服务于研究目标的工作准则，不声称是唯一指标。
5. **重要性**：`I_f=[L(基准)−L(基准+f)]/L(基准)`，主排序按I；`J_f=[L(十特征去掉f)−L(十特征)]/L(基准)`检验其他声音特征存在时的信息。每一比较使用相同测试来源；增益百分比不是覆盖率百分点。
6. **筛选**：I>0、至少3/4代改善，且五种设定（3/5/10条门槛、λ=1/10/100，主设定只计一次）平均I均>0；条件支持还要求J>0、删除后至少3/4代变差。3/4是允许一次年代反转、要求多数时期一致的透明工作门槛，不是统计显著性界线。

| 特征 | 单独增益I | 条件增益J | 结论 |
|---|---:|---:|---|
| danceability | +4.06% | +5.36% | 两项稳定性检查支持 |
| acousticness | +2.41% | +7.31% | 两项稳定性检查支持 |
| speechiness | +1.86% | +2.43% | 两项稳定性检查支持 |
| liveness | +1.13% | −0.79% | 仅单独加入较稳定 |
| energy | −0.23% | −0.77% | 无稳定改善；长期变化幅度不能替代预测重要性 |

完整十特征排序见`prediction_feature_importance.csv`。全部十特征模型的平均MSE反而增加8.12%；世代等权RMSE从5.79升至6.02个百分点。不能声称音频整体优于网络基准，也不能将筛出的三项直接当作已独立验证的最优组合。

这是回溯网络的探索性、按时间留出的预测比较。边没有形成日期，前期作品也可能包含再发行；源艺术家可跨期重复，世代之间不独立。不是当年实时预测或因果效应估计。筛选后的最终子集若要宣称泛化更好，需要嵌套选择或额外独立测试。

## 十特征展示与描述性变化案例

R6总体轨迹、五位领袖轨迹及六次年代分解全部展示十个特征。energy与acousticness保留为变化来源的详细案例，**不再作为预测重要性的前两项**。

1950s→2010s均值净变化绝对值中，acousticness为1.701 SD、energy为1.428 SD，主门槛下排名第1、2；3/5/10条门槛下energy为3/2/2，log duration为2/3/4。1990s→2000s两特征的持续组与组成项方向不同，适合说明总体变化的来源。描述性变化与预测重要性回答不同问题。

描述性核查表仍为`feature_selection_metrics.csv`与`feature_selection_sensitivity.csv`，`07_feature_selection.png`只作变化幅度诊断；新增`08_predictive_importance.png`与`09_prediction_examples.png`是正式报告中的预测标准和实例图。各类表是同一分析的不同输出，没有另存研究版本。

## 与R1–R4的整合

保留来源→追随者方向、正式ID、去重记录和固定十维尺度。R4的全生涯匹配平均差与R5/R6限定时间和来源后的结果是不同研究对象，不能直接替代。合并课程终稿时，从本报告选取与论点对应的图表，不应把本28页研究报告整段附加到已有正文。

## CSV表整合

output/r5、r6、r7中的CSV分别无损合并为tables.zip，共三个表包。压缩包内保留原文件名、列、数值和精度；需要手动查看时可按需解压单表。代码通过csv_store.read_csv直接读取包内表，无须解压。run_all.py重算后自动重新打包，验证及报告生成可独立运行。文中CSV文件名均指对应tables.zip内的成员。
