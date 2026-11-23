from csv_store import read_csv
"""Build one data-bound report; HTML and PDF use the same page content."""
from pathlib import Path
import base64,csv,html,json,math
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output';DEST=ROOT/'report';DEST.mkdir(exist_ok=True)
def read(section,name):return read_csv(OUT/section/(name+'.csv'))
def num(v,d=3):return '—' if pd.isna(v) else f'{v:.{d}f}'
def pct(v):return '—' if pd.isna(v) else f'{v:.1%}'
def table(headers,rows,widths=None):return {'headers':headers,'rows':[[str(x) for x in r] for r in rows], 'widths':widths}
pages=[]
def page(title,paras=(),fig=None,caption='',tables=(),height=350,kicker='MAT3300 · Phase 3'):
    pages.append({'title':title,'kicker':kicker,'paragraphs':list(paras),'figure':fig,'caption':caption,'tables':list(tables),'figure_max_height':height})

annual=read('r5','annual');five=read('r5','five_year').set_index('start');shift=read('r5','bin_origin_stability')
candidates=read('r5','transition_cases');early=read('r5','early_cases');sensitivity=read('r5','early_novelty_sensitivity')
source_al=read('r5','alignment');source_dir=read('r5','source_matched_direction');alt=read('r5','follower_matched_summary')
leaders=read('r6','cohort_leaders');decomp=read('r6','all_decade_decomposition');r6support=read('r6','leader_follower_support');r6match=read('r6','leader_follower_summary')
calendar=read('r6','leader_follower_calendar_sensitivity');country=read('r7','country_cultural_support').set_index('target_cohort')
variance=read('r7','sonic_variance_decomposition');fixed=variance[variance.population.eq('common_fixed')].set_index('period_start')
genrepairs=read('r7','genre_pair_comparison');network=read('r7','cohort_signatures').set_index('cohort')
summary=json.loads((OUT/'r7/summary.json').read_text())
peak=shift.loc[shift.groupby('origin_offset').J.idxmax()].sort_values('origin_offset')
rho=summary['spearman_audio_vs_stratified_enrichment']

page('音乐演化、代际传承与文化联系',[
'R5–R7 综合研究报告 · 2026年11月23日。分析承接 Phase 2 的正式去重网络、作品年份与固定十维尺度，使用题目提供的数据。每个图表均有同名或对应的 CSV 数据，全文的 HTML 与 PDF 由同一内容源生成。',
f'主要转折在不同五年分箱起点下保持稳定：五种起点的最大位移都位于1960年代前后。艺术家按与本次转折的声音对应量及后续网络覆盖选择，历史距离不再是硬门槛；时序与十特征分量说明候选对应了什么变化，不把相似性当因果传播。',
'Country 的领袖随追随者世代更替，声音变化的来源也随时期改变。新增按时间留出的预测标准中，danceability、acousticness、speechiness提供较稳定的增量信息；energy未通过稳定性规则，全部十特征模型总体误差仍高于网络基准。',
f'文化层面同时存在跨代保留和来源更替。固定四流派权重后，流派内部差异始终占总体声音差异的大部分。调整来源与目标出道世代后，{summary["reliable_directed_genre_pairs"]}个可靠有向流派对的声音相似度与连接富集的秩相关为 {rho:+.3f}，未显示明显单调关系。'],tables=[table(['问题','本轮新增的直接证据','结论强度'],[
['R5 重大跃迁','分箱稳定、十特征转折对应、时间位置、相关后继证据','转折与候选；不认定首次发明或因果传播'],
['R6 流派演化','世代指标、十特征预测重要性、时间匹配与年代分解','网络位置与声音变化分别得到量化'],
['R7 文化影响','来源传统、世代间隔、声音差异分解、分层机会基线','描述传承与交流；不识别外部事件效应']],[.16,.43,.41])])

page('数据口径与三个时间概念',[
'网络使用5,603个艺术家节点和42,770条去重的来源→追随者边，其中42,759条边的两端流派均已知。作品时间表包含5,486位艺术家、22,669条艺术家—年记录及96,755条署名观测；多人合作作品可以贡献多条署名。',
'十个连续特征固定为 danceability、energy、valence、tempo、loudness、acousticness、instrumentalness、liveness、speechiness、log duration。每项用 Phase 2 的参考均值和标准差标准化；它们是回顾性统一单位，不是每个历史时点重新拟合的单位。',
'多年画像先在艺术家内部按实际署名数合并年度均值，再对艺术家等权。一个人作品很多，不会在时期总体均值中获得更多权重。用于双方比较的独立署名作品排除了同一合作录音进入两端的直接重叠。'],tables=[table(['时间/人群','具体含义','分析中如何处理'],[
['作品发行年份','可能包括再发行，非创作或首次录音时间','可比较目录画像，不能推断已故艺术家晚期创作'],
['出道世代','网络提供的十年区间','用于世代覆盖与时间资格；不是关系形成日期'],
['首次有效作品年','本样本中首条可用观测','早期窗口起点；不能等同真实出道年'],
['R5 总体轨迹','所有有效艺术家—年；年度主样本≥20人','分段截至2019；完整五年期比较'],
['R6 声音画像','Phase 2合格Country艺术家，每年代≥5条','主分析1950s–2010s；组成分解共同者≥10'],
['R7 声音分解','合格艺术家，每五年≥5条，已知流派','1960–2019；共同流派每期≥10人']],[.2,.38,.42])])

moves=five.loc[1970,['z_acousticness','z_energy','z_log_duration']]-five.loc[1960,['z_acousticness','z_energy','z_log_duration']]
page('R5 主要变化集中于1960年代',[
f'1960–64→1965–69与1965–69→1970–74的十维RMS位移分别为 {five.at[1965,"J"]:.4f}、{five.at[1970,"J"]:.4f}，为固定1950–2019分箱中最大的两次。1960–64到1970–74，acousticness {moves.iloc[0]:+.3f} SD、energy {moves.iloc[1]:+.3f} SD、log duration {moves.iloc[2]:+.3f} SD。',
'声学特征与energy的长期变化此前已开始，时长的变化更集中。这提示多特征的转折候选区间；两次大位移并不自动排除旧趋势或往返波动，后文另检查方向延续。时长增加不能解释为歌曲内部结构更复杂。'],fig='r5/01_time_trends.png',height=470,caption='图1　浅线为年度均值，深线为五年均值。年度与五年位移的时间步长不同；共同艺术家的净位移会抵消方向相反的个人变化。黄色为描述性候选区间。')

segments=read('r5','segments');best=segments.loc[segments.groupby('spec').working_BIC.idxmin()]
page('R5 转折对分箱起点的稳定性',[
'将五年分箱起点依次平移0、1、2、3、4年，五种口径的最高峰均落在相邻窗口跨越1960年代的比较。它说明主要结果并非固定整五年边界制造的峰值；这些口径共享数据，不能解释为五次独立重复实验。',
'分段线性模型允许每段独立截距与斜率，使用动态规划最小化SSE，最多三个断点，以工作BIC选择段数。相关特征、重复艺术家和时间相关意味着该准则只用于描述性选模，不给出革命日期的显著性。'],fig='r5/05_bin_origin_stability.png',height=230,caption='图2　每条线对应一种分箱起点，仅在该线内部比较相邻五年位移。',tables=[table(['起点平移','最高峰比较','J'],[[int(r.origin_offset),f'{r.before_start}–{r.before_start+4} → {r.after_start}–{r.after_start+4}',num(r.J,4)] for r in peak.itertuples()],[.18,.6,.22]),table(['分段口径','最低段长','所选新段开始'],[[r.spec,int(r.min_periods),str(r.breaks)] for r in best.itertuples()],[.43,.18,.39])])

linked=read('r5','transition_evidence')
links=read('r5','transition_feature_links')
selection_sens=read('r5','transition_selection_sensitivity')
persist=read('r5','transition_persistence').set_index('start')
phase_names={1960:'转折早段',1965:'转折形成期',1970:'转折后段'}
page('R5 哪些艺术家对应这次重大变化',[
'题目先要求识别revolutions（major leaps），再要求识别网络中的revolutionaries（influencers of major change）。前者根据声音变化判断，后者需要把具体变化、时间位置和网络联系连接起来；一般影响力高或历史距离大，都不能单独完成这种连接。',
'先定义总体方向m=μ(1970–74)−μ(1960–64)。对每位艺术家，重算两端均值并排除本人，再取单位方向u=m/||m||。其窗口画像相对于前五年同流派历史中心的偏离为d，转折对应量P=d·u/√10，单位为参考SD。P>0表示沿本次转折方向的偏离；这是画像的回顾性对应量，不是对总体变化的因果贡献。',
'每窗口至少5条作品，历史同流派每人≥3条且至少20人。保留后续覆盖同窗口百分位≥75%、后继数>0及P>0的候选，再按P降序选两例（同分按ID）。历史新颖性不再是硬门槛。75%和每窗两例是此次修订的回顾性展示规则；CSV同时给出50%/90%覆盖门槛与去liveness后的选择。同向性用于筛选，不能再作为独立验证。',
'后继仅计来源窗口后20年内完整落入的出道世代，按全网同资格节点计算覆盖。时间窗角色只表示目录中的位置：1960–64为早段，1965–69为形成期，1970–74为后段；均不自动等于首次创造者。'],fig='r5/06_transition_candidates.png',height=235,caption='转折对应量与后续覆盖分别展示；新颖性仍保留在候选表中，不合成为创新总分。',tables=[table(['窗口/位置','候选人','对应量P','方向余弦','后继数'],[[f'{r.start}–{r.stop} / {phase_names[r.start]}',r.name,num(r.transition_projection),num(r.transition_cosine),int(r.later_followers)] for r in linked.itertuples()],[.29,.27,.15,.15,.14])])

feature_rows=[]
for r in linked.itertuples():
    q=links[(links.artist_id==r.artist_id)&(links.start==r.start)].sort_values('projection_component',ascending=False)
    top=q[q.projection_component>0].head(3)
    feature_rows.append([f'{r.name} ({r.start})','；'.join(f'{v.feature} {v.deviation:+.2f}' for v in top.itertuples()),pct(r.novelty_pct)])
page('R5 用十特征说明人物与转折的联系',[
'把P拆成十项c(f)=d(f)u(f)/√10，严格满足Σc(f)=P。正项表示与总体转折同向，负项表示反向；同时展示原始偏离d，避免把负向声音变化误读成负贡献。全部十特征保留，不能只寻找支持入选的特征。',
'新颖性回答“与历史同行相差多大”，这里回答“差异是否对应本次重大变化”。两者可以不同：独特但方向无关者不一定代表本次转折；历史距离不突出者也不能直接归类为传统延续者。表中列出各例最大的三个正向对应项，不是因果解释。六例中五例的最大正项是log duration，说明该规则较偏重时长；不能称为所有创新形式的全面排名。',
'个体偏离不等于本人前后改变。这张图把个人声音与总体变化联系起来，但不能证明这些人汇聚造成了宏观转折。新颖性平方占比另存于transition_feature_links.csv，与有正负号的转折对应分量严格区分。'],fig='r5/08_transition_feature_links.png',height=330,caption='上图为候选相对前五年同流派中心的偏离；下图为沿总体转折方向的有符号分量，两图色标不同。',tables=[table(['候选及窗口','主要同向特征及偏离SD','历史距离百分位'],feature_rows,[.30,.53,.17])])

retention='、'.join(f'{t}–{t+4}: {persist.at[t,"retention_vs_1970"]:.2f}' for t in [1970,1975,1980])
page('R5 时间位置：早段、形成期与后段',[
'对每个入选案例，以固定1955–59同流派历史均值为基线，沿该案例排除本人后得到的总体方向u，画出1950–79的五年画像。同期同行均值也排除本人；个人每窗≥5条、同行及历史参照≥20人，不足处留空。黄色只表示入选窗口。',
'这一步检查个人特征何时已经出现、同行何时接近或超过；曲线领先只构成时间对应，不能识别边的形成日期、首次发明或因果传播。早段窗口仍与转折区间重叠，若转折前个人观测不足，就不称其为先驱。',
f'总体状态的延续另按R(t)=[μ(t)−μ(1960–64)]·m/||m||²检查：{retention}。R=1为1970–74的投影位置，R=0为原基线；它只测同一方向的位置，不能排除此前趋势或正交方向的变化。分箱稳定与状态延续是两种不同证据。'],fig='r5/09_transition_timing.png',height=440,caption='各案例使用同一个固定历史时期，但参照流派和排除本人后的方向可能不同。缺失线段表示目录不足；不能据此判定尚未出现该风格。')

support_rows=[]
for r in linked.itertuples():
    n=selection_sens[(selection_sens.artist_id==r.artist_id)&(selection_sens.start==r.start)].shape[0]
    support_rows.append([f'{r.name} ({r.start})',f'{n}/6',int(r.later_followers),int(r.matched),num(r.reported_transition_advantage)])
page('R5 网络证据支持到哪一步',[
'主结论是“与转折相关的代表候选”，不是已确认的革命创造者。覆盖率提供后续谱系联系；声音比较作为补充，只检验这些联系是否也对应本次转折方向，不以匹配正负作为革命是否存在的条件。',
'复用原追随者侧匹配：目标流派、出道世代、首次作品年代、作品数档和入度档相同，每人至少5个非追随者对照，目标作品严格晚于来源且在后续20年内。将源和目标投影到总体转折方向u，计算“对照到来源的平均绝对投影距离−追随者到来源的距离”，再对追随者等权平均；至少5个目标才报告。正值表示方向上的距离更近，并不证明变化经由这条边传播。仅The Kinks与Deep Purple达到5人的报告门槛，前者优势很小，其余四例没有足够匹配支持。',
'来源侧一般相似性与完整十维追随者侧匹配属于R4式补充诊断，全部CSV与图保留。Metallica的来源侧平均差−0.00424、追随者侧+0.0517使用不同对照；不能互相抵消，也不用于解释1960年代转折。低新颖性、高覆盖的The Beatles保留作诊断，不自动命名为传统延续者。',
'下表的入选频次来自覆盖门槛50%/75%/90%×十维/去liveness两种方向，共六种设定，不是独立重复。Bo Diddley只在主设定入选，名单不能称为唯一或高度稳定。候选筛选、共用对照和回溯目录均限制解释；负值和不可估计者完整保留。'],tables=[table(['候选及窗口','入选设定','后继数','匹配数','转折方向距离优势SD'],support_rows,[.33,.15,.14,.14,.24]),table(['证据','允许的解释','不能推出'],[
['重大位移与延续','观测样本存在突出转折区间','某个人造成了革命'],
['个人转折对应量','声音差异与宏观方向一致','首次创新或总体变化贡献率'],
['后续网络及匹配','谱系联系及条件声音对应','已识别传播机制或因果效应']],[.25,.40,.35])])

page('R6 Country的十特征总体变化',[
'按艺术家等权，完整呈现十个特征的1950s–2010s轨迹。每个作品年代至少五条有效署名，七个年代分别有21、55、51、54、89、70、50位艺术家。全部面板共享纵轴尺度，不用不同的局部放大程度制造视觉重要性。',
'acousticness总体持续降低，energy在1980s以后连续上升，log duration也有较大长期增加。loudness、tempo有明显中途波动；danceability和speechiness端点净变化较小，但不能因此认为中间没有变化。'],fig='r6/01_genre_trends.png',height=530,caption='图7　十个特征采用同一Phase 2标准化尺度。标题显示1950s→2010s的净变化；阴影标出1990s→2000s。绿色仅标记深入解释的两个案例，不表示其他特征被排除。')

prediction_info=json.loads((OUT/'r6/prediction_summary.json').read_text())
importance=read('r6','prediction_feature_importance')
pred_loss=read('r6','prediction_fold_losses')
pred_scores=read('r6','prediction_artist_scores')
pred_contrib=read('r6','prediction_contributions')
pred_support=read('r6','prediction_cohort_support')
pred_leaders=read('r6','prediction_leader_eligibility')
main_loss=pred_loss[pred_loss.minimum_credits.eq(5)&pred_loss.penalty.eq(10)&pred_loss.model.eq('baseline')]
page('R6 用下一代覆盖率定义预测目标',[
'声音变化幅度不是动态影响力的重要性。目标定义为 y(i,b)=F(i,b)/N(b)：来源i指向下一Country出道世代b的不同追随者数，除以该代全部Country节点数。先按实际覆盖率识别每代动态领袖，再检验哪些先前声音特征能帮助预测该覆盖率；不只在已经入选的五位领袖中拟合。',
f'每行将来源艺术家ID与“目标世代b之前的作品年代[b−10,b−1]”精确连接，要求更早出道、该窗口至少5条作品，并沿用Phase 2资格。得到{prediction_info["eligible_panel_rows"]}条来源—世代记录、{prediction_info["unique_sources"]}位不同来源。1950s只有3条支持记录，故训练从1960s开始。覆盖不足者记为不可用，不把其音频或影响力补零。',
'基准模型控制log(1+先前Country追随者数)、来源与目标的出道世代间隔、log(1+前一年代作品数)和目标世代的线性时间项。先前追随者只计更早来源指向b之前目标世代的边。加入音频的模型p=logistic(α+γ·基准变量+β·先前音频画像)，用二项损失与岭惩罚拟合（所有非截距斜率平方和乘λ/2）；截距不惩罚，其余斜率统一惩罚。',
'主λ=10为预先固定的工作设定，并报告λ=1、100敏感性，未用留出结果选择最佳λ。每次只在更早目标世代拟合，标准化均值与SD也只来自该训练集；逐次留出1980s、1990s、2000s、2010s。记录关系来自回溯网络，图没有形成日期，因此这是按时间留出的回顾性预测验证，不是当年的实时预测。'],tables=[table(['留出目标世代','训练目标世代','训练行数','测试来源数','音频窗口'],[[str(r.target_cohort)+'s',f'1960s–{r.target_cohort-10}s',int(r.train_rows),int(r.test_rows),f'{r.target_cohort-10}–{r.target_cohort-1}'] for r in main_loss.itertuples()],[.17,.25,.17,.17,.24]),table(['连接步骤','可复核字段'],[
['目标Y','followers / all_country_nodes；每代分母相同'],
['特征X','artist_id与前一作品年代匹配；全部十个z_*'],
['背景条件','prior_followers、cohort_gap、n_songs、target_cohort'],
['未拟合检验','训练target_cohort严格小于测试target_cohort'],
['结果输出','每位来源的实际/预测覆盖率、排名及十特征贡献']],[.28,.72])])

importance_rows=[]
for r in importance.itertuples():
    state='增益及条件检验均支持' if r.conditional_supported else '仅单独加入较稳定' if r.stable_candidate else '未过稳定性规则'
    importance_rows.append([r.feature,f'{r.addition_gain*100:+.2f}%',f'{r.conditional_drop_gain*100:+.2f}%',f'{r.addition_positive_folds}/4',f'{int(r.positive_settings)}/5',state])
page('R6 十特征的重要性由预测增益衡量',[
'在每个留出世代，对所有可用来源计算覆盖率均方误差L(b)=mean[(y−p)²]，再对四个世代等权平均，避免人数较多的一代支配比较；MSE惩罚较大的覆盖误差。主重要性I(f)=[L(基准)−L(基准+f)]/L(基准)；正值表示该特征改善了未拟合世代的覆盖率预测。不是按特征波动、系数大小或拟合内相关来排名。',
'辅助量J(f)=[L(全部十特征去掉f)−L(全部十特征)]/L(基准)，用于检查其他特征存在时是否仍有增量信息。单独加入的稳定性规则为I>0、至少3/4个世代改善、五种设置的平均增益均为正（λ=1/10/100和作品门槛3/5/10，主设置只计一次）。条件支持另要求J>0且删除后至少3/4个世代变差。3/4允许一次年代反转、要求多数时期一致，是透明的描述性工作门槛，不是显著性或因果阈值。',
'主排序前三为danceability、acousticness、speechiness；它们也得到条件检验支持。liveness单独加入较稳定，但在全模型中删除它反而改善误差，提示信息重复或模型不稳定。energy的单独增益为负，不能再把它的长期变化排名当作影响力重要性。',
f'基准的世代等权RMSE为{prediction_info["baseline_cohort_equal_rmse"]*100:.2f}个百分点，全部十特征为{prediction_info["full_cohort_equal_rmse"]*100:.2f}个百分点，MSE变化对应增益{prediction_info["full_model_relative_gain"]:.1%}。全部音频一起加入没有总体改善。这里选出的特征仍是同一组验证数据中的探索性候选，未对选择后的子集作独立测试。'],fig='r6/08_predictive_importance.png',height=205,caption='预测重要性　左为相对基准MSE的增益百分比，并非覆盖率百分点。右比较各留出世代误差；负增益必须保留。',tables=[table(['特征','单独增益I','条件增益J','改善世代','正增益设置','判定'],importance_rows,[.19,.14,.14,.13,.14,.26])])

leader_rows=[]
for r in pred_leaders.itertuples():
    q=pred_scores[pred_scores.target_cohort.eq(r.target_cohort)&pred_scores.artist_id.eq(r.artist_id)&pred_scores.model.eq('full')]
    actual=leaders[leaders.target_cohort.eq(r.target_cohort)].iloc[0]
    leader_rows.append([f'{r.name} → {r.target_cohort}s',f'{r.source_window_start}–{r.source_window_stop}',int(r.prior_credits),pct(actual.cohort_coverage),pct(q.predicted_coverage.iloc[0]) if len(q) else '不可估计',int(q.predicted_rank.iloc[0]) if len(q) else '—'])
page('R6 从窗口匹配到动态领袖检验',[
'特征与目标的匹配以艺术家ID和时间窗口为准，不是找听起来相似的人。先用前一年代作品形成十维画像，再用只在更早世代拟合的模型计算下一代覆盖率，最后在同一目标世代的音频合格来源中排名，与实际网络排名比较。真实领袖仍由实际覆盖率决定，预测不能替代观察。',
'下图使用全部十特征模型展示完整映射过程，不将它当成表现最佳的模型。George Strait的2000s实际与预测覆盖率较接近，但Merle Haggard和Kenny Chesney被低估。Vince Gill缺少至少五条1980s作品，无法给出1990s预测；其网络领袖地位仍保留。',
'逐人特征贡献是“该特征的训练标准分×本折系数”，单位为log odds；它解释本次预测的构成，不等于跨人群的特征重要性I或因果效应。热图保留全部十特征，不能只展示贡献为正的部分。'],fig='r6/09_prediction_examples.png',height=280,caption='匹配实例　左为实际网络领袖的覆盖率与预测。右为完整十特征对这些预测log odds的贡献；贡献之和还需要加截距与基准变量项。',tables=[table(['真实领袖与目标世代','来源作品窗','作品n','实际覆盖','十特征预测','预测排名'],leader_rows,[.29,.19,.10,.14,.15,.13])])

example=pred_contrib[pred_contrib.artist_id.eq(944084)&pred_contrib.target_cohort.eq(2000)]
audio_terms=example[example.term.str.startswith('z_')]
base_terms=example[~example.term.str.startswith('z_')]
intercept=float(example.intercept.iloc[0]);base_part=intercept+base_terms.log_odds_contribution.sum();audio_part=audio_terms.log_odds_contribution.sum();eta=base_part+audio_part
example_score=pred_scores[pred_scores.artist_id.eq(944084)&pred_scores.target_cohort.eq(2000)&pred_scores.model.eq('full')].iloc[0]
page('R6 标准如何落到一个实际预测',[
f'以George Strait→2000s为例。来源作品窗为1990–1999，共{int(example_score.credits)}条；此前已记录{int(example_score.prior_followers)}位Country追随者。目标世代有{int(example_score.target_nodes)}位Country节点，其中{int(example_score.followers)}位与其相连，实际覆盖率为{int(example_score.followers)}/{int(example_score.target_nodes)}={example_score.actual_coverage:.2%}。这些目标边不进入该折模型拟合。',
'先读取每项Phase 2标准化画像x，再按训练集（1960s–1990s目标世代）的均值m与SD s变换成u=(x−m)/s；本折系数乘u，得到逐项log odds贡献。训练集统计量、系数和全部原值都存入prediction_contributions.csv，下面列出十项完整计算。',
f'本全模型的截距与四个基准变量贡献之和为{base_part:+.4f}，十项音频贡献之和为{audio_part:+.4f}，总η={eta:+.4f}。预测p=1/(1+exp(−η))={example_score.predicted_coverage:.2%}，在当代{int(main_loss[main_loss.target_cohort.eq(2000)].test_rows.iloc[0])}位音频合格来源中排第{int(example_score.predicted_rank)}。这里的基准变量贡献使用全模型系数，并非单独重拟合的基准模型预测。',
'这个例子只说明标准如何应用，不能用一个预测较准的领袖证明整体有效；四个留出世代的误差均已报告。若希望根据已筛出的三项特征建立最终简化模型，应另做嵌套选择或在新世代独立检验，避免用同一批测试结果同时选择和证明模型。'],tables=[table(['特征','原画像x','训练均值m','训练SD s','本折系数β','贡献βu'],[[r.term[2:],num(r.raw_value),num(r.training_mean),num(r.training_sd),num(r.coefficient),num(r.log_odds_contribution)] for r in audio_terms.itertuples()],[.22,.16,.16,.16,.15,.15])])

page('R6 动态领袖与世代覆盖',[
'对某一Country目标出道世代，分别统计更早Country来源的不同追随者数、占全部入边的份额、占全部Country节点的覆盖率、排名与跨代持续性。三种数量在同一目标世代内给出相同排序，但解释不同，不是三份独立证据。',
'领袖从Hank Williams转为Merle Haggard、Vince Gill、George Strait、Kenny Chesney。George Strait的2000s→2010s入边份额由3.57%升至4.90%，但覆盖率由31.11%降至20.00%；分母缩小可以制造份额上升。'],fig='r6/02_dynamic_influencers.png',height=340,caption='图8　横轴为追随者出道世代，非影响发生年份。来源尚不满足“更早世代”条件时留空，不能填零。',tables=[table(['目标世代','领先来源','后继数','全部入边','节点数','覆盖率'],[[str(r.target_cohort)+'s',r.name,int(r.followers),int(r.all_incoming_edges),int(r.all_country_nodes),pct(r.cohort_coverage)] for r in leaders.itertuples(index=False)],[.14,.3,.12,.15,.13,.16])])

leader_ids=leaders.artist_id.drop_duplicates().tolist()
leader_names=leaders.drop_duplicates('artist_id').set_index('artist_id')['name'].to_dict()
for i in leader_ids:
    page('R6 '+leader_names[i]+'的十特征轨迹',[
    '实线为该来源的目录画像，虚线为排除本人后的同期Country均值。十个特征全部展示，每点至少五条有效署名；缺失年代保留缺口，不跨越连接。每个面板内两条线采用相同尺度，不同面板允许不同范围以看清个体变化。',
    '艺术家可以在某一特征接近流派趋势，在另一特征保持差异。曲线同步不等于引领；作品年代可能来自再发行，尤其不能将Hank Williams的晚期署名目录解释为其晚期创作。'],fig=f'r6/03_leader_{i}.png',height=530,caption='个人轨迹　图顶列出各支持年代的署名数。全部十维数值与同行差值保存在leader_style_profiles.csv。')

main=decomp[decomp.minimum_credits.eq(5)&decomp.feature.isin(['energy','acousticness'])]
page('R6 十特征在不同年代的变化来源',[
'精确恒等式：总体变化 = 持续组均值变化 + 离样比例×(持续组前期均值−离样组均值) + 入样比例×(入样组均值−持续组后期均值)。所有十特征均按此规则分解；这是以持续组为参照的描述性恒等式，不是人员更替的因果效应。',
'十个面板共用纵轴，避免把微小变化视觉放大。黑线是总体变化，三根柱是持续组、离样和入样组成项。energy和acousticness的机制对照只是其中的详细案例，其他特征的完整分解一并保留。'],fig='r6/05_all_decade_decomposition.png',height=530,caption='十特征分解　三项代数和等于总变化。六次比较分别有16、20、22、30、34、29位共同艺术家；门槛敏感性与条件重采样结果见CSV。')

focus=decomp[decomp.minimum_credits.eq(5)&decomp.before.eq(1990)]
page('R6 1990s到2000s的个体证据',[
'这里的energy和acousticness是变化来源的演示案例，不是预测重要性筛选出的前两名。89位前期艺术家与70位后期艺术家中有34位共同入样。energy总体增加0.588 SD，持续组增加0.258 SD，组成项合计增加0.330 SD。acousticness总体下降0.308 SD，持续组均值却略升0.053 SD，组成项合计下降0.361 SD。',
'对持续组艺术家的成对变化进行1000次有放回重采样，固定种子33006。区间仅描述给定可比样本下的均值不稳定性，没有重建整个历史目录抽样机制；不同艺术家也可能共享作品或影响来源。3/5/10条门槛的全部结果写入同一分解表。'],fig='r6/04_change_decomposition.png',height=350,caption='图11　右侧每条细线是一位持续入样艺术家，粗线是同一批人的均值。典型个体的变化不一定等于总体均值变化。',tables=[table(['特征','总体','持续组','离样项','入样项','持续组95%区间'],[[r.feature,num(r.total),num(r.within),num(r.exit_composition),num(r.entry_composition),f'[{num(r.within_boot_low)}, {num(r.within_boot_high)}]'] for r in focus.itertuples()],[.22,.12,.13,.13,.13,.27])])

page('R6 网络传承能否对应声音传承',[
'将五位领袖的首次观测五年solo画像固定，比较其后Country追随者的早期画像与同期非追随者。匹配流派、出道世代、首次观测作品年代、作品数档及入度档，每个目标至少五个对照；目标作品严格晚于来源作品。来源在该早期窗口的作品数及最终覆盖见表。',
'严格匹配后，仅Hank Williams对1990s目标达到每格五人的报告门槛，平均相似度差约−0.0057；其余格的支持不足。去掉作品年代匹配时，Vince Gill的1990s格有6人、平均差+0.0456，但严格匹配后为0人。因此无法把较宽口径的正值写成稳健传播结论。'],fig='r6/06_leader_follower_bridge.png',height=270,caption='图12　空白表示不可报告，不等于零影响。右图显示网络目标→具备时间合格音频→满足匹配的样本流。',tables=[table(['来源','来源作品窗','solo数','网络目标','音频合格','已匹配'],[[r.source_name,f'{r.source_start}–{r.source_stop}',int(r.source_solo_credits),int(r.network_targets),int(r.audio_targets),int(r.matched_targets)] for r in r6support.itertuples()],[.25,.19,.13,.15,.15,.13])])

page('R7 文化交流表现为来源结构变化',[
'R7选择题目的第一条路径：音乐传统如何在时间与情境中传承。影响记录来自艺术家与行业专家，以下称为“被记录的音乐谱系”，不把所有连接解释为艺术家本人主动表达的文化认同。',
f'Country的2010s外流派入边份额为{country.at[2010,"edge_external"]:.1%}，追随者等权后为{country.at[2010,"follower_equal_external"]:.1%}；两个口径回答不同问题。先对每个人算来源比例，再平均，可以避免来源很多的人主导结果；无记录来源的节点只计入覆盖表。',
f'来源出道世代至少早两个年代档的入边份额，从1950s的{country.at[1950,"two_or_more_cohorts_earlier"]:.1%}升至2010s的{country.at[2010,"two_or_more_cohorts_earlier"]:.1%}。这是回溯网络中的世代间隔；较晚目标拥有更大的历史来源池，这一上升本身不能证明文化记忆变强或传播变慢。'],fig='r7/country_cultural_structure.png',height=390,caption='图13　上排比较边等权与追随者等权的来源流派结构。下排展示来源世代间隔与样本覆盖。记录较晚世代指向较早世代的边单独保留。')

page('R7 全网交流与集中度并非单向变化',[
f'全网已知流派入边中，跨流派份额在1960s、1990s、2010s分别为{network.at[1960,"cross_edge_share"]:.1%}、{network.at[1990,"cross_edge_share"]:.1%}、{network.at[2010,"cross_edge_share"]:.1%}。在共同Country、Pop/Rock、R&B目标中固定流派权重后，1990s至2010s出现回升，但不能外推为所有流派的统一趋势。',
'来源多样性使用exp(−Σp ln p)，先在单个追随者上计算，再对拥有3–5个已知来源的追随者平均；单一来源流派为1。Top-10份额及两条不同边共享同一来源的概率衡量来源集中度，两个指标均受来源池与记录覆盖影响。',
'晚期跨流派联系、多样性和头部集中度可以同时回升。数据不支持“交流扩大必然伴随去中心化”的单向叙事，更不能据此识别互联网或特定政治事件的效应。'],fig='r7/network_signatures.png',height=440,caption='图14　每个横坐标是目标出道世代。度基线在该目标世代的入边集合内重算，是连接机会参照，不是历史反事实。')

weights=read('r7','audio_fixed_weights')
page('R7 将声音差异分成流派内与流派间',[
'将原来的平均RMS半径扩展为平均平方距离，使总体差异可精确分解：T = Σg wg E[||z−μg||²/10] + Σg wg ||μg−μ||²/10 = W + B。共同流派每个五年期至少10位合格艺术家；固定权重按整个时期的艺术家—期计数比例计算。',
'共同流派为Country、Latin、Pop/Rock、R&B，权重分别为'+ '、'.join(pct(v) for v in weights.fixed_weight)+'。固定与实际权重比较使用相同流派和资格。',
f'固定权重下，流派间项占总差异的比例为{fixed.between_fraction.min():.1%}–{fixed.between_fraction.max():.1%}，声音差异主要存在于流派内部。2000–04到2015–19，内部项由{fixed.at[2000,"within"]:.3f}降至{fixed.at[2015,"within"]:.3f}，流派间项由{fixed.at[2000,"between"]:.3f}降至{fixed.at[2015,"between"]:.3f}；这是合格目录样本的声音分散度，不能解释为社会总体文化多样性下降。'],fig='r7/sonic_variance_decomposition.png',height=340,caption='图15　单位为参考标准差的平方；不能与R5的RMS位移直接比较。固定流派权重仍不控制流派内艺术家组成、再发行与录音条件。',tables=[table(['时期','艺术家n','内部W','流派间B','总量T','B/T'],[[f'{t}–{t+4}',int(fixed.at[t,'artists']),num(fixed.at[t,'within']),num(fixed.at[t,'between']),num(fixed.at[t,'total']),pct(fixed.at[t,'between_fraction'])] for t in [1960,1980,2000,2015]],[.2,.16,.16,.16,.16,.16])])

pairrows=[]
for g,h in [('Country','Folk'),('Blues','Country')]:
    q=genrepairs[(genrepairs.source==g)&(genrepairs.target==h)].iloc[0]
    pairrows.append([g+' → '+h,int(q.edges),num(q.audio_similarity),num(q.unstratified_ratio),num(q.cohort_stratified_ratio), '是' if q.reliable else '否'])
page('R7 谱系联系与声音接近并不等同',[
'在每个“来源出道世代×目标出道世代”分层内，按来源/目标流派的边端点数量计算期望边数，再跨层求和。此基线保留观测到的有序世代配对结构，可控制部分年代机会差异；仍允许理论上的自环与重复边，不是因果模型或显著性检验。',
f'可靠比较要求流派不同、两边至少20位音频艺术家、实际边数≥10、分层期望≥5，共{summary["reliable_directed_genre_pairs"]}个有向流派对，Spearman相关为{rho:+.3f}。这说明被记录的谱系联系与十维声音距离提供互补信息，不能用一个替代另一个。',
'音频相似度沿用R3全生涯流派比较，不与网络世代分层等价；因此图中是两类汇总量的描述关系。Blues→Country仅6条边，显示作诊断，不进入可靠对的相关计算。'],fig='r7/genre_lineage_vs_sound.png',height=300,caption='图16　点大小随边数增大。纵轴为对数尺度；虚线=度数与世代机会基线。流派对共享节点，不提供独立样本相关显著性。',tables=[table(['流派方向','边数','声音相似度','原度数比','世代分层比','可靠'],pairrows,[.3,.1,.16,.16,.18,.1])])

page('全部特征与流派组成检查',[
'主分析保留全部十个连续特征，避免只展示事后最符合叙事的变量。图中的全部艺术家曲线与四流派曲线人群不同；判断流派权重效应时，只比较同四流派的实际权重与固定权重。',
'年度共同流派从1960–2019每年每流派至少五人中选择。R7的固定权重使用五年期且每人至少五条作品，因此即使流派名单相同，权重也不相同，不应混用。'],fig='r5/02_all_features.png',height=540,caption='图17　所有坐标统一使用Phase 2参考尺度。liveness可能体现录音场景，故另列去除它的候选敏感性；主结果并未事后删除该变量。')

page('研究边界、验证与复现',[
'三个问题的证据链：R5识别时期变化与候选，R6解释Country的世代位置和组成变化，R7将其归纳为传承与交流结构。网络关联、声音接近、早于后继者出现的风格，均不能单独证明因果传播。',
'外部机制只作为待检验解释。技术采用、社会运动或政治限制需要事件时间、暴露差异、地区和可比群体；当前四表没有这些变量。不能把valence当社会情绪、acousticness当技术采用率，或发行年份分组的popularity当历史受欢迎程度。',
'复现：在项目根目录执行 python phase3_xyt/code/run_all.py。需要numpy、pandas、scipy、matplotlib；PDF渲染需要reportlab与中文TrueType字体，可用 --pdf-python 指定安装reportlab的解释器。流程仅从Phase 2正式输出读取数据，顺序执行R5→R6与预测检验→R7→校验→报告，所有分析固定随机种子。',
'验证覆盖：十维标准化与ID唯一性；五年汇总和位移重建；R5排除本人后的转折方向、十特征分量、选人规则、时序及方向匹配复算；后续窗口和对照资格；来源份额与分母守恒；Country多年代分解恒等式；声音总方差W+B恒等式；世代分层期望边数守恒；R6预测目标、训练尺度、留出误差、十特征增益与逐人贡献复算。input_hashes.json、validation.json和manifest.json记录输入摘要、检查结果及最终产物哈希。'],tables=[table(['参考材料','本报告采用的思想及处理'],[
['题目 Problem D，第2、4–5页','R5–R7作答范围与变量定义；仅使用提供的数据'],
['D21018555，第15–17页','影响力、风格差异与后续方向；保留其研究问题结构，避免同步即因果'],
['D21060286，第17–20页','先定位时期再选代表人物；不采用历史popularity或Bollinger显著性推断']],[.35,.65])])

# Number figures after assembling every feature and leader page.
import re
figure_number=0
for item in pages:
    if item['figure']:
        figure_number+=1
        item['caption']=f'图{figure_number}　'+re.sub(r'^(?:图\d+|选择依据|个人轨迹|十特征分解)\s*','',item['caption'])
nav_indices={'R5':next(i for i,p in enumerate(pages,1) if p['title'].startswith('R5 ')),
             'R6':next(i for i,p in enumerate(pages,1) if p['title'].startswith('R6 ')),
             'R7':next(i for i,p in enumerate(pages,1) if p['title'].startswith('R7 '))}
# One report content source; both exports below consume these same pages.
(DEST/'report_content.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
css='''body{margin:0;background:#e9eef1;color:#20323d;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}header{background:#123f4c;color:white;padding:42px max(6vw,24px)}header h1{font-size:30px;margin:0}header p{color:#c4dce2}nav{padding:18px max(6vw,24px);background:#dce8eb;display:flex;gap:18px;flex-wrap:wrap}nav a{color:#165c6b}main{max-width:1100px;margin:auto;padding:24px}section{background:white;border-radius:8px;padding:34px;margin-bottom:26px;box-shadow:0 2px 12px #1534400b}h2{color:#174f5e;font-size:25px;margin-top:0}.kicker{font-size:12px;letter-spacing:1.5px;color:#5d7b85}img{display:block;max-width:100%;height:auto;margin:20px auto}figure{margin:18px 0}figcaption{font-size:13px;color:#526873}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th{background:#e5f0f0;color:#174f5e}td,th{padding:9px 10px;text-align:left;border-bottom:1px solid #dce5e9;vertical-align:top}tr:nth-child(even){background:#f6f9fa}footer{padding:20px;text-align:center;font-size:13px;color:#526873}@media print{body{background:white}nav{display:none}section{box-shadow:none;break-before:page}}'''
parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MAT3300 Phase 3 R5–R7 研究报告</title><style>'+css+'</style><header><h1>音乐演化、代际传承与文化联系</h1><p>MAT3300 · R5–R7 · 统一分析与结果报告 · 2026-11-23</p></header><nav><a href="#p1">研究结论</a><a href="#p3">R5 重大跃迁</a><a href="#p9">R6 Country</a><a href="#p20">R7 文化联系</a><a href="#p25">验证与复现</a></nav><main>']
for ix,p in enumerate(pages,1):
    parts.append(f'<section id="p{ix}"><div class="kicker">{html.escape(p["kicker"])} · {ix:02d}</div><h2>{html.escape(p["title"])}</h2>')
    parts.extend('<p>'+html.escape(t)+'</p>' for t in p['paragraphs'])
    if p['figure']:
        data=base64.b64encode((OUT/p['figure']).read_bytes()).decode()
        parts.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(p["caption"])}"><figcaption>{html.escape(p["caption"])}</figcaption></figure>')
    for t in p['tables']:
        parts.append('<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in t['headers'])+'</tr></thead><tbody>')
        parts.extend('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in t['rows'])
        parts.append('</tbody></table>')
    parts.append('</section>')
parts.append('</main><footer>HTML与PDF是同一报告的两种阅读格式；所有数值可追溯至 phase3_xyt/output。</footer></html>')
html_text=''.join(parts)
nav='<nav><a href="#p1">研究结论</a>'+''.join(f'<a href="#p{nav_indices[k]}">{k}</a>' for k in ['R5','R6','R7'])+f'<a href="#p{len(pages)}">验证与复现</a></nav>'
html_text=re.sub(r'<nav>.*?</nav>',nav,html_text)
(DEST/'Phase3_Report.html').write_text(html_text,encoding='utf-8')
print(f'Built {len(pages)} report sections from final CSV outputs.')
