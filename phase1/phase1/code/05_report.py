"""Build a compact, offline HTML stage report from the saved numerical results."""
import base64
import html
import json
import pandas as pd
from settings import ROOT, RESULTS, FIGURES

audit=json.loads((RESULTS/"data_audit.json").read_text())
network=json.loads((RESULTS/"network_summary.json").read_text())
sim=json.loads((RESULTS/"similarity_summary.json").read_text())
checks=json.loads((RESULTS/"verification.json").read_text())
metrics=pd.read_csv(RESULTS/"network_artist_metrics.csv")
sub=pd.read_csv(RESULTS/"subnetwork_metrics.csv")
genre=pd.read_csv(RESULTS/"r2_genre_results.csv")
sens=pd.read_csv(RESULTS/"similarity_sensitivity.csv")
nsens=pd.read_csv(RESULTS/"network_sensitivity.csv")
coverage=pd.read_csv(RESULTS/"genre_coverage.csv")
pca=pd.read_csv(RESULTS/"pca_variance.csv")

def table(frame):
    return '<div class="table-wrap">'+frame.to_html(index=False,border=0,float_format=lambda x:f"{x:.4f}",na_rep="—")+'</div>'

def figure(name,caption):
    encoded=base64.b64encode((FIGURES/f"{name}.png").read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'

def equation(body):
    return '<div class="equation">'+body+'</div>'

input_table=pd.DataFrame([{"文件":k+".csv","行数":audit[k]["rows"],"列数":audit[k]["columns"],
                           "空值单元格":audit[k]["missing_cells"],"完全重复行":audit[k]["duplicate_rows"]}
                          for k in ["influence_data","full_music_data","data_by_artist","data_by_year"]])
top=metrics.head(8)[["artist_name","genre","out_degree","pr_rank","degree_rank"]].rename(columns={
    "artist_name":"音乐人","genre":"流派","out_degree":"直接追随者","pr_rank":"反向 PR 名次","degree_rank":"出度名次"})
local=sub.head(7)[["artist_name","local_out_degree","local_reverse_pr","local_betweenness","global_rank_within_genre","local_pr_rank"]].rename(columns={
    "artist_name":"音乐人","local_out_degree":"内部出度","local_reverse_pr":"局部 PR","local_betweenness":"局部介数","global_rank_within_genre":"全网 PR 在 Vocal 中名次","local_pr_rank":"局部名次"})
gt=genre[["genre","n","within_similarity","between_similarity","difference"]].rename(columns={
    "genre":"流派","n":"可比音乐人数","within_similarity":"同流派","between_similarity":"跨流派","difference":"差值"})
chosen=["Main Euclidean","Cosine",f"PCA {sim['pcs_for90pct']} PCs (90% variance)","PCA 2 PCs","Exclude loudness",
        "Exclude liveness and speechiness","At least 3 songs","At least 10 songs","Fractional collaborative credits",
        "Artist medians","Solo-only (common eligible artists)","Main on solo-eligible artists","Keep ambiguous IDs (diagnostic)"]
labels=["主模型 Euclidean","Cosine（尺度不同）","PCA 保留 7 维","PCA 只保留 2 维","排除 loudness", "排除 liveness 与 speechiness",
        "至少 3 首","至少 10 首","合作作品分摊权重","艺术家中位数","只用独立署名作品","相同人群的主模型","保留争议 ID（诊断）"]
st=sens.set_index("variant").loc[chosen].reset_index()
st["variant"]=labels
st=st.rename(columns={"variant":"替代口径","artists":"可比音乐人数","genres":"流派数","macro_difference":"总体差值","positive_genres":"差值为正流派数"})
cov=coverage[coverage.genre.isin(["Country","Latin","Vocal","Jazz","Blues"])][["genre","artists_5songs","artists_3_decades_5songs"]].rename(columns={
    "genre":"流派","artists_5songs":"整体至少 5 首","artists_3_decades_5songs":"至少 3 个年代且每年代 ≥5 首"})

body=f'''
<header><p class="kicker">MAT3300 Mathematical Modeling · 2026年11月16日</p>
<h1>Problem C 第一阶段建模与结果</h1>
<p>2021 ICM Problem D — The Influence of Music<br>数据基础、R1 影响网络、R2 音乐相似性</p>
<button onclick="window.print()">打印 / 保存为 PDF</button></header>
<div class="lead"><p><strong>本阶段的主要发现：</strong>同流派音乐人在可观测特征上总体更相似，这个结果在年代分层、流派等权和主要替代口径下保留；但不同流派的差异强度不同。反向 PageRank 与出度给出不同的核心人物，不能把它们当成同一种影响力。PCA 有解释价值，但二维表示丢失超过一半方差，不替代十维主模型。</p>
<p>结果来自实际运行。阶段范围止于 R1、R2；对后续要求只检查数据支持，不拟合后续模型。图中使用英文标签，便于与数据字段对应。</p></div>
<nav><b>阅读目录</b><a href="#formulation">1 数学问题与假设</a><a href="#data">2 数据处理与 EDA</a><a href="#network">3 R1 影响网络</a><a href="#similarity">4 R2 音乐相似性</a><a href="#pca">5 PCA 的实际作用</a><a href="#validation">6 验证与敏感性</a><a href="#next">7 局限与下一阶段</a><a href="#sources">来源与复现</a></nav>

<section id="formulation"><h2>1 数学问题与最少假设</h2>
<p>官方题目第 2 页的 R1 要求建立有向影响网络、定义影响指标，并分析子网络；R2 要求定义音乐相似性并比较流派内外。课程评分强调正确完整地回答要求、合理验证和可复现性，本阶段据此保留少量有明确含义的模型。</p>
<p>令 <i>V</i> 为无身份冲突的音乐人 ID 集合，<i>E</i> 为记录中的影响关系。<i>g</i><sub>i</sub> 表示主流派，<i>c</i><sub>i</sub> 表示职业开始年代，均来自 <code>influence_data.csv</code>。对歌曲记录 <i>s</i>，<i>u</i><sub>s</sub> 是十维风格向量，<i>y</i><sub>s</sub> 是发行年份，<i>I</i><sub>s</sub> 是署名 ID 列表，来自 <code>full_music_data.csv</code>。</p>
{equation('G=(V,E)，　A<sub>ij</sub>=1 ⇔ (i,j)∈E ⇔ 数据记录 i 影响了 j。')}
{equation('S<sub>i</sub> = {s: i∈I<sub>s</sub> 且特征有效}，　n<sub>i</sub>=|S<sub>i</sub>|，　x<sub>i</sub>=(1/n<sub>i</sub>) ∑<sub>s∈Sᵢ</sub> u<sub>s</sub>。')}
<p>音乐人是 R2 的观测单位，歌曲数不是独立音乐人数。音乐人—年代向量以同一公式在有效时间记录内聚合，供后续复用。本阶段不设定影响边的形成时间。</p>
<table><thead><tr><th>假设或建模约定</th><th>为什么需要及数据依据</th><th>不成立时的影响与处理</th></tr></thead><tbody>
<tr><td>记录边可表示“被报告的影响”</td><td>原始表只有二元关系，没有可靠强度；用无权静态图描述它</td><td>漏记关系会改变排名；结果只针对该网络，不假定完整音乐史</td></tr>
<tr><td>样本歌曲的均值可概括可观测风格</td><td>十个数值特征可以共同表示作品轮廓；主分析要求至少 5 首有效记录</td><td>多风格音乐人被平均化；用中位数、歌曲门槛和合作口径检查</td></tr>
<tr><td>主流派标签适合进行粗分组</td><td>使用官方给定标签，不自行按风格重新训练标签</td><td>跨流派作品与宽泛标签会弱化差异；Unknown 不用于流派检验，小组单独标明</td></tr>
<tr><td>置换检验的条件零假设</td><td>给定出道年代与各流派人数，若标签与风格无关联，则年代内标签可交换</td><td>检验的是这一条件下的分组关联；不证明因果，也不保证未观测因素已被控制</td></tr>
</tbody></table>
<p>“各标准化特征等权”“流派等权”是透明的分析选择，不冒充自然规律。前者避免原始单位支配距离，后者避免 Pop/Rock 的人数支配研究问题；替代口径检查它们的实际后果。</p></section>

<section id="data"><h2>2 数据基础与影响模型选择的发现</h2>
{table(input_table)}
<p><strong>没有空值不等于没有数据问题。</strong>原始网络包含 5,603 个 ID；三个 ID（303506、893383、957340）对应冲突名称。本阶段不自动纠错，排除这些节点及 81 条相关边，保留原始文件。相同名称但不同 ID 不自动合并；Oasis 等映射疑点保存在审计表中。</p>
<table><thead><tr><th>问题</th><th>实际发现</th><th>本阶段处理规则</th></tr></thead><tbody>
<tr><td>完全重复与版本</td><td>189 条全行重复</td><td>去掉完全重复；同名但字段不同的版本保留，不声称已识别独立作品</td></tr>
<tr><td>合作歌曲</td><td>原始 4,410 条多人记录，展开 103,487 条署名</td><td>主口径给每位署名者一条记录；额外比较 1/m 分摊及仅独立署名作品</td></tr>
<tr><td>特征异常</td><td>26 条 tempo=0；15 条正响度；29 条短于 30 秒、83 条长于 20 分钟</td><td>tempo=0 的记录不进入十维表示；其余不凭阈值删去，时长先取对数，中位数作为稳健口径</td></tr>
<tr><td>时间冲突</td><td>4 条日期年份冲突；原始 2,756 条署名早于出道年代</td><td>不用于 Artist–Decade 表；静态风格表保留特征有效记录，并保存时间标记</td></tr>
<tr><td>关系时间顺序</td><td>733 条关系的影响者出道年代更晚</td><td>不据此删边；这不等于关系错误，也不能将出道年代当边形成日期</td></tr>
<tr><td>汇总表口径</td><td>官方 count 合计 198,733；仅 241 个 ID 与当前明细署名数相同；3,006 个 ID 的 11 个连续均值全部匹配至 10<sup>−5</sup></td><td>主表示从当前明细重算；汇总表用于审计，不拼接补历史特征</td></tr>
<tr><td>年份覆盖</td><td>明细缺 1922、1923 年，而年度汇总表有；两者均值不能完整对上</td><td>不拿年度汇总补音乐人年份；早期稀疏年份保留覆盖说明</td></tr>
</tbody></table>
<p>最终得到 <strong>{network['nodes']:,} 个网络节点、{network['edges']:,} 条边、{audit['prepared']['artists_features']:,} 个音乐人特征向量、{audit['prepared']['artist_decade_rows']:,} 条音乐人—年代记录</strong>。特征有效且有至少 5 首作品、流派已知的标准化参考样本有 {sim['reference_artists']:,} 人。原始字段与全部清洗标记保存在 <code>data/artist_song_records.csv.gz</code>，可追溯到原文件行号。</p>
{figure('02_network_coverage','图 1　网络的流派与出道年代覆盖。Pop/Rock 占据大部分节点，因此后续采用流派等权；年代也需控制。')}
{figure('04_feature_distributions','图 2　音乐人层面十个特征的分布。instrumentalness、speechiness、liveness 等并非对称正态；不假设距离服从正态分布。')}
{figure('05_correlation_and_counts','图 3　相关性与歌曲数量。低作品量音乐人的均值更不稳定；强相关特征提醒我们检查重复计权。')}
<p>最强的相关结构之一是 energy–loudness（<strong>{sim['feature_correlation_energy_loudness']:.3f}</strong>）及 energy–acousticness（<strong>{sim['feature_correlation_energy_acousticness']:.3f}</strong>）。所以主模型保留可解释的原始维度，同时检查排除 loudness；PCA 用于描述这种共同变化，而不是预设一定更好。大量低作品量音乐人促使我们用 5 首作为初始门槛，并比较 3 首和 10 首。</p>
{figure('06_genre_feature_profiles','图 4　各流派平均特征轮廓。这里只描述分布差异，不把音乐人主流派当作每首歌的独立标签，也不在本阶段完成 R3。')}
</section>

<section id="network"><h2>3 Model 1　有向音乐影响网络</h2>
<p><strong>主指标使用反向 PageRank，但将其明确称为“递归结构地位”；出度作为必须并列的直接覆盖指标。</strong>前者回答“是否被结构地位高的追随者认可”，后者回答“直接影响了多少被记录的音乐人”。不把二者合成一个任意总分。Betweenness 只用于完整 Vocal 子网络的桥梁分析，跨流派输出作为补充描述。</p>
{equation('k<sub>i</sub><sup>out</sup>=∑<sub>j</sub>A<sub>ij</sub>，　k<sub>i</sub><sup>in</sup>=∑<sub>j</sub>A<sub>ji</sub>。')}
{equation('r<sub>i</sub>=(1−α)/N + α [∑<sub>j:Aᵢⱼ=1</sub> r<sub>j</sub>/k<sub>j</sub><sup>in</sup> + (1/N)∑<sub>j:kⱼⁱⁿ=0</sub>r<sub>j</sub>]，　α=0.85。')}
<p>第二个求和项把反向图中无出边节点的分数均匀分配，避免分数丢失。所有节点的得分总和为 1。原图仍显示“影响者 → 追随者”，只在计算 PageRank 时反向；普通 PageRank 默认沿入边获得分数。[方法 1]</p>
{figure('01_network_structure','图 5　直接关系集中度与两种指标的差异。出度尾部很长；左图包含零度节点，右图并不意味着两种指标可以互换。')}
<p>网络有 {network['weak_components']} 个弱连通分量和 {network['isolates']} 个孤立节点（身份排除后仍保留的节点）。在两端流派都已知的 {network['known_genre_edges']:,} 条边中，{network['cross_genre_edges']:,} 条为跨流派，占 <strong>{network['cross_genre_fraction']:.2%}</strong>。按出度排序，前 {network['top1pct_artist_count']} 人约占全部节点 1%，覆盖 <strong>{network['top1pct_edge_share']:.2%}</strong> 的边。</p>
{table(top)}
<p>The Beatles 直接追随者最多（615），反向 PR 排第 7；Cab Calloway 只有 27 个直接追随者，却在反向 PR 排第 1。两者没有矛盾：递归指标会把后代节点的结构分数向前分摊。因此 Cab Calloway 的排名不应被解释为“比 The Beatles 更受欢迎”或“真实影响更强”。</p>
<p><strong>年龄偏差很明显：</strong>出道年代与反向 PR 的 Spearman 相关为 {network['pr_start_decade_spearman']:.3f}，早期节点更容易积累谱系信用。记录覆盖、上游缺失和流派规模都可能参与形成这种差异。保存的年代内百分位可辅助阅读，但本阶段不声称已消除年龄偏差。</p>
<h3>一个由实际统计选出的 Vocal 案例</h3>
<p>选择规则是：流派规模在 50–200 人、至少 50 条内部边，再按跨流派输出边数排序。Vocal 的 877 条跨流派输出略高于 Blues 的 871 条，因此选 Vocal。它是说明结构差异的案例，不代表所有流派，也不是唯一合理选择。</p>
{figure('03_subnetwork','图 6　Vocal 的可读局部视图。全部指标在 162 人、537 条边的完整 Vocal 子图上计算；这里只画局部 PR 前 10 名与介数前 5 名的并集，共 14 人、22 条边。圆形排布无统计含义。')}
{table(local)}
{equation('BC<sub>H</sub>(v) = [1/((N<sub>H</sub>−1)(N<sub>H</sub>−2))] ∑<sub>s≠v≠t, s≠t</sub> σ<sub>st</sub>(v)/σ<sub>st</sub>。')}
<p>这里按有向最短路径计数；不可达配对贡献 0。它衡量结构上的桥梁位置，不证明音乐真的沿最短路径传播。Billie Holiday 的局部 PR 最高但介数为 0；Frank Sinatra 的局部 PR 排第 5，介数却最高（0.02143）。前者像受到后代反复认可的源头，后者更常位于可达路径的中间。</p>
<p>全网与局部 PR 在 Vocal 内的秩相关为 <strong>{network['sub_global_local_spearman']:.3f}</strong>。Ella Fitzgerald 从“按全网 PR 比较 Vocal”的第 6 名上升为局部第 2 名，说明切掉外部流派后，衡量的是 Vocal 内部认可，不能把局部和全网分数直接当同一量比较。以上构成 R1 的回答：影响范围、递归地位与桥梁角色应区分，并由具体子网络解释。</p>
</section>

<section id="similarity"><h2>4 Model 2　音乐特征相似性与 R2 的回答</h2>
<p>十个主特征为 danceability、energy、valence、tempo、loudness、acousticness、instrumentalness、liveness、speechiness、log_duration。<code>log_duration=ln(duration_ms/1000)</code> 先在歌曲层面计算，再聚合到音乐人，因此对应对数时长均值，不是平均时长再取对数。ID、year、popularity、key、mode 与 explicit 不进入这个连续距离。</p>
{equation('z<sub>ik</sub>=(x<sub>ik</sub>−μ<sub>k</sub>)/s<sub>k</sub>，　d<sub>ij</sub>=√[(1/10)∑<sub>k=1</sub><sup>10</sup>(z<sub>ik</sub>−z<sub>jk</sub>)²]，　S<sub>ij</sub>=1/(1+d<sub>ij</sub>)。')}
<p>μ、s 来自 {sim['reference_artists']:,} 个参考音乐人，每位音乐人等权，标准差使用总体分母。单位被统一后，距离就是十个特征差异的均方根；相似度在 (0,1] 内，1 表示相同特征向量。这不衡量旋律或歌词的完整相似性。参考尺度已保存并复用于 Artist–Decade 表；未来若做真正的历史预测，需要重新用当时可用数据拟合尺度。</p>
<h3>比较对象与权重先定义清楚</h3>
<p>先选择至少 5 首有效歌曲、已知流派的音乐人；用于主推断的流派须在这一步至少有 20 人，得到 {sim['descriptive_artists']:,} 人、14 个流派。再保留流派×出道年代至少有 5 人、且该年代可与其他流派比较的单元格，最终有 <strong>{sim['controlled_artists']:,} 人、14 个流派</strong>。限制后个别流派总人数可少于 20，例如 Comedy/Spoken 为 12 人。这是共同支持范围缩小，不把被排除者视为零相似。</p>
<p>对音乐人 i，W<sub>i</sub> 为与同流派、同出道年代的其他人的平均相似性；B<sub>i</sub> 先在每个其他流派内求同年代平均，再对这些流派等权。最后先在每个流派内对音乐人求均值，再对流派等权：</p>
{equation('Δ<sub>i</sub>=W<sub>i</sub>−B<sub>i</sub>，　T=(1/|𝒢|)∑<sub>g∈𝒢</sub>[(1/n<sub>g</sub>)∑<sub>i:gᵢ=g</sub>Δ<sub>i</sub>]。')}
<p>这样不会让 Pop/Rock 的巨大配对数支配结论，也不会将不同年代的音乐人不加区分地混在一起。它仍只控制出道年代，不保证作品发行年代分布或其他背景完全相同。</p>
{figure('08_r2_comparison','图 7　左图为每流派等量抽样的同年代配对分布，仅用于展示；右图和下面数值使用全部符合条件的音乐人精确计算。')}
<p><strong>R2 的明确回答：在主分析可比样本中，同流派音乐人平均更相似。</strong>流派等权的同流派相似度为 <strong>{sim['macro_within']:.4f}</strong>，跨流派为 <strong>{sim['macro_between']:.4f}</strong>，差值 <strong>{sim['macro_delta']:.4f}</strong>。14 个流派的点估计均为正，但强度差别明显；这不等于对每个流派都完成了独立显著性检验。</p>
{table(gt)}
<p>Reggae、Country 的差异较大；International 只有约 0.0061，Electronic 约 0.0195。主标签宽泛、风格重叠或样本覆盖都可能解释较小差异，本阶段不替它们作因果归因。</p>
{figure('09_similarity_heatmap','图 8　音乐人配对相似性的流派均值，排除自配对。此图是跨全部年代的描述，使用音乐人之间的距离，而非两个流派中心点之间的距离；正式结论采用上面的年代分层比较。')}
<p>对全部 19 个已知流派另外保存探索性结果；Avant-Garde 只有 7 名达到作品门槛的音乐人，其未控制年代的差值为负。因此不能把主样本的总体结论扩大为“每一个流派、每一对音乐人都如此”。Classical 等小样本没有被删除，只是不进入主推断。</p>
<h3>统计核验</h3>
<p>固定风格向量和所有配对相似性，在各出道年代内重排音乐人的流派标签，保留每个年代的流派人数，重复 499 次。随机化单位是音乐人标签，不是配对；共享音乐人造成的配对依赖因此不会被错误地当作新增独立样本。</p>
{equation('p = [1 + #{T<sub>b</sub> ≥ T<sub>observed</sub>}]/(499+1) = '+f'{sim["permutation_p"]:.3f}'+ '。')}
<p>观测差值为 {sim['macro_delta']:.4f}，置换零分布均值约 {sim['null_mean']:.5f}，其 2.5%–97.5% 分位区间为 [{sim['null_95_percentiles'][0]:.4f}, {sim['null_95_percentiles'][1]:.4f}]。这一区间是<strong>零假设参考区间，不是观测效应的置信区间</strong>。p=0.002 是 499 次置换下的最小可报告值，不能写成 p=0。结果支持“流派标签与可观测风格有关联”，并不检验音乐人之间的因果影响。</p>
</section>

<section id="pca"><h2>5 PCA 有解释价值，但不替代主距离</h2>
<p>PCA 在同一标准化参考样本拟合，不使用流派标签。设 Z 的协方差特征向量为 V，得分 T=ZV；选取前 q 列就是保留方差最大的 q 个正交方向。本阶段不做 whitening，不把低方差方向重新放大。[方法 2]</p>
{figure('07_pca','图 9　解释方差、前三个 PC 的相关载荷与 PC1–PC2 散点。图中载荷是 corr(特征, PC)，单位长度的特征向量系数另存于 pca_components.csv。散点只为观察重叠，不是分类准确率。')}
<p>PC1 解释 {pca.variance_ratio.iloc[0]:.2%}，PC2 解释 {pca.variance_ratio.iloc[1]:.2%}，合计 <strong>{sim['pc1_pc2_variance']:.2%}</strong>；前三维为 {pca.cumulative.iloc[2]:.2%}，达到至少 90% 要 <strong>{sim['pcs_for90pct']} 维</strong>（累计 {pca.cumulative.iloc[6]:.2%}）。音乐风格有共同变化方向，但并没有被两三个维度充分概括。</p>
<ul><li>PC1 与 energy（0.911）、loudness（0.877）正相关，与 acousticness（−0.848）负相关，可解释为强烈响亮与原声倾向之间的方向。</li><li>PC2 主要关联 danceability（0.806）和 valence（0.755）。</li><li>PC3 主要关联 speechiness（0.809）与 liveness（0.735），说明只看前两维会漏掉语音和现场特征。</li></ul>
<p><strong>数学上要澄清：</strong>保留全部 PC 是正交旋转，Euclidean 距离完全不变；实际检查的最大误差为 {sim['full_pca_distance_error']:.2e}。保留 7 维才改变距离，因为删去了低方差方向，计算仍除以 √10，避免额外改变尺度。PCA 不自动解决“相关特征重复计权”；保留全维甚至根本不改变距离。</p>
<p>7 维版本的总体相似性差值为 0.0675，与主模型 0.0646 相近；在按 ID 顺序选取的前 500 个主描述样本音乐人上，原距离与 7 维距离的 Spearman 相关为 {sim['pca90_distance_rank_correlation']:.4f}。二维版本虽仍产生正差值，但损失超过一半方差，不能因为更能分开某些流派就称为更准确。</p>
<p><strong>决定：</strong>保留十维标准化 Euclidean 为主模型；PCA 保留用于解释和 7 维稳健性检查。Cosine 作为方向相似性的替代，不把不同尺度上的差值大小当作模型优劣。</p>
</section>

<section id="validation"><h2>6 验证与敏感性分析</h2>
<p>网络方向通过 The Beatles → David Bowie 的实际输入关系核对；一个手工星形图中，源节点反向 PR 为 0.5420，三个追随者各为 0.1527，方向符合“追随者信用回流”。全网 PR 固定点的 L1 残差为 {network['pr_residual_l1']:.2e}。相似矩阵检查了对称性与对角线；另按定义独立逐项重算 8 个音乐人的同流派／跨流派差值，最大误差为 {checks['independent_focal_max_error']:.2e}。</p>
{table(nsens.rename(columns={'comparison':'网络比较','spearman':'Spearman','top20_overlap':'前 20 名交集人数'}))}
<p>阻尼参数改变和保留争议 ID 时，PR 核心结果较稳定；但出度与 PR 的前 20 名只重合 4 人，即使全体秩相关达到 0.954，也不能声称头部排序一致。这正是保留两个不同指标、避免人为综合分数的理由。</p>
{table(st)}
<p>各替代口径的总体差值均为正。Cosine 的数值尺度不同，只比较方向。歌曲门槛改变时样本及可比流派也改变，不能把差值变化全部归因于精度；仅独立署名版本与“相同人群的主模型”并列，才更接近隔离合作归属规则的影响。中位数、去掉部分特征或提高门槛后，个别流派变为负值，说明<strong>总体结论稳定不等于每个流派结论都稳定</strong>。逐流派替代结果见 <code>results/genre_sensitivity.csv</code>。</p>
{figure('10_validation','图 10　左为年代内标签置换的零分布，右为有针对性的替代口径。没有将所有选择排列组合，也没有依据这些结果再调参追求更显著。')}
</section>

<section id="next"><h2>7 局限与下一阶段判断</h2>
<p>主结果仍受样本选择影响：网络节点中的低作品量音乐人没有进入主相似性样本；标准化参考样本也不代表世界音乐总体。一个主流派无法完整描述音乐人多变的作品；作品版本、采样差异、名称乱码和未解决的 ID 映射都保留了不确定性。相似性只涉及十个可观测特征；没有歌词、旋律、精确影响时间，不能由 R2 推出 R4 的因果影响。</p>
<table><thead><tr><th>下一阶段事项</th><th>由本阶段真实结果得到的判断</th><th>优先动作</th></tr></thead><tbody>
<tr><td>保留哪些模型</td><td>二元有向图、出度、反向 PR、十维距离、年代分层与流派等权</td><td>沿用统一 ID、特征表和尺度，避免各模块重复清洗</td></tr>
<tr><td>PCA</td><td>有可解释方向，但要 7 维保留至少 90%；二维不足</td><td>保留为解释和稳健性工具，不另加 Factor Analysis</td></tr>
<tr><td>R4 匹配比较</td><td>{sim['r4_supported_edges']:,} 条边两端均达到特征参考样本门槛；其中 {sim['r4_edges_with5controls']:,} 条有至少 5 个同影响者流派、同出道年代的候选对照</td><td>先设计固定追随者的对照；候选对照排除已记录影响者与追随者本人。此处只有支持数，没有做影响检验</td></tr>
<tr><td>R6 单流派时间案例</td><td>Country、Latin 的近期覆盖和持续音乐人数量较好；Vocal 可承接 R1，但后期稀疏</td><td>优先 Country，保留 Latin 作为备选；按年代最低样本量选择区间</td></tr>
<tr><td>R5 变化检测</td><td>已有统一 Artist–Decade 表，但早期年度稀疏、再发行与成员变化可能造成人工突变</td><td>值得在覆盖较好的流派进一步评估；先核验样本构成再决定变化检测，当前没有拟合变点</td></tr>
</tbody></table>
{table(cov)}
<p>这里“至少三个年代”不要求连续，且每个年代至少有五条有效时间记录。Country 在 1950—2010 年各年代都有 21–89 位满足年代内五首门槛的音乐人；Blues 到 2000 年代只剩 1 位、2010 年代为 0。与选题阶段相比，真实覆盖使 Country 比 Blues 更适合连续演化研究。2020 只代表单年，不能当作完整年代。</p>
<p><strong>下一阶段先做 R4 的比较设计与时间资格检查。</strong>优先复用现有表，明确候选对照与时间限制，之后再深入单流派轨迹。初步研究已经回答 R1、R2，但没有提前完成 R3–R7，也没有制作 Final Report、Summary Sheet、ICM 信或 Slides。</p></section>

<section id="sources"><h2>来源与复现</h2>
<p>全部音乐数据仅来自四个指定 CSV。原文件未覆盖，SHA-256、清洗统计与验证结果分别保存在 <code>data_audit.json</code> 和 <code>verification.json</code>。阶段数据、表格与每张图均由脚本重建；HTML 内嵌图像，可离线阅读或打印。</p>
<ul><li><a href="../../Final_Project_Guidelines.pdf">Final Project Guidelines</a>，第 1–3 页：范围、数据限制、评分与可复现要求。</li>
<li><a href="../../Problems/ProblemC_2021ICM-D_The_Influence_of_Music.pdf">2021 ICM Problem D 官方题目</a>，第 2 页 R1、R2；第 3–5 页变量定义。</li>
<li><a href="../../References/C_2021ICM-D/D21018555.pdf">Measure of Influence and Similarity in Dynamic Networks of Musicians</a>，第 6–9 页：跨年代结构地位、加权 Cosine 的启发。本阶段未采用其边形成时间假设、网络训练相似性或相似度加权 PR。</li>
<li><a href="../../References/C_2021ICM-D/D21060286.pdf">Mining the Influence of Music Based on Social Network Analysis and Statistical Methods</a>，第 5–11 页：多种结构指标与低维表示的启发。本阶段未照搬熵权总分、三度传播规则或 Factor Analysis。</li>
<li><a href="../../output/MAT3300_ProblemC_深入理解与项目规划.html">此前项目规划</a>：用于核对已识别问题，统计值均在本阶段重算。</li>
<li>数据：<a href="../../Problems/ProblemC_Music_data/influence_data.csv">influence_data.csv</a>、<a href="../../Problems/ProblemC_Music_data/full_music_data.csv">full_music_data.csv</a>、<a href="../../Problems/ProblemC_Music_data/data_by_artist.csv">data_by_artist.csv</a>、<a href="../../Problems/ProblemC_Music_data/data_by_year.csv">data_by_year.csv</a>。</li>
<li>[方法 1] <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html">NetworkX PageRank 文档</a>：方向、阻尼与 dangling 节点处理。</li>
<li>[方法 2] <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">scikit-learn PCA 文档</a>：主成分、解释方差与 whitening 定义。外部来源仅用于方法，不引入额外音乐数据。</li></ul>
<p>运行方式见 <a href="../README.md">README</a>；主要数值表在 <a href="../results/r2_genre_results.csv">R2 分流派结果</a>、<a href="../results/network_artist_metrics.csv">网络指标</a>和 <a href="../results/similarity_sensitivity.csv">敏感性结果</a>。</p></section>
'''

css='''
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef3f6;color:#20303d;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1120px;margin:28px auto;background:#fff;padding:42px 58px;border-radius:10px}h1{font-size:32px;line-height:1.4;color:#153e58;margin:12px 0}h2{font-size:25px;color:#153e58;margin-top:42px;line-height:1.5}h3{font-size:19px;margin-top:28px;color:#28516a}.kicker{color:#5a7183;font-size:13px;letter-spacing:.05em}p{margin:13px 0}strong{color:#173d56}header{margin-bottom:25px}button{border:0;border-radius:5px;background:#234d68;color:white;padding:10px 18px;font:inherit;font-size:14px;cursor:pointer}.lead{background:#f0f6fa;padding:15px 22px;border-left:4px solid #739ab3}nav{padding:20px 0;border-bottom:1px solid #dce4ea;display:flex;flex-wrap:wrap;gap:10px 24px}nav b{width:100%}a{color:#21668a;text-underline-offset:3px}nav a{text-decoration:none;font-size:14px}section{scroll-margin-top:20px}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6}.table-wrap{overflow-x:auto;margin:20px 0}td,th{border:1px solid #d8e2e9;padding:9px 10px;text-align:left;vertical-align:top}th{background:#eaf1f6;color:#234b64}tr:nth-child(even){background:#f8fafc}figure{margin:24px 0}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#526b7c;line-height:1.6;margin:10px 0 0}.equation{font-family:"Times New Roman","Songti SC",serif;text-align:center;padding:15px 10px;background:#f8fafb;margin:18px 0;font-size:18px;overflow-x:auto}code{font-size:.9em;background:#f2f5f7;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}li{margin:5px 0} @media(max-width:740px){main{margin:0;padding:24px 18px}h1{font-size:27px}h2{font-size:22px}table{min-width:650px}.equation{font-size:16px}} @media print{@page{size:A4;margin:15mm}body{background:#fff;font-size:10pt;line-height:1.55}main{max-width:none;margin:0;padding:0;border-radius:0}h1{font-size:23pt}h2{font-size:16pt;margin-top:24pt}h3{font-size:12pt;margin-top:17pt}h2,h3{break-after:avoid}p{margin:7pt 0;orphans:3;widows:3}button,nav{display:none}.lead{padding:8pt 12pt}table{font-size:8.2pt;line-height:1.4;min-width:0;break-inside:avoid}td,th{padding:5pt;overflow-wrap:anywhere}thead{display:table-header-group}tr{break-inside:avoid}.table-wrap{overflow:visible;margin:12pt 0}figure{break-inside:avoid;margin:15pt 0}figure img{max-height:175mm;object-fit:contain}figcaption{font-size:8.3pt}.equation{break-inside:avoid;font-size:11pt;padding:9pt 4pt}a{color:inherit;text-decoration:none}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
'''
out=ROOT/"report"/"phase1_report.html"
out.parent.mkdir(exist_ok=True)
out.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MAT3300 Problem C 第一阶段建模与结果</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
print(out)
