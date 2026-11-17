"""R1: directed network, interpretable influence measures and a data-selected subnetwork."""
import json
import textwrap
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from settings import DATA, RESULTS, RAW, SEED, plt, save_figure

nodes = pd.read_csv(DATA / "artist_nodes.csv").set_index("artist_id")
edges = pd.read_csv(DATA / "influence_edges.csv")
G = nx.DiGraph()
G.add_nodes_from(nodes.index)
G.add_edges_from(edges.itertuples(index=False, name=None))  # source=influencer, target=follower
reverse_pr = nx.pagerank(G.reverse(), alpha=0.85, tol=1e-12, max_iter=1000, weight=None)
stats = nodes.copy()
stats["out_degree"] = pd.Series(dict(G.out_degree()))
stats["in_degree"] = pd.Series(dict(G.in_degree()))
stats["reverse_pagerank"] = pd.Series(reverse_pr)
stats["pr_rank"] = stats.reverse_pagerank.rank(ascending=False, method="min").astype(int)
stats["degree_rank"] = stats.out_degree.rank(ascending=False, method="min").astype(int)
annotated = edges.join(nodes[["genre", "active_start"]], on="source").rename(
    columns={"genre": "source_genre", "active_start": "source_decade"})
annotated = annotated.join(nodes[["genre", "active_start"]], on="target").rename(
    columns={"genre": "target_genre", "active_start": "target_decade"})
known = annotated.source_genre.ne("Unknown") & annotated.target_genre.ne("Unknown")
cross = annotated[known & annotated.source_genre.ne(annotated.target_genre)]
stats["cross_genre_followers"] = cross.groupby("source").size().reindex(stats.index, fill_value=0)
stats["known_genre_followers"] = annotated[known].groupby("source").size().reindex(stats.index, fill_value=0)
stats["cross_genre_fraction"] = stats.cross_genre_followers.div(stats.known_genre_followers.replace(0, np.nan))
stats["pr_percentile_in_start_decade"] = stats.groupby("active_start").reverse_pagerank.rank(pct=True)
stats.sort_values(["reverse_pagerank"], ascending=False).to_csv(RESULTS / "network_artist_metrics.csv")
stats.groupby("active_start").agg(n=("artist_name", "size"), median_out=("out_degree", "median"),
    mean_pr=("reverse_pagerank", "mean"), median_pr=("reverse_pagerank", "median")).to_csv(RESULTS / "network_cohort_summary.csv")

# Mathematical checks: degree accounting, a hand-checkable star, PageRank fixed point.
toy = nx.DiGraph([(0, 1), (0, 2), (0, 3)])
toy_pr = nx.pagerank(toy.reverse(), tol=1e-13)
assert toy_pr[0] > toy_pr[1] and np.isclose(toy_pr[1], toy_pr[2])
assert stats.out_degree.sum() == stats.in_degree.sum() == len(edges)
assert G.has_edge(754032, 531986)  # The Beatles -> David Bowie, checked against input.
p = pd.Series(reverse_pr).reindex(stats.index)
dangling = p[stats.in_degree.eq(0)].sum()
incoming = pd.Series(0.0, index=stats.index)
contribution = annotated.target.map(reverse_pr).to_numpy() / annotated.target.map(stats.in_degree).to_numpy()
incoming = pd.Series(contribution).groupby(annotated.source.to_numpy()).sum().reindex(stats.index, fill_value=0)
rhs = (1-0.85)/len(stats) + 0.85*(incoming + dangling/len(stats))
residual = float(np.abs(p-rhs).sum())
assert np.isclose(p.sum(), 1) and residual < 1e-7

# A compact genre with substantial outgoing cross-genre links is a useful case.
# The rule is fixed before looking at the identities of the top artists.
genre_stats = stats.groupby("genre").agg(n=("artist_name", "size"),
    outgoing=("out_degree", "sum"), outgoing_cross=("cross_genre_followers", "sum"))
genre_stats["internal_edges"] = annotated[annotated.source_genre.eq(annotated.target_genre)].groupby("source_genre").size()
genre_stats["internal_edges"] = genre_stats.internal_edges.fillna(0).astype(int)
genre_stats.to_csv(RESULTS / "network_genre_summary.csv")
candidates = genre_stats[(genre_stats.n.between(50, 200)) & (genre_stats.internal_edges.ge(50))]
selected_genre = candidates.sort_values(["outgoing_cross", "n"], ascending=[False, True]).index[0]
sub_ids = stats.index[stats.genre.eq(selected_genre)]
H = G.subgraph(sub_ids).copy()
sub = stats.loc[sub_ids].copy()
sub["local_out_degree"] = pd.Series(dict(H.out_degree()))
sub["local_reverse_pr"] = pd.Series(nx.pagerank(H.reverse(), alpha=.85, tol=1e-12, max_iter=1000))
sub["local_betweenness"] = pd.Series(nx.betweenness_centrality(H, normalized=True, weight=None))
sub["local_pr_rank"] = sub.local_reverse_pr.rank(ascending=False, method="min").astype(int)
sub["global_rank_within_genre"] = sub.reverse_pagerank.rank(ascending=False, method="min").astype(int)
sub.sort_values("local_reverse_pr", ascending=False).to_csv(RESULTS / "subnetwork_metrics.csv")
pd.DataFrame(H.edges(), columns=["source", "target"]).to_csv(RESULTS / "subnetwork_edges.csv", index=False)

# A selected view of the genre graph, while ALL local metrics use the complete H.
view_ids = sorted(set(sub.nlargest(10, "local_reverse_pr").index) |
                  set(sub.nlargest(5, "local_betweenness").index))
view = H.subgraph(view_ids)
pos = nx.circular_layout(view)
fig, ax = plt.subplots(figsize=(11, 8))
sizes = [250 + 1000*sub.loc[i, "local_reverse_pr"]/sub.local_reverse_pr.max() for i in view]
nx.draw_networkx_edges(view, pos, ax=ax, arrows=True, arrowsize=14, alpha=.32,
    node_size=sizes, connectionstyle="arc3,rad=0.08", edge_color="#557184")
node_plot = nx.draw_networkx_nodes(view, pos, ax=ax, node_size=sizes,
    node_color=[sub.loc[i, "local_betweenness"] for i in view], cmap="YlGnBu", edgecolors="#35536b")
labels = {i: textwrap.fill(sub.loc[i, "artist_name"], width=18) for i in view}
label_pos = {i: 1.23 * xy for i, xy in pos.items()}
nx.draw_networkx_labels(view, label_pos, labels=labels, ax=ax, font_size=9,
    bbox={"facecolor": "white", "alpha": .88, "edgecolor": "none", "pad": 1.5})
ax.set_title(f"{selected_genre}: {len(view)} selected artists, {view.number_of_edges()} displayed edges\n"
             f"Full genre subnetwork: {len(H)} artists, {H.number_of_edges()} edges", pad=15)
ax.text(.5, -.04, "Arrow: influencer -> follower. Size: local reverse PageRank. Color: local betweenness.",
        transform=ax.transAxes, ha="center", fontsize=9)
fig.colorbar(node_plot, ax=ax, fraction=.025, pad=.01, label="Directed betweenness in full genre graph")
ax.set_xlim(-1.6,1.6);ax.set_ylim(-1.45,1.45); ax.axis("off")
save_figure(fig, "03_subnetwork")

# EDA figures have distinct questions: concentration, coverage, metric disagreement.
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for col, color in [("in_degree", "#d68c45"), ("out_degree", "#287b8e")]:
    v = np.sort(stats[col].to_numpy()); axes[0].step(v, (len(v)-np.arange(len(v)))/len(v), label=col, color=color)
axes[0].set(xscale="symlog", yscale="log", xlabel="Degree (zero retained)", ylabel="Fraction with degree >= x", title="A. Degree tails")
axes[0].legend()
v = np.sort(stats.out_degree)[::-1]
axes[1].plot(np.arange(1,len(v)+1)/len(v), np.cumsum(v)/v.sum(), color="#287b8e")
axes[1].plot([0,1],[0,1],"--",color="gray",lw=1)
axes[1].set(xlabel="Top fraction of artists", ylabel="Share of outgoing edges", title="B. Recorded influence concentration")
axes[2].scatter(stats.out_degree+1, stats.reverse_pagerank, alpha=.25, s=8, color="#426b89")
axes[2].set(xscale="log", yscale="log", xlabel="Out-degree + 1", ylabel="Reverse PageRank", title="C. Breadth versus recursive credit")
fig.tight_layout();save_figure(fig,"01_network_structure")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
gs = stats.genre.value_counts().sort_values()
axes[0].barh(gs.index, gs.values, color="#4f8195");axes[0].set(xlabel="Artists", title="Genre coverage in the network")
cs = stats.active_start.value_counts().sort_index()
axes[1].bar(cs.index.astype(str), cs.values, color="#b18356");axes[1].tick_params(axis="x",rotation=45)
axes[1].set(xlabel="Career-start decade",ylabel="Artists",title="Cohort coverage (not edge dates)")
fig.tight_layout();save_figure(fig,"02_network_coverage")

top20 = set(stats.nlargest(20,"reverse_pagerank").index)
checks = []
for alpha in [.7, .95]:
    alternative = pd.Series(nx.pagerank(G.reverse(), alpha=alpha,tol=1e-12,max_iter=1000)).reindex(stats.index)
    checks.append({"comparison": f"PR alpha {alpha}", "spearman": spearmanr(p, alternative).statistic,
                   "top20_overlap": len(top20 & set(alternative.nlargest(20).index))})
raw = pd.read_csv(RAW/"influence_data.csv")
rawG = nx.from_pandas_edgelist(raw, "influencer_id", "follower_id", create_using=nx.DiGraph)
rawpr = pd.Series(nx.pagerank(rawG.reverse(),tol=1e-12,max_iter=1000)).reindex(stats.index)
checks.append({"comparison":"Keep ambiguous IDs (common-node comparison)",
    "spearman":spearmanr(p,rawpr).statistic, "top20_overlap":len(top20 & set(rawpr.nlargest(20).index))})
checks.append({"comparison":"Out-degree versus reverse PR", "spearman":spearmanr(stats.out_degree,p).statistic,
    "top20_overlap":len(top20 & set(stats.nlargest(20,"out_degree").index))})
pd.DataFrame(checks).to_csv(RESULTS/"network_sensitivity.csv",index=False)
result = {"nodes":len(G),"edges":G.number_of_edges(),"weak_components":nx.number_weakly_connected_components(G),
    "isolates":len(list(nx.isolates(G))), "known_genre_edges":int(known.sum()),
    "cross_genre_edges":len(cross),"cross_genre_fraction":len(cross)/int(known.sum()),
    "top1pct_artist_count":int(np.ceil(.01*len(stats))),
    "top1pct_edge_share":float(v[:int(np.ceil(.01*len(stats)))].sum()/v.sum()),
    "top20_edge_share":float(v[:20].sum()/v.sum()),"pr_residual_l1":residual,
    "toy_star_pr":{str(k):float(v) for k,v in toy_pr.items()},
    "pr_start_decade_spearman":spearmanr(stats.active_start,stats.reverse_pagerank).statistic,
    "selected_genre":selected_genre,"sub_nodes":len(H),"sub_edges":H.number_of_edges(),
    "sub_global_local_spearman":spearmanr(sub.reverse_pagerank,sub.local_reverse_pr).statistic,
    "sub_view_nodes":len(view),"sub_view_edges":view.number_of_edges()}
(RESULTS/"network_summary.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
