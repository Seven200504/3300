"""R2: feature distances, PCA, cohort-controlled comparisons and limited sensitivity checks."""
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from settings import DATA, RESULTS, FEATURES, SEED, plt, save_figure

rng = np.random.default_rng(SEED)
artists = pd.read_csv(DATA / "artist_features.csv").set_index("artist_id")
reference = artists[artists.n_songs.ge(5) & artists.genre.ne("Unknown")].copy()
scaler = StandardScaler().fit(reference[FEATURES])
scale_table = pd.DataFrame({"feature": FEATURES, "mean": scaler.mean_, "scale": scaler.scale_,
                           "reference_artists": len(reference)})
scale_table.to_csv(RESULTS / "feature_scaler.csv", index=False)
all_z = pd.DataFrame(scaler.transform(artists[FEATURES]), index=artists.index,
                     columns=["z_"+c for c in FEATURES])
artists.join(all_z).to_csv(DATA / "artist_features_standardized.csv")
decades = pd.read_csv(DATA / "artist_decade_features.csv")
decade_z = pd.DataFrame(scaler.transform(decades[FEATURES]), columns=["z_"+c for c in FEATURES])
pd.concat([decades, decade_z], axis=1).to_csv(DATA / "artist_decade_standardized.csv", index=False)
Z = scaler.transform(reference[FEATURES])
pca = PCA(svd_solver="full", whiten=False).fit(Z)
scores = pca.transform(Z)
q90 = int(np.searchsorted(pca.explained_variance_ratio_.cumsum(), .90)+1)
variance = pd.DataFrame({"PC": np.arange(1,11), "variance_ratio": pca.explained_variance_ratio_,
                         "cumulative": pca.explained_variance_ratio_.cumsum()})
variance.to_csv(RESULTS/"pca_variance.csv",index=False)
pd.DataFrame(pca.components_.T, index=FEATURES, columns=[f"PC{i}" for i in range(1,11)]).to_csv(RESULTS/"pca_components.csv")
# Correlation loadings are distinct from unit-length eigenvector coefficients.
loading = pca.components_.T * np.sqrt(pca.explained_variance_) / Z.std(axis=0, ddof=1)[:,None]
pd.DataFrame(loading,index=FEATURES,columns=[f"PC{i}" for i in range(1,11)]).to_csv(RESULTS/"pca_correlation_loadings.csv")
pd.DataFrame(scores,index=reference.index,columns=[f"PC{i}" for i in range(1,11)]).to_csv(DATA/"artist_pca_scores.csv")
rotation_error = float(np.max(np.abs(pdist(Z[:100])-pdist(scores[:100]))))
assert rotation_error < 1e-10
assert np.allclose(Z.mean(axis=0),0,atol=1e-12) and np.allclose(Z.std(axis=0),1)

# Musician-level distances. A matrix of ~3,000 artists is modest; no song-pair matrix.
genre_counts = reference.genre.value_counts()
genres = sorted(genre_counts[genre_counts.ge(20)].index)
sample = reference[reference.genre.isin(genres)].copy()
X = scaler.transform(sample[FEATURES])
D = cdist(X, X)/np.sqrt(len(FEATURES))
S = 1/(1+D)
assert np.allclose(D,D.T) and np.allclose(np.diag(D),0) and np.allclose(np.diag(S),1)

def contrast_from_blocks(blocks, labels, n_genres):
    """Equal artists within each genre, then equal genres; exclude self-pairs."""
    within = np.full(len(labels),np.nan)
    between = np.full(len(labels),np.nan)
    for positions, similarity in blocks:
        lab = labels[positions]
        H = (lab[:,None] == np.arange(n_genres)).astype(float)
        counts = H.sum(axis=0)
        present = counts > 0
        if present.sum() < 2:
            continue
        sums = similarity @ H
        means = np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
        own = means[np.arange(len(lab)),lab]
        within[positions] = (sums[np.arange(len(lab)),lab]-np.diag(similarity))/(counts[lab]-1)
        between[positions] = (means[:,present].sum(axis=1)-own)/(present.sum()-1)
    difference = within-between
    valid = np.isfinite(difference)
    group_n = np.bincount(labels[valid],minlength=n_genres)
    group_sum = np.bincount(labels[valid],weights=difference[valid],minlength=n_genres)
    group_delta = np.divide(group_sum,group_n,out=np.full(n_genres,np.nan),where=group_n>0)
    return float(np.nanmean(group_delta)), within, between, group_delta

def evaluate(frame, feature_array, metric="euclidean"):
    """Restrict to >=20 artists/genre and >=5 artists per genre/start-decade cell."""
    keep_genres = frame.genre.value_counts().loc[lambda s:s.ge(20)].index
    cell_n = frame.groupby(["genre","active_start"]).artist_name.transform("size")
    eligible = frame.genre.isin(keep_genres) & cell_n.ge(5)
    m = frame.loc[eligible].copy()
    a = feature_array[eligible.to_numpy()]
    names = sorted(m.genre.unique())
    labels = pd.Categorical(m.genre,categories=names).codes
    # PCA uses /sqrt(original p), so truncation does not add an arbitrary rescaling.
    distance = cdist(a,a,metric=metric)
    similarity = 1-distance if metric=="cosine" else 1/(1+distance/np.sqrt(a.shape[1]))
    blocks=[]
    for _, positions in m.reset_index().groupby("active_start").indices.items():
        blocks.append((positions,similarity[np.ix_(positions,positions)]))
    delta,wi,be,gd = contrast_from_blocks(blocks,labels,len(names))
    return delta,m,names,labels,blocks,wi,be,gd,similarity

baseline = evaluate(sample,X)
delta,matched,matched_genres,labels,blocks,wi,be,gd,matched_S = baseline
matched["within_similarity"] = wi
matched["between_similarity"] = be
matched["difference"] = wi-be
matched.to_csv(RESULTS/"r2_artist_contrasts.csv")
genre_result = matched.groupby("genre").agg(n=("artist_name","size"),
    within_similarity=("within_similarity","mean"),between_similarity=("between_similarity","mean"),
    difference=("difference","mean"))
all_labels = pd.Categorical(sample.genre,categories=genres).codes
unadjusted = contrast_from_blocks([(np.arange(len(sample)),S)],all_labels,len(genres))
genre_result["unadjusted_difference"] = pd.Series(unadjusted[3],index=genres)
genre_result.to_csv(RESULTS/"r2_genre_results.csv")

# The randomization unit is an artist label, NOT an artist pair.
# Permutations retain all within-cohort genre counts and reuse each fixed distance block.
null = np.empty(499)
for b in range(len(null)):
    shuffled = labels.copy()
    for positions,_ in blocks:
        shuffled[positions] = rng.permutation(shuffled[positions])
    null[b] = contrast_from_blocks(blocks,shuffled,len(matched_genres))[0]
p_value = (1+np.sum(null>=delta))/(len(null)+1)
pd.DataFrame({"permutation":np.arange(1,len(null)+1),"macro_difference":null}).to_csv(RESULTS/"r2_permutation.csv",index=False)

# Descriptive heatmap averages ALL distinct artist pairs (not centroid similarities).
heat = np.empty((len(genres),len(genres)))
for g,genre in enumerate(genres):
    ig=np.flatnonzero(sample.genre.eq(genre))
    for h,other in enumerate(genres):
        ih=np.flatnonzero(sample.genre.eq(other))
        values=S[np.ix_(ig,ih)]
        heat[g,h]=(values.sum()-len(ig))/(len(ig)*(len(ig)-1)) if g==h else values.mean()
pd.DataFrame(heat,index=genres,columns=genres).to_csv(RESULTS/"genre_similarity.csv")

# Limited sensitivity checks: no parameter grid, fixed main scaling throughout.
sensitivity=[]
genre_sensitivity=[]
def record(label,frame,array,metric="euclidean"):
    r=evaluate(frame,array,metric)
    sensitivity.append({"variant":label,"artists":len(r[1]),"genres":len(r[2]),
        "macro_difference":r[0],"positive_genres":int(np.sum(r[7]>0))})
    genre_sensitivity.extend({"variant":label,"genre":g,"difference":float(d)}
                             for g,d in zip(r[2],r[7]))
    return r
record("Main Euclidean",sample,X)
record("Cosine",sample,X,"cosine")
# Add zero columns: normalization stays sqrt(10), as in the main distance.
pca_X=pca.transform(X)
pca90=np.zeros_like(pca_X);pca90[:,:q90]=pca_X[:,:q90]
record(f"PCA {q90} PCs (90% variance)",sample,pca90)
pca2=np.zeros_like(pca_X);pca2[:,:2]=pca_X[:,:2]
record("PCA 2 PCs",sample,pca2)
record("Exclude loudness",sample,np.delete(X,FEATURES.index("loudness"),axis=1))
record("Exclude liveness and speechiness",sample,np.delete(X,[FEATURES.index("liveness"),FEATURES.index("speechiness")],axis=1))
for threshold in [3,10]:
    frame=artists[artists.n_songs.ge(threshold)&artists.genre.ne("Unknown")]
    record(f"At least {threshold} songs",frame,scaler.transform(frame[FEATURES]))
long=pd.read_csv(DATA/"artist_song_records.csv.gz")
usable=long[long.in_clean_network&long.valid_features].copy()
weighted=(usable[FEATURES].mul(usable.credit_weight,axis=0).groupby(usable.artist_id).sum()
          .div(usable.groupby("artist_id").credit_weight.sum(),axis=0))
record("Fractional collaborative credits",sample,scaler.transform(weighted.loc[sample.index,FEATURES]))
medians=usable.groupby("artist_id")[FEATURES].median()
record("Artist medians",sample,scaler.transform(medians.loc[sample.index,FEATURES]))
solo=usable[usable.n_artists.eq(1)]
solo_mean=solo.groupby("artist_id")[FEATURES].mean()
solo_n=solo.groupby("artist_id").size()
solo_ids=sample.index.intersection(solo_n[solo_n.ge(5)].index)
record("Solo-only (common eligible artists)",sample.loc[solo_ids],scaler.transform(solo_mean.loc[solo_ids,FEATURES]))
record("Main on solo-eligible artists",sample.loc[solo_ids],scaler.transform(sample.loc[solo_ids,FEATURES]))
# Retain ambiguous IDs only as a diagnostic; labels remain raw, not claimed corrected.
raw_nodes=pd.read_csv(RESULTS/"raw_node_labels.csv").set_index("artist_id")
amb=long[long.artist_id.isin(raw_nodes.index[raw_nodes.identity_conflict]) & long.valid_features]
amb_features=amb.groupby("artist_id")[FEATURES].mean().join(amb.groupby("artist_id").size().rename("n_songs"))
amb_features=raw_nodes[["artist_name","genre","active_start"]].join(amb_features,how="inner")
amb_features=amb_features[amb_features.n_songs.ge(5)]
augmented=pd.concat([sample,amb_features])
record("Keep ambiguous IDs (diagnostic)",augmented,scaler.transform(augmented[FEATURES]))
sens=pd.DataFrame(sensitivity)
sens.to_csv(RESULTS/"similarity_sensitivity.csv",index=False)
pd.DataFrame(genre_sensitivity).to_csv(RESULTS/"genre_sensitivity.csv",index=False)

# EDA: shapes, correlated dimensions, song-count imbalance and genre profiles.
fig,axes=plt.subplots(2,5,figsize=(14,6))
for ax,col in zip(axes.flat,FEATURES):
    ax.hist(reference[col],bins=35,color="#4b8093",alpha=.85)
    ax.set_title(col,fontsize=10);ax.set_ylabel("Artists")
fig.suptitle("Artist feature distributions (at least 5 usable songs)")
fig.tight_layout();save_figure(fig,"04_feature_distributions")
corr=reference[FEATURES].corr()
corr.to_csv(RESULTS/"feature_correlations.csv")
fig,axes=plt.subplots(1,2,figsize=(13,5),gridspec_kw={"width_ratios":[1.3,1]})
im=axes[0].imshow(corr,vmin=-1,vmax=1,cmap="RdBu_r")
axes[0].set_xticks(range(10),FEATURES,rotation=55,ha="right",fontsize=8)
axes[0].set_yticks(range(10),FEATURES,fontsize=8)
axes[0].set_title("Artist-level Pearson correlations")
fig.colorbar(im,ax=axes[0],fraction=.045)
axes[1].hist(artists.n_songs,bins=np.geomspace(1,artists.n_songs.max()+1,35),color="#b18356")
axes[1].set(xscale="log",xlabel="Usable song records per artist",ylabel="Artists",title="Uneven precision of artist averages")
axes[1].axvline(5,color="#333",ls="--",label="Main threshold: 5");axes[1].legend()
fig.tight_layout();save_figure(fig,"05_correlation_and_counts")
profiles=pd.DataFrame(X,index=sample.index,columns=FEATURES).groupby(sample.genre).mean().reindex(genres)
fig,ax=plt.subplots(figsize=(10,6))
im=ax.imshow(profiles,cmap="RdBu_r",vmin=-2,vmax=2,aspect="auto")
ax.set_xticks(range(10),FEATURES,rotation=45,ha="right");ax.set_yticks(range(len(genres)),genres)
ax.set_title("Genre profiles: mean standardized artist features (descriptive)")
fig.colorbar(im,ax=ax,label="Standard deviations from reference mean")
fig.tight_layout();save_figure(fig,"06_genre_feature_profiles")

fig,axes=plt.subplots(1,3,figsize=(16,5),gridspec_kw={"width_ratios":[1,1.15,1.2]})
axes[0].bar(variance.PC,variance.variance_ratio,color="#4b8093",label="Individual")
axes[0].plot(variance.PC,variance.cumulative,"o-",color="#bd8748",label="Cumulative")
axes[0].axhline(.9,ls="--",color="gray");axes[0].set(xlabel="Principal component",ylabel="Variance fraction",title="Variance retained",xticks=range(1,11));axes[0].legend()
im=axes[1].imshow(loading[:,:3],vmin=-1,vmax=1,cmap="RdBu_r",aspect="auto")
axes[1].set_yticks(range(10),FEATURES,fontsize=8);axes[1].set_xticks(range(3),["PC1","PC2","PC3"])
axes[1].set_title("Feature-PC correlations")
for i in range(10):
    for j in range(3):axes[1].text(j,i,f"{loading[i,j]:.2f}",ha="center",va="center",fontsize=7,color="black")
largest=list(reference.genre.value_counts().head(5).index)
for k,g in enumerate(largest):
    indices=np.flatnonzero(reference.genre.eq(g))
    chosen=rng.choice(indices,size=min(150,len(indices)),replace=False)
    axes[2].scatter(scores[chosen,0],scores[chosen,1],s=12,alpha=.45,label=g)
axes[2].set(xlabel=f"PC1 ({variance.variance_ratio.iloc[0]:.1%})",ylabel=f"PC2 ({variance.variance_ratio.iloc[1]:.1%})",title="Balanced display of five largest genres")
axes[2].legend(fontsize=8,loc="best")
fig.tight_layout();save_figure(fig,"07_pca")

# Pair histograms are illustration only; exact macro statistics above are primary.
within_pairs=[];between_pairs=[]
for g in matched_genres:
    focals=np.flatnonzero(matched.genre.eq(g)&np.isfinite(wi))
    for i in rng.choice(focals,size=300,replace=True):
        cohort=matched.active_start.iloc[i]
        same=np.flatnonzero(matched.genre.eq(g)&matched.active_start.eq(cohort));same=same[same!=i]
        others=matched.loc[matched.active_start.eq(cohort)&matched.genre.ne(g),"genre"].unique()
        h=rng.choice(others)
        other=np.flatnonzero(matched.genre.eq(h)&matched.active_start.eq(cohort))
        within_pairs.append(matched_S[i,rng.choice(same)])
        between_pairs.append(matched_S[i,rng.choice(other)])
fig,axes=plt.subplots(1,2,figsize=(13,5))
axes[0].hist(within_pairs,bins=30,range=(0,1),density=True,alpha=.55,label="Within genre",color="#287b8e")
axes[0].hist(between_pairs,bins=30,range=(0,1),density=True,alpha=.5,label="Between genres",color="#bf8b51")
axes[0].set(xlabel="Similarity 1/(1 + standardized distance)",ylabel="Density",title="Same-cohort, genre-balanced sampled pairs")
axes[0].legend()
ordered=genre_result.sort_values("difference")
axes[1].barh(ordered.index,ordered.difference,color=np.where(ordered.difference>0,"#287b8e","#c47056"))
axes[1].axvline(0,color="gray",lw=1);axes[1].set(xlabel="Mean within - mean between similarity",title="Exact artist and genre-balanced contrasts")
fig.tight_layout();save_figure(fig,"08_r2_comparison")
fig,ax=plt.subplots(figsize=(10,8))
im=ax.imshow(heat,vmin=0,vmax=1,cmap="YlGnBu")
ax.set_xticks(range(len(genres)),genres,rotation=50,ha="right");ax.set_yticks(range(len(genres)),genres)
for i in range(len(genres)):
    for j in range(len(genres)):ax.text(j,i,f"{heat[i,j]:.2f}",ha="center",va="center",fontsize=6)
ax.set_title("Average distinct-artist similarity by genre\nDescriptive all-cohort means; self-pairs excluded")
fig.colorbar(im,ax=ax,label="Similarity");fig.tight_layout();save_figure(fig,"09_similarity_heatmap")
fig,axes=plt.subplots(1,2,figsize=(14,5),gridspec_kw={"width_ratios":[1,1.6]})
axes[0].hist(null,bins=25,color="#a5bbc9")
axes[0].axvline(delta,color="#a74d39",lw=2,label=f"Observed {delta:.4f}")
axes[0].set(xlabel="Macro within-minus-between difference",ylabel="Permutations",title=f"Within-cohort label permutations (B=499)\np = {p_value:.3f}")
axes[0].legend()
se=sens[sens.variant.ne("Cosine")].iloc[::-1]
axes[1].barh(se.variant,se.macro_difference,color="#4b8093")
axes[1].axvline(0,color="gray");axes[1].set(xlabel="Macro similarity difference",title="Selected Euclidean sensitivity checks")
axes[1].tick_params(axis="y",labelsize=8)
fig.tight_layout();save_figure(fig,"10_validation")

# Feasibility only: no R4 influence tests, R5 change-points or R6 dynamics are fitted.
edge=pd.read_csv(DATA/"influence_edges.csv")
eligible_ids=set(reference.index)
supported=edge[edge.source.isin(eligible_ids)&edge.target.isin(eligible_ids)].copy()
supported=supported.join(reference[["genre","active_start"]],on="source")
control_sizes=[]
incoming=edge.groupby("target").source.agg(set)
for row in supported.itertuples():
    candidates=set(reference.index[reference.genre.eq(row.genre)&reference.active_start.eq(row.active_start)])
    control_sizes.append(len(candidates-incoming[row.target]-{row.target}))
supported["potential_controls_same_source_genre_cohort"]=control_sizes
supported.to_csv(RESULTS/"r4_support_only.csv",index=False)
summary={"reference_artists":len(reference),"descriptive_artists":len(sample),"descriptive_genres":len(genres),
    "controlled_artists":len(matched),"controlled_genres":len(matched_genres),
    "macro_within":float(genre_result.within_similarity.mean()),"macro_between":float(genre_result.between_similarity.mean()),
    "macro_delta":delta,"unadjusted_macro_delta":unadjusted[0],"positive_genres":int((gd>0).sum()),
    "permutation_B":len(null),"permutation_p":float(p_value),"null_mean":float(null.mean()),
    "null_95_percentiles":[float(x) for x in np.quantile(null,[.025,.975])],
    "pc1_pc2_variance":float(variance.variance_ratio.iloc[:2].sum()),"pcs_for90pct":q90,
    "full_pca_distance_error":rotation_error,
    "pca90_distance_rank_correlation":float(spearmanr(pdist(X[:500]),pdist(pca90[:500])).statistic),
    "r4_supported_edges":len(supported),"r4_edges_with5controls":int(supported.potential_controls_same_source_genre_cohort.ge(5).sum()),
    "feature_correlation_energy_loudness":float(corr.loc["energy","loudness"]),
    "feature_correlation_energy_acousticness":float(corr.loc["energy","acousticness"])}
(RESULTS/"similarity_summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print(sens.to_string(index=False))
