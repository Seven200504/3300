"""Small independent numerical checks and coverage diagnostics; no new requirement models."""
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from settings import DATA, RESULTS, RAW, FEATURES, SEED

meta = pd.read_csv(RESULTS/"r2_artist_contrasts.csv").set_index("artist_id")
scale = pd.read_csv(RESULTS/"feature_scaler.csv").set_index("feature")
Z = (meta[FEATURES]-scale.loc[FEATURES,"mean"].to_numpy())/scale.loc[FEATURES,"scale"].to_numpy()
errors=[]
# Recompute eight focal contrasts directly from the defining scalar distance formula.
for artist_id in meta.sample(8,random_state=SEED).index:
    focal=meta.loc[artist_id]
    same_time=meta.index[meta.active_start.eq(focal.active_start)]
    similarity=1/(1+np.sqrt(((Z.loc[same_time]-Z.loc[artist_id])**2).mean(axis=1)))
    grouped=meta.loc[same_time,["genre"]].join(similarity.rename("s"))
    w=grouped.loc[grouped.genre.eq(focal.genre)&grouped.index.to_series().ne(artist_id),"s"].mean()
    b=grouped.loc[grouped.genre.ne(focal.genre)].groupby("genre").s.mean().mean()
    errors.append(abs((w-b)-focal.difference))
assert max(errors)<1e-12
genre=pd.read_csv(RESULTS/"r2_genre_results.csv")
summary=json.loads((RESULTS/"similarity_summary.json").read_text())
assert np.isclose(genre.difference.mean(),summary["macro_delta"])

# Coverage and flags, never automatically repair names or reassign an artist ID.
long=pd.read_csv(DATA/"artist_song_records.csv.gz")
nodes=pd.read_csv(DATA/"artist_nodes.csv")
artists=pd.read_csv(DATA/"artist_features.csv").set_index("artist_id")
provided=pd.read_csv(RAW/"data_by_artist.csv")
alias=provided[provided.artist_name.duplicated(keep=False)].sort_values(["artist_name","artist_id"])
alias.to_csv(RESULTS/"same_name_multiple_ids.csv",index=False)
coverage=nodes.groupby("genre").agg(network=("artist_id","size"),has_features=("has_detailed_features","sum"))
coverage["main_R2_controlled"]=meta.groupby("genre").size()
coverage.fillna(0).astype(int).to_csv(RESULTS/"analysis_sample_flow.csv")

# All 19 known genre labels receive exploratory summaries; small groups are not
# included in the main inference (which requires >=20 artists before cohort restriction).
ref=artists[artists.n_songs.ge(5)&artists.genre.ne("Unknown")]
refZ=(ref[FEATURES]-scale.loc[FEATURES,"mean"].to_numpy())/scale.loc[FEATURES,"scale"].to_numpy()
similarity=1/(1+cdist(refZ,refZ)/np.sqrt(len(FEATURES)))
rows=[]
for g,idx in ref.reset_index().groupby("genre").indices.items():
    own=similarity[np.ix_(idx,idx)]
    within=(own.sum()-len(idx))/(len(idx)*(len(idx)-1)) if len(idx)>1 else np.nan
    between=[]
    for h,jdx in ref.reset_index().groupby("genre").indices.items():
        if h!=g:between.append(similarity[np.ix_(idx,jdx)].mean())
    rows.append({"genre":g,"artists_5songs":len(idx),"within":within,
                 "between_genre_balanced":np.mean(between),"difference":within-np.mean(between)})
pd.DataFrame(rows).to_csv(RESULTS/"all_genres_exploratory.csv",index=False)

usable=long[long.in_clean_network&long.valid_features]
time_rates=usable.groupby("genre").time_ok.agg(["size","mean"])
time_rates["flagged_fraction"]=1-time_rates["mean"]
time_rates.to_csv(RESULTS/"time_flag_rates.csv")
byyear=usable[usable.time_ok].groupby("year").agg(credits=("track_row","size"),artists=("artist_id","nunique"))
byyear.to_csv(RESULTS/"year_coverage_only.csv")
checks={"independent_focal_max_error":float(max(errors)),"focal_checks":len(errors),
        "r2_missing_contrasts":int(meta.difference.isna().sum()),
        "genre_coverage_main":len(genre),"genre_coverage_exploratory":len(rows),
        "main_sample_before1960":int(meta.active_start.lt(1960).sum())}
(RESULTS/"verification.json").write_text(json.dumps(checks,indent=2))
print(checks)
