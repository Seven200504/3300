"""Read the four supplied CSVs; audit, then prepare four reusable data tables."""
import ast
import hashlib
import json
import numpy as np
import pandas as pd
from settings import RAW, DATA, RESULTS, FEATURES

edges = pd.read_csv(RAW / "influence_data.csv")
songs = pd.read_csv(RAW / "full_music_data.csv")
artist_summary = pd.read_csv(RAW / "data_by_artist.csv")
year_summary = pd.read_csv(RAW / "data_by_year.csv")
files = dict(influence_data=edges, full_music_data=songs,
             data_by_artist=artist_summary, data_by_year=year_summary)
audit = {name: {"rows": len(df), "columns": len(df.columns),
                "missing_cells": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "sha256": hashlib.sha256((RAW / f"{name}.csv").read_bytes()).hexdigest()}
         for name, df in files.items()}

# Build a node table from both ends; do not merge artists merely by name.
appearances = pd.concat([
    edges[[c for c in edges if c.startswith(side)]].rename(columns=lambda c: c[len(side):])
    for side in ["influencer_", "follower_"]], ignore_index=True)
conflict_counts = appearances.groupby("id")[["name", "main_genre", "active_start"]].nunique()
conflict_ids = conflict_counts.index[conflict_counts.gt(1).any(axis=1)]
appearances[appearances.id.isin(conflict_ids)].drop_duplicates().to_csv(
    RESULTS / "identity_conflicts.csv", index=False)
nodes_all = appearances.drop_duplicates("id").rename(
    columns={"id": "artist_id", "name": "artist_name", "main_genre": "genre"})
nodes_all["genre"] = nodes_all.genre.replace({"R&B;": "R&B"})
nodes_all["identity_conflict"] = nodes_all.artist_id.isin(conflict_ids)
nodes = nodes_all[~nodes_all.identity_conflict].copy().sort_values("artist_id")
clean_edges = edges[~edges.influencer_id.isin(conflict_ids)
                    & ~edges.follower_id.isin(conflict_ids)].copy()
clean_edges = clean_edges.rename(columns={"influencer_id": "source", "follower_id": "target"})
clean_edges = clean_edges[["source", "target"]].drop_duplicates()
assert not nodes.artist_id.duplicated().any()
assert not clean_edges.source.eq(clean_edges.target).any()
assert set(clean_edges.source) | set(clean_edges.target) <= set(nodes.artist_id)
clean_edges.to_csv(DATA / "influence_edges.csv", index=False)

# Audit the unmodified detailed data against the supplied summaries.
raw_ids = songs.artists_id.map(ast.literal_eval)
raw_names = songs.artist_names.map(ast.literal_eval)
assert (raw_ids.str.len() == raw_names.str.len()).all()
assert raw_ids.map(lambda v: len(v) == len(set(v)) and len(v) > 0).all()
raw_long = songs.assign(artist_id=raw_ids).explode("artist_id")
raw_long["artist_id"] = raw_long.artist_id.astype(int)
numeric = [c for c in FEATURES if c != "log_duration"] + ["duration_ms", "popularity"]
recomputed = raw_long.groupby("artist_id")[numeric].mean()
provided = artist_summary.set_index("artist_id")
comparison = (recomputed - provided.loc[recomputed.index, numeric]).abs()
count_compare = provided[["count"]].join(raw_long.groupby("artist_id").size().rename("observed_count"))
count_compare.to_csv(RESULTS / "summary_count_audit.csv")
comparison.to_csv(RESULTS / "summary_feature_abs_difference.csv")
year_difference = (songs.groupby("year")[numeric].mean()
                   - year_summary.set_index("year")[numeric]).abs()
year_difference.to_csv(RESULTS / "year_summary_abs_difference.csv")

# Time fields are audited independently of musical features.
date_year = pd.to_numeric(songs.release_date.str.extract(r"(\d{4})")[0])
raw_long = raw_long.merge(nodes_all[["artist_id", "active_start"]], on="artist_id", how="left",
                          validate="many_to_one")
raw_before = raw_long.year < raw_long.active_start
audit["raw_checks"] = {
    "network_nodes": len(nodes_all), "conflicting_ids": [int(x) for x in conflict_ids],
    "duplicate_edges": int(edges.duplicated(["influencer_id", "follower_id"]).sum()),
    "self_edges": int(edges.influencer_id.eq(edges.follower_id).sum()),
    "multi_artist_tracks": int(raw_ids.str.len().gt(1).sum()),
    "artist_song_credits": len(raw_long), "detailed_artist_ids": int(raw_long.artist_id.nunique()),
    "summary_count_total": int(artist_summary["count"].sum()),
    "matching_counts": int(count_compare["count"].eq(count_compare.observed_count).sum()),
    "matching_all_means_1e5": int(comparison.lt(1e-5).all(axis=1).sum()),
    "missing_song_years": sorted(set(range(1921, 2021)) - set(songs.year)),
    "release_year_conflicts": int(date_year.ne(songs.year).sum()),
    "credits_before_career": int(raw_before.sum()),
    "artists_before_career": int(raw_long.loc[raw_before, "artist_id"].nunique()),
    "tempo_zero": int(songs.tempo.eq(0).sum()), "positive_loudness": int(songs.loudness.gt(0).sum()),
    "duration_under_30s": int(songs.duration_ms.lt(30000).sum()),
    "duration_over_20min": int(songs.duration_ms.gt(1200000).sum()),
    "edges_later_start": int(edges.influencer_active_start.gt(edges.follower_active_start).sum()),
    "edges_same_start": int(edges.influencer_active_start.eq(edges.follower_active_start).sum())}

# Exact duplicate rows are removed; different versions of the same title remain.
tracks = songs.drop_duplicates().copy()
tracks["track_row"] = tracks.index  # Stable source-row identifier, not an invented Spotify track ID.
tracks["artist_id"] = tracks.artists_id.map(ast.literal_eval)
tracks["n_artists"] = tracks.artist_id.str.len()
tracks["first_artist_id"] = tracks.artist_id.str[0]
tracks["date_year_conflict"] = pd.to_numeric(tracks.release_date.str.extract(r"(\d{4})")[0]).ne(tracks.year)
tracks["valid_features"] = (tracks.tempo > 0) & (tracks.duration_ms > 0)
tracks["log_duration"] = np.log(tracks.duration_ms / 1000)
bounded = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "liveness", "speechiness"]
assert tracks[bounded].ge(0).all().all() and tracks[bounded].le(1).all().all()
assert np.isfinite(tracks[FEATURES]).all().all()
tracks.loc[~tracks.valid_features | tracks.date_year_conflict,
           ["track_row", "artist_names", "year", "release_date", "tempo", "duration_ms",
            "valid_features", "date_year_conflict"]].to_csv(RESULTS / "flagged_tracks.csv", index=False)
long = tracks.explode("artist_id").copy()
long["artist_id"] = long.artist_id.astype(int)
long = long.merge(nodes_all[["artist_id", "genre", "active_start", "identity_conflict"]],
                  on="artist_id", how="left", validate="many_to_one")
long["in_clean_network"] = long.artist_id.isin(nodes.artist_id)
long["time_ok"] = (~long.date_year_conflict & long.active_start.notna()
                   & long.year.ge(long.active_start))
long["decade"] = long.year // 10 * 10
long["credit_weight"] = 1 / long.n_artists
long["is_first_artist"] = long.artist_id.eq(long.first_artist_id)
long.to_csv(DATA / "artist_song_records.csv.gz", index=False)
usable = long[long.in_clean_network & long.valid_features].copy()

artist_features = usable.groupby("artist_id")[FEATURES].mean()
artist_features = artist_features.join(usable.groupby("artist_id").agg(
    n_songs=("track_row", "nunique"), first_release=("year", "min"), last_release=("year", "max"),
    fraction_collaborative=("n_artists", lambda s: s.gt(1).mean()),
    fraction_time_flagged=("time_ok", lambda s: 1-s.mean())))
artist_features = nodes[["artist_id", "artist_name", "genre", "active_start"]].merge(
    artist_features, on="artist_id", how="inner", validate="one_to_one")
artist_features.to_csv(DATA / "artist_features.csv", index=False)
temporal = usable[usable.time_ok].copy()
decade_features = temporal.groupby(["artist_id", "decade"])[FEATURES].mean()
decade_features = decade_features.join(temporal.groupby(["artist_id", "decade"]).size().rename("n_songs"))
decade_features = decade_features.reset_index().merge(nodes[["artist_id", "genre", "active_start"]],
                                                      on="artist_id", validate="many_to_one")
decade_features.to_csv(DATA / "artist_decade_features.csv", index=False)
nodes["has_summary"] = nodes.artist_id.isin(artist_summary.artist_id)
nodes["has_detailed_features"] = nodes.artist_id.isin(artist_features.artist_id)
nodes = nodes.merge(artist_features[["artist_id", "n_songs"]], on="artist_id", how="left", validate="one_to_one")
nodes["n_songs"] = nodes.n_songs.fillna(0).astype(int)
nodes.to_csv(DATA / "artist_nodes.csv", index=False)
nodes_all.to_csv(RESULTS / "raw_node_labels.csv", index=False)

coverage = nodes.groupby("genre").agg(network_artists=("artist_id", "size"),
    artists_with_songs=("has_detailed_features", "sum"), artists_5songs=("n_songs", lambda s: s.ge(5).sum()))
good_decades = decade_features[decade_features.n_songs.ge(5)].groupby("artist_id").size()
coverage["artists_3_decades_5songs"] = nodes[nodes.artist_id.isin(good_decades[good_decades.ge(3)].index)].groupby("genre").size()
coverage = coverage.fillna(0).astype(int)
coverage.to_csv(RESULTS / "genre_coverage.csv")
decade_features.groupby(["genre", "decade"]).agg(artists=("artist_id", "nunique"),
    songs=("n_songs", "sum"), artists_5songs=("n_songs", lambda s: s.ge(5).sum())).to_csv(RESULTS / "temporal_coverage.csv")
audit["prepared"] = {"nodes": len(nodes), "edges": len(clean_edges),
    "removed_edges_identity": len(edges)-len(clean_edges), "deduplicated_tracks": len(tracks),
    "usable_credits": len(usable), "artists_features": len(artist_features),
    "artists_at_least5": int(artist_features.n_songs.ge(5).sum()),
    "artist_decade_rows": len(decade_features), "temporal_credits": len(temporal)}
(RESULTS / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
print(json.dumps(audit["prepared"], indent=2))
