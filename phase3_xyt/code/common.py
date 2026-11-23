from csv_store import read_csv
"""Shared Phase 2 inputs, fixed feature geometry and time-ordered comparisons."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mat3300-mpl')
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from functools import lru_cache
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT.parent / 'phase2'
BASE = P2 / 'analysis/output/phase2'
shared = BASE / 'shared'
scale = read_csv(shared / 'feature_scaler.csv', index_col=0)
F = scale.index.tolist()
Z = ['z_' + f for f in F]
year = read_csv(shared / 'artist_year.csv.gz')
nodes = read_csv(shared / 'nodes.csv').set_index('artist_id')
edges = read_csv(shared / 'edges.csv')
incoming = edges.groupby('follower_id').influencer_id.agg(set).to_dict()
outgoing = edges.groupby('influencer_id').follower_id.agg(set).to_dict()
first_year = year.groupby('artist_id').year.min()
assert len(F) == 10 and nodes.index.is_unique
assert not year.duplicated(['artist_id', 'year']).any()
assert not edges.duplicated(['influencer_id', 'follower_id']).any()
assert np.allclose(year[Z], (year[F].to_numpy() - scale['mean'].to_numpy()) / scale.sd.to_numpy())
plt.rcParams.update({'figure.dpi': 115, 'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .18, 'axes.titleweight': 'bold'})

def profile(rows):
    count = rows.groupby('artist_id').n_songs.sum()
    means = rows[Z].mul(rows.n_songs, axis=0).groupby(rows.artist_id).sum().div(count, axis=0)
    return means.join(count).join(nodes[['genre', 'active_start']])

@lru_cache(None)
def window(start, stop):
    return profile(year[year.year.between(start, stop)])

def export(fig, directory, name):
    fig.savefig(directory / (name + '.png'), dpi=175, bbox_inches='tight')
    plt.close(fig)

@lru_cache(None)
def solo_records():
    songs = read_csv(P2 / 'code/output/full_music_data_clean.csv')
    credit = songs.artists_id.map(ast.literal_eval)
    s = songs[credit.str.len().eq(1)].copy()
    s['artist_id'] = credit[credit.str.len().eq(1)].str[0].astype(int)
    s = s.join(nodes[['genre', 'active_start']], on='artist_id', how='inner')
    valid = (s.tempo.gt(0) & s.duration_ms.gt(0) & s.year.ge(s.active_start)
             & s.release_date.str.extract(r'(\d{4})')[0].astype(int).eq(s.year))
    s = s[valid].copy()
    s['log_duration'] = np.log(s.duration_ms / 1000)
    s[Z] = ((s[F] - scale['mean']) / scale.sd).to_numpy()
    s['n_songs'] = 1
    return s

@lru_cache(None)
def early_solo_profiles():
    """First five observed catalogue years; do not impute the true career beginning."""
    s = solo_records().copy()
    t = s.artist_id.map(first_year)
    p = profile(s[s.year.ge(t) & s.year.le(t + 4)])
    p['start'] = p.index.map(first_year)
    p['stop'] = p.start + 4
    p['recording_decade'] = (p.start // 10) * 10
    p['count_band'] = np.digitize(p.n_songs, [10, 30, 100])
    p['degree_band'] = np.digitize(p.index.map(lambda i: len(incoming.get(i, set()))), [3, 6, 11])
    return p[p.n_songs.ge(5) & p.stop.le(2020) & p.genre.ne('Unknown')]

def compare_followers(source_id, start, stop, target_genre=None, follow_end=None, match_recording_decade=True):
    """Follower-side matching. Positive advantage = followers closer than matched nonfollowers.

    This is deliberately separate from the R5 source-side comparison. Controls share the
    follower's genre, career cohort, first-observed recording decade, credit band and
    incoming-degree band. No causal claim.
    """
    s = solo_records()
    source_rows = s[s.artist_id.eq(source_id) & s.year.between(start, stop)]
    all_targets = set(outgoing.get(source_id, set()))
    eligible = nodes.loc[nodes.index.intersection(list(all_targets))]
    eligible = eligible[eligible.active_start.gt(nodes.at[source_id, 'active_start'])]
    if target_genre:
        eligible = eligible[eligible.genre.eq(target_genre)]
    if follow_end is not None:
        eligible = eligible[eligible.active_start.ge(stop + 1) & (eligible.active_start + 9).le(follow_end)]
    diag = {'artist_id': source_id, 'source_name': nodes.at[source_id, 'artist_name'],
            'source_start': start, 'source_stop': stop, 'source_solo_credits': len(source_rows),
            'network_targets': len(eligible), 'audio_targets': 0, 'matched_targets': 0}
    if len(source_rows) < 5:
        return [], [], diag
    source = source_rows[Z].mean().to_numpy()
    p = early_solo_profiles()
    p = p[p.start.gt(stop)]
    if follow_end is not None:
        p = p[p.stop.le(follow_end)]
    followers = p.loc[p.index.intersection(eligible.index)]
    diag['audio_targets'] = len(followers)
    nonfollowers = p.drop(index=list(all_targets | {source_id}), errors='ignore')
    hist = window(start - 5, start - 1)
    hist = hist[hist.genre.eq(nodes.at[source_id, 'genre']) & hist.n_songs.ge(3)].drop(index=source_id, errors='ignore')
    direction = source - hist[Z].mean().to_numpy() if len(hist) >= 20 else None
    u = direction / np.linalg.norm(direction) if direction is not None and np.linalg.norm(direction) > 0 else None
    pairs, features = [], []
    for j, r in followers.iterrows():
        ctrl = nonfollowers[(nonfollowers.genre == r.genre) & (nonfollowers.active_start == r.active_start)
                            & (nonfollowers.count_band == r.count_band) & (nonfollowers.degree_band == r.degree_band)]
        if match_recording_decade:
            ctrl = ctrl[ctrl.recording_decade.eq(r.recording_decade)]
        if len(ctrl) < 5:
            continue
        y = r[Z].to_numpy(float)
        cv = ctrl[Z].to_numpy(float)
        real = 1 / (1 + np.sqrt(np.mean((source - y) ** 2)))
        baseline = np.mean(1 / (1 + np.sqrt(np.mean((source - cv) ** 2, axis=1))))
        directional = (np.abs((cv - source) @ u).mean() - abs((y - source) @ u)) / np.sqrt(len(F)) if u is not None else np.nan
        pairs.append([source_id, j, int(r.active_start), int(r.start), int(r.stop), int(r.n_songs), len(ctrl), real, baseline, real - baseline, directional, ';'.join(map(str, ctrl.index))])
        advantage = np.abs(cv - source).mean(axis=0) - np.abs(y - source)
        features.extend([[source_id, j, int(r.active_start), f, v] for f, v in zip(F, advantage)])
    diag['matched_targets'] = len(pairs)
    return pairs, features, diag

PAIR_COLUMNS = ['artist_id', 'follower_id', 'target_cohort', 'target_start', 'target_stop', 'target_credits',
                'controls', 'similarity', 'control_similarity', 'excess_similarity', 'direction_advantage', 'control_ids']
FEATURE_COLUMNS = ['artist_id', 'follower_id', 'target_cohort', 'feature', 'distance_advantage']
