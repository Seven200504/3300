from csv_store import read_csv
"""Independent checks of the final tables, matching eligibility and accounting identities."""
from common import *
import json
OUT=ROOT/'output'
checks={}
def check(name,condition):
    if not bool(condition):raise AssertionError(name)
    checks[name]=True
# Reconstruct one temporal aggregate directly from artist-year rows.
a=read_csv(OUT/'r5/annual.csv').set_index('year')
check('annual_mean_reconstructed',np.allclose(a.loc[2015,Z],year[year.year.eq(2015)][Z].mean()))
local=read_csv(OUT/'r5/transition_candidates.csv')
check('period_candidates_meet_declared_rules',((local.loc[local.selected,'reach_pct']>=.75)&(local.loc[local.selected,'later_followers']>0)&(local.loc[local.selected,'transition_projection']>0)).all())
for r in local[local.selected].itertuples():
    target=edges[edges.influencer_id.eq(r.artist_id)&edges.follower_active_start.ge(r.stop+1)&(edges.follower_active_start+9).le(r.start+24)]
    check(f'period_{r.artist_id}_{r.start}_reach',len(target)==r.later_followers)
    h=window(r.start-5,r.start-1);h=h[h.genre.eq(r.genre)&h.n_songs.ge(3)].drop(index=r.artist_id,errors='ignore')
    p=window(r.start,r.stop).loc[r.artist_id,Z].to_numpy(float)
    nearest=np.sort(np.linalg.norm(h[Z].to_numpy()-p,axis=1)/np.sqrt(10))[:3].mean()
    check(f'period_{r.artist_id}_{r.start}_nearest',np.isclose(nearest,r.nearest3_distance))
# Independent reconstruction of the new revolution-linked evidence.
features=read_csv(OUT/'r5/transition_feature_links.csv')
for r in local.itertuples():
    h=window(r.start-5,r.start-1);h=h[h.genre.eq(r.genre)&h.n_songs.ge(3)].drop(index=r.artist_id,errors='ignore')
    v=window(r.start,r.stop).loc[r.artist_id,Z].to_numpy(float)-h[Z].mean().to_numpy()
    m=window(1970,1974).drop(index=r.artist_id,errors='ignore')[Z].mean().to_numpy()-window(1960,1964).drop(index=r.artist_id,errors='ignore')[Z].mean().to_numpy()
    u=m/np.linalg.norm(m);q=features[(features.artist_id==r.artist_id)&(features.start==r.start)].set_index('feature').loc[F]
    assert np.allclose(q.deviation,v) and np.allclose(q.macro_shift_excluding_artist,m)
    assert np.allclose(q.projection_component,v*u/np.sqrt(10))
    assert np.isclose(q.projection_component.sum(),r.transition_projection)
    assert np.isclose(q.novelty_squared_share.sum(),1) and np.allclose(q.novelty_squared_share,v*v/(v@v))
checks['r5_all_transition_directions_exclude_artist_and_components_reconstruct']=True
sensitivity=read_csv(OUT/'r5/transition_selection_sensitivity.csv')
for cutoff in [.5,.75,.9]:
    for metric in ['transition_projection','projection_no_liveness']:
        for t,g in local.groupby('start'):
            pool=g[g.reach_pct.ge(cutoff)&g.later_followers.gt(0)&g[metric].gt(0)]
            expected=set(pool.sort_values([metric,'artist_id'],ascending=[False,True]).head(2).artist_id)
            observed=sensitivity[(sensitivity.reach_cutoff==cutoff)&(sensitivity.projection_metric==metric)&(sensitivity.start==t)]
            assert set(observed.artist_id)==expected
            if cutoff==.75 and metric=='transition_projection':assert set(g[g.selected].artist_id)==expected
checks['r5_new_selection_and_all_six_sensitivity_settings_reconstructed']=True
timing=read_csv(OUT/'r5/transition_timing.csv')
for r in timing.itertuples():
    g=nodes.at[r.artist_id,'genre'];h=window(1955,1959);h=h[h.genre.eq(g)&h.n_songs.ge(3)].drop(index=r.artist_id,errors='ignore')
    w=window(r.period_start,r.period_start+4);peer=w[w.genre.eq(g)&w.n_songs.ge(3)].drop(index=r.artist_id,errors='ignore')
    u=features[(features.artist_id==r.artist_id)&(features.start==r.selected_start)].set_index('feature').loc[F,'unit_direction'].to_numpy()
    assert r.baseline_n==len(h) and r.peer_n==len(peer)
    if r.credits>=5 and len(h)>=20:
        assert np.isclose(r.artist_projection,(w.loc[r.artist_id,Z].to_numpy(float)-h[Z].mean().to_numpy())@u/np.sqrt(10))
    else:assert pd.isna(r.artist_projection)
    if len(peer)>=20 and len(h)>=20:assert np.isclose(r.peer_projection,(peer[Z].mean().to_numpy()-h[Z].mean().to_numpy())@u/np.sqrt(10))
    else:assert pd.isna(r.peer_projection)
checks['r5_timing_fixed_baseline_missingness_and_peer_exclusion']=True
retention=read_csv(OUT/'r5/transition_persistence.csv').set_index('start')
five=read_csv(OUT/'r5/five_year.csv').set_index('start');m=(five.loc[1970,Z]-five.loc[1960,Z]).to_numpy()
for t,r in retention.iterrows():
    d=(five.loc[t,Z]-five.loc[1960,Z]).to_numpy()
    assert np.allclose(r[Z].to_numpy(float),d) and np.isclose(r.retention_vs_1970,d@m/(m@m))
checks['r5_direction_retention_reconstructed']=True
shift=read_csv(OUT/'r5/bin_origin_stability.csv')
check('all_five_bin_origins',set(shift.origin_offset)==set(range(5)))
source=read_csv(OUT/'r5/alignment.csv')
check('source_match_contrasts',np.allclose(source.real-source.baseline,source.delta))
check('source_match_original_support',len(source)==34 and source.controls.ge(5).all())
# Recompute each follower-side contrast and inspect actual control IDs, not only counts.
p=early_solo_profiles();raw=solo_records()
for section in ['r5','r6']:
    filename='follower_matched_pairs' if section=='r5' else 'leader_follower_pairs'
    pairs=read_csv(OUT/section/(filename+'.csv'))
    support=read_csv(OUT/section/('follower_matched_support.csv' if section=='r5' else 'leader_follower_support.csv'))
    check(section+'_support_conservation',len(pairs)==support.matched_targets.sum())
    for r in pairs.itertuples():
        start=int(r.source_window_start) if section=='r5' else int(first_year[r.artist_id])
        stop=start+4;y=p.loc[r.follower_id];ids=list(map(int,r.control_ids.split(';')))
        ctrl=p.loc[ids]
        assert len(ids)==r.controls>=5 and len(set(ids))==len(ids)
        assert not set(ids).intersection(outgoing.get(r.artist_id,set())|{r.artist_id})
        for key in ['genre','active_start','recording_decade','count_band','degree_band']:assert ctrl[key].eq(y[key]).all()
        assert y.start>stop and ctrl.start.gt(stop).all()
        if section=='r5':assert y.stop<=start+24 and ctrl.stop.le(start+24).all()
        source_profile=raw[raw.artist_id.eq(r.artist_id)&raw.year.between(start,stop)][Z].mean().to_numpy()
        actual=1/(1+np.sqrt(np.mean((source_profile-y[Z].to_numpy(float))**2)))
        base=np.mean(1/(1+np.sqrt(np.mean((ctrl[Z].to_numpy()-source_profile)**2,axis=1))))
        assert np.isclose(actual,r.similarity) and np.isclose(base,r.control_similarity) and np.isclose(actual-base,r.excess_similarity)
    checks[section+'_all_match_ids_time_order_and_contrasts']=True

tf=read_csv(OUT/'r5/transition_followers.csv')
original=read_csv(OUT/'r5/follower_matched_pairs.csv')
for r in tf.itertuples():
    old=original[(original.artist_id==r.artist_id)&(original.source_window_start==r.start)&(original.follower_id==r.follower_id)].iloc[0]
    assert old.control_ids==r.control_ids
    ids=list(map(int,r.control_ids.split(';')))
    source=raw[raw.artist_id.eq(r.artist_id)&raw.year.between(r.start,r.start+4)][Z].mean().to_numpy()
    u=features[(features.artist_id==r.artist_id)&(features.start==r.start)].set_index('feature').loc[F,'unit_direction'].to_numpy()
    real=abs((p.loc[r.follower_id,Z].to_numpy(float)-source)@u)/np.sqrt(10)
    control=np.abs((p.loc[ids,Z].to_numpy()-source)@u).mean()/np.sqrt(10)
    assert np.isclose(real,r.source_projection_distance) and np.isclose(control,r.control_projection_distance)
    assert np.isclose(control-real,r.transition_distance_advantage)
evidence=read_csv(OUT/'r5/transition_evidence.csv')
for r in evidence.itertuples():
    q=tf[(tf.artist_id==r.artist_id)&(tf.start==r.start)]
    assert len(q)==r.matched
    if len(q)>=5:assert np.isclose(q.transition_distance_advantage.mean(),r.reported_transition_advantage)
    else:assert pd.isna(r.reported_transition_advantage)
checks['r5_transition_matching_reuses_controls_and_reconstructs_direction']=True
d=read_csv(OUT/'r6/all_decade_decomposition.csv')
keep=d.continuing_n.ge(10)
check('country_all_decomposition_identities',np.allclose(d.loc[keep,'total'],d.loc[keep,['within','exit_composition','entry_composition']].sum(axis=1)))
check('country_low_support_components_missing',d.loc[~keep,['within','exit_composition','entry_composition']].isna().all().all())
check('country_threshold_sensitivity_complete',set(d.minimum_credits)=={3,5,10} and len(d)==180)
dyn=read_csv(OUT/'r6/dynamic_indicators.csv')
check('country_dynamic_denominators',np.allclose(dyn.cohort_coverage,dyn.followers/dyn.all_country_nodes) and np.allclose(dyn.incoming_share,dyn.followers/dyn.all_incoming_edges))
for f in ['country_source_shares','country_follower_equal_sources','country_generation_gaps']:
    x=read_csv(OUT/'r7'/(f+'.csv'),index_col=0)
    check(f+'_sums_to_one',np.allclose(x.sum(axis=1),1))
v=read_csv(OUT/'r7/sonic_variance_decomposition.csv')
g=read_csv(OUT/'r7/sonic_genre_components.csv')
check('sonic_total_within_between_identity',np.allclose(v.total,v['within']+v.between))
for r in v.itertuples():
    x=g[g.period_start.eq(r.period_start)&g.population.eq(r.population)]
    assert np.isclose(x.weight.sum(),1)
    assert np.isclose((x.weight*x.within_variance).sum(),r.within)
    assert np.isclose((x.weight*x.center_squared_distance).sum(),r.between)
checks['sonic_genre_contributions_reconstructed']=True
pairs=read_csv(OUT/'r7/genre_pair_comparison.csv')
known=edges[edges.influencer_main_genre.ne('Unknown')&edges.follower_main_genre.ne('Unknown')]
check('cohort_stratified_expected_edge_conservation',np.isclose(pairs.cohort_stratified_expected.sum(),len(known)))
check('genre_pair_observed_edge_conservation',pairs.edges.sum()==len(known))
# Ten-feature presentation and the retrospective selection rationale.
trajectory=read_csv(OUT/'r6/genre_trajectories.csv').set_index('decade')
selection=read_csv(OUT/'r6/feature_selection_metrics.csv').set_index('feature').reindex(F)
check('r6_all_ten_features_present',set(selection.index)==set(F) and set(Z).issubset(trajectory.columns))
delta=trajectory.loc[2010,Z].to_numpy()-trajectory.loc[1950,Z].to_numpy()
step=trajectory[Z].diff().dropna().to_numpy()
check('r6_selection_metrics_reconstructed',np.allclose(selection.endpoint_change,delta) and np.allclose(selection['range'],trajectory[Z].max().to_numpy()-trajectory[Z].min().to_numpy()) and np.allclose(selection.path_efficiency,np.abs(delta)/np.abs(step).sum(axis=0)))
check('r6_primary_magnitude_ranking',set(selection.nsmallest(2,'endpoint_rank').index)=={'energy','acousticness'} and np.array_equal(selection.endpoint_rank,pd.Series(np.abs(delta)).rank(ascending=False,method='min').to_numpy()))
tracks=read_csv(OUT/'r6/leader_style_profiles.csv')
check('r6_each_leader_decade_has_ten_features',tracks.groupby(['artist_id','decade']).feature.nunique().eq(10).all() and tracks.artist_id.nunique()==5)
sens=read_csv(OUT/'r6/feature_selection_sensitivity.csv')
dec=read_csv(shared/'artist_decade.csv.gz')
c=dec[dec.genre.eq('Country')&dec.artist_id.isin(nodes.index[nodes.eligible])]
for threshold in [3,5,10]:
    means=c[c.n_songs.ge(threshold)].groupby('decade')[Z].mean()
    change=means.loc[2010]-means.loc[1950]
    q=sens[sens.minimum_credits.eq(threshold)].set_index('feature').reindex(F)
    assert np.allclose(q.endpoint_change,change.to_numpy())
    assert np.array_equal(q.endpoint_rank,change.abs().rank(ascending=False,method='min').to_numpy())
checks['r6_threshold_selection_sensitivity_reconstructed']=True
# Reconstruct the out-of-time prediction pipeline from its stored inputs and coefficients.
from scipy.special import expit
panel=read_csv(OUT/'r6/prediction_panel.csv')
loss=read_csv(OUT/'r6/prediction_fold_losses.csv')
coef=read_csv(OUT/'r6/prediction_coefficients.csv')
scores=read_csv(OUT/'r6/prediction_artist_scores.csv')
imp=read_csv(OUT/'r6/prediction_feature_importance.csv').set_index('feature')
settings=read_csv(OUT/'r6/prediction_importance_sensitivity.csv')
check('r6_prediction_time_and_audio_eligibility',panel.source_window_start.eq(panel.target_cohort-10).all() and panel.source_window_stop.eq(panel.target_cohort-1).all() and panel.source_cohort.lt(panel.target_cohort).all() and panel.n_songs.ge(5).all() and not panel.duplicated(['artist_id','target_cohort']).any())
country_edges=edges[edges.follower_main_genre.eq('Country')&edges.influencer_main_genre.eq('Country')&edges.influencer_active_start.lt(edges.follower_active_start)]
for r in panel.itertuples():
    e=country_edges[country_edges.influencer_id.eq(r.artist_id)]
    f=e[e.follower_active_start.eq(r.target_cohort)].follower_id.nunique()
    n=((nodes.genre=='Country')&(nodes.active_start==r.target_cohort)).sum()
    prior=e[e.follower_active_start.lt(r.target_cohort)].follower_id.nunique()
    a=dec[dec.artist_id.eq(r.artist_id)&dec.decade.eq(r.source_window_start)].iloc[0]
    assert f==r.followers and n==r.all_country_nodes and prior==r.prior_followers
    assert np.isclose(r.cohort_coverage,f/n) and np.allclose(a[Z].to_numpy(float),np.array([getattr(r,z) for z in Z]))
    assert np.isclose(r.log_prior_followers,np.log1p(prior)) and np.isclose(r.log_credits,np.log1p(a.n_songs))
    assert np.isclose(r.cohort_gap,(r.target_cohort-r.source_cohort)/10) and np.isclose(r.cohort_time,(r.target_cohort-1960)/10)
checks['r6_prediction_targets_prior_reach_and_audio_reconstructed']=True
primary=loss[loss.minimum_credits.eq(5)&loss.penalty.eq(10)]
for (cohort,model),cfs in coef.groupby(['target_cohort','model']):
    train=panel[panel.target_cohort.lt(cohort)];test=panel[panel.target_cohort.eq(cohort)]
    cfs=cfs.set_index('term');columns=cfs.index.drop('intercept').tolist()
    mean=train[columns].mean();sd=train[columns].std(ddof=0);sd=sd.where(sd>1e-12,1.)
    assert np.allclose(cfs.loc[columns,'training_mean'],mean) and np.allclose(cfs.loc[columns,'training_sd'],sd)
    prediction=expit(cfs.at['intercept','coefficient']+((test[columns]-mean)/sd).to_numpy()@cfs.loc[columns,'coefficient'].to_numpy())
    recorded=primary[primary.target_cohort.eq(cohort)&primary.model.eq(model)].iloc[0]
    assert recorded.train_rows==len(train) and recorded.test_rows==len(test)
    assert np.isclose(recorded.mse,np.mean((prediction-test.cohort_coverage.to_numpy())**2))
    if model in ['baseline','full']:
        q=scores[scores.target_cohort.eq(cohort)&scores.model.eq(model)].set_index('artist_id').loc[test.artist_id]
        assert np.allclose(q.predicted_coverage,prediction)
checks['r6_prediction_training_only_scaling_and_losses']=True
for (minimum,penalty),group in loss.groupby(['minimum_credits','penalty']):
    w=group.pivot(index='target_cohort',columns='model',values='mse');base=w.baseline.mean()
    for f in F:
        a=w.baseline-w['add_'+f];d=w['drop_'+f]-w.full
        r=settings[settings.minimum_credits.eq(minimum)&settings.penalty.eq(penalty)&settings.feature.eq(f)].iloc[0]
        assert np.isclose(r.addition_gain,a.mean()/base) and np.isclose(r.conditional_drop_gain,d.mean()/base)
        assert r.addition_positive_folds==a.gt(0).sum() and r.drop_positive_folds==d.gt(0).sum()
        if minimum==5 and penalty==10:
            assert np.isclose(imp.at[f,'addition_gain'],r.addition_gain) and np.isclose(imp.at[f,'conditional_drop_gain'],r.conditional_drop_gain)
            positive=settings[settings.feature.eq(f)].addition_gain.gt(0).sum()
            stable=(r.addition_gain>0 and r.addition_positive_folds>=3 and positive==5)
            assert imp.at[f,'positive_settings']==positive and imp.at[f,'stable_candidate']==stable
            assert imp.at[f,'conditional_supported']==(stable and r.conditional_drop_gain>0 and r.drop_positive_folds>=3)
check('r6_prediction_all_ten_importance_and_gates',set(imp.index)==set(F) and np.array_equal(imp.importance_rank,imp.addition_gain.rank(ascending=False,method='min')))
contrib=read_csv(OUT/'r6/prediction_contributions.csv')
for (cohort,artist),q in contrib.groupby(['target_cohort','artist_id']):
    assert len(q)==14 and set(Z).issubset(q.term)
    assert np.allclose(q.log_odds_contribution,(q.raw_value-q.training_mean)/q.training_sd*q.coefficient)
    prediction=expit(q.intercept.iloc[0]+q.log_odds_contribution.sum())
    r=scores[scores.target_cohort.eq(cohort)&scores.artist_id.eq(artist)&scores.model.eq('full')].iloc[0]
    assert np.isclose(prediction,r.predicted_coverage)
checks['r6_prediction_each_artist_contribution_reconstructed']=True
support=read_csv(OUT/'r6/prediction_cohort_support.csv')
for r in support.itertuples():
    q=panel[panel.target_cohort.eq(r.target_cohort)]
    assert r.audio_supported_sources==len(q) and r.sources_with_positive_reach==q.followers.gt(0).sum()
ledger=read_csv(OUT/'r6/prediction_leader_eligibility.csv')
for r in ledger.itertuples():
    a=dec[dec.artist_id.eq(r.artist_id)&dec.decade.eq(r.source_window_start)]
    assert r.prior_credits==(a.n_songs.iloc[0] if len(a) else 0)
    assert r.model_eligible==bool(((panel.artist_id==r.artist_id)&(panel.target_cohort==r.target_cohort)).any())
checks['r6_prediction_support_and_missing_leader_ledger']=True
(OUT/'validation.json').write_text(json.dumps({'status':'passed','checks':checks,'check_count':len(checks),'matching_note':'Each stored control ID checked; no significance tests inferred.'},indent=2))
print(f'PASS: {len(checks)} named checks, including all stored matched contrasts.')
