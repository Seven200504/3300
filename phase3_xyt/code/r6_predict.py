from csv_store import read_csv
"""Time-ordered retrospective prediction of next-cohort Country network coverage.

All eligible sources, not just observed leaders, enter the model. Feature importance
is held-out predictive gain, never an estimate of a causal sonic influence effect.
"""
from common import *
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
import json
DEST=ROOT/'output/r6'
COHORTS=list(range(1960,2020,10))
TEST_COHORTS=[1980,1990,2000,2010]
BASE_COLUMNS=['log_prior_followers','cohort_gap','log_credits','cohort_time']
MAIN_PENALTY=10.
PENALTIES=[1.,10.,100.]
dec=read_csv(shared/'artist_decade.csv.gz')
dynamic=read_csv(DEST/'dynamic_indicators.csv')
country_edges=edges[edges.follower_main_genre.eq('Country')&edges.influencer_main_genre.eq('Country')&edges.influencer_active_start.lt(edges.follower_active_start)]


def make_panel(minimum):
    x=dynamic[dynamic.target_cohort.isin(COHORTS)].copy()
    x['decade']=x.target_cohort-10
    x=x.merge(dec,on=['artist_id','decade'],validate='one_to_one',suffixes=('','_audio'))
    x=x[x.n_songs.ge(minimum)&x.artist_id.isin(nodes.index[nodes.eligible])].copy()
    prior=[]
    for r in x.itertuples():
        prior.append(country_edges[country_edges.influencer_id.eq(r.artist_id)&country_edges.follower_active_start.lt(r.target_cohort)].follower_id.nunique())
    x['prior_followers']=prior
    x['log_prior_followers']=np.log1p(x.prior_followers)
    x['cohort_gap']=(x.target_cohort-x.source_cohort)/10
    x['log_credits']=np.log1p(x.n_songs)
    x['cohort_time']=(x.target_cohort-1960)/10
    x['source_window_start']=x.decade
    x['source_window_stop']=x.decade+9
    return x.sort_values(['target_cohort','artist_id']).reset_index(drop=True)


def fit_predict(train,test,columns,penalty):
    # Fit scaling only on earlier cohorts; targets, controls and all ten features remain distinct.
    mean=train[columns].mean().to_numpy();sd=train[columns].std(ddof=0).to_numpy()
    sd=np.where(sd>1e-12,sd,1.)
    a=np.column_stack([np.ones(len(train)),(train[columns].to_numpy()-mean)/sd])
    b=np.column_stack([np.ones(len(test)),(test[columns].to_numpy()-mean)/sd])
    successes=train.followers.to_numpy(float);trials=train.all_country_nodes.to_numpy(float)
    def objective(beta):
        eta=a@beta
        loss=np.sum(trials*np.logaddexp(0,eta)-successes*eta)+.5*penalty*np.dot(beta[1:],beta[1:])
        gradient=a.T@(trials*expit(eta)-successes)
        gradient[1:]+=penalty*beta[1:]
        return loss,gradient
    initial=np.zeros(a.shape[1]);rate=successes.sum()/trials.sum()
    initial[0]=np.log(rate/(1-rate))
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-12,'gtol':1e-7})
    if not result.success:raise RuntimeError(result.message)
    prediction=expit(b@result.x)
    return prediction,result.x,mean,sd


panel=make_panel(5)
panel[['artist_id','name','source_cohort','target_cohort','source_window_start','source_window_stop','n_songs','followers','all_country_nodes','cohort_coverage','prior_followers',*BASE_COLUMNS,*Z]].to_csv(DEST/'prediction_panel.csv',index=False)
models={'baseline':BASE_COLUMNS,'full':BASE_COLUMNS+Z}
for f,z in zip(F,Z):
    models['add_'+f]=BASE_COLUMNS+[z]
    models['drop_'+f]=BASE_COLUMNS+[v for v in Z if v!=z]
losses=[];predictions=[];coefficients=[];contributions=[]
# Penalty and coverage sensitivities are reported, not tuned against future test outcomes.
settings=[(5,p) for p in PENALTIES]+[(3,MAIN_PENALTY),(10,MAIN_PENALTY)]
for minimum,penalty in settings:
    data=panel if minimum==5 else make_panel(minimum)
    for cohort in TEST_COHORTS:
        train=data[data.target_cohort.lt(cohort)];test=data[data.target_cohort.eq(cohort)]
        assert train.target_cohort.max()<cohort and train.source_window_stop.max()<cohort
        y=test.cohort_coverage.to_numpy()
        for model,columns in models.items():
            pred,beta,mean,sd=fit_predict(train,test,columns,penalty)
            mse=float(np.mean((pred-y)**2))
            ce=float(np.mean(np.logaddexp(0,np.log(pred/(1-pred)))-y*np.log(pred/(1-pred))))
            losses.append([minimum,penalty,cohort,model,len(train),len(test),mse,np.sqrt(mse),ce])
            if minimum==5 and penalty==MAIN_PENALTY:
                coefficients.extend([[cohort,model,'intercept',beta[0],0.,1.]]+[[cohort,model,c,v,m,s] for c,v,m,s in zip(columns,beta[1:],mean,sd)])
                if model in ['baseline','full']:
                    predictions.extend([[cohort,int(r.artist_id),r['name'],model,int(r.source_window_start),int(r.source_window_stop),int(r.n_songs),int(r.prior_followers),int(r.followers),int(r.all_country_nodes),r.cohort_coverage,float(p)] for (_,r),p in zip(test.iterrows(),pred)])
                if model=='full':
                    standardized=(test[columns].to_numpy()-mean)/sd
                    for k,(_,r) in enumerate(test.iterrows()):
                        contributions.extend([[cohort,int(r.artist_id),r['name'],c,float(r[c]),float(mean[j]),float(sd[j]),float(beta[j+1]),float(standardized[k,j]*beta[j+1]),float(beta[0])] for j,c in enumerate(columns)])
loss=pd.DataFrame(losses,columns=['minimum_credits','penalty','target_cohort','model','train_rows','test_rows','mse','rmse','cross_entropy'])
loss.to_csv(DEST/'prediction_fold_losses.csv',index=False)
coef=pd.DataFrame(coefficients,columns=['target_cohort','model','term','coefficient','training_mean','training_sd'])
coef.to_csv(DEST/'prediction_coefficients.csv',index=False)
pred=pd.DataFrame(predictions,columns=['target_cohort','artist_id','name','model','source_window_start','source_window_stop','credits','prior_followers','followers','target_nodes','actual_coverage','predicted_coverage'])
pred['actual_rank']=pred.groupby(['target_cohort','model']).actual_coverage.rank(method='min',ascending=False)
pred['predicted_rank']=pred.groupby(['target_cohort','model']).predicted_coverage.rank(method='min',ascending=False)
pred.to_csv(DEST/'prediction_artist_scores.csv',index=False)
contrib=pd.DataFrame(contributions,columns=['target_cohort','artist_id','name','term','raw_value','training_mean','training_sd','coefficient','log_odds_contribution','intercept'])
contrib.to_csv(DEST/'prediction_contributions.csv',index=False)

# Equal weight per held-out cohort, then relative gain against the baseline loss.
importance=[];fold_importance=[]
for (minimum,penalty),group in loss.groupby(['minimum_credits','penalty']):
    wide=group.pivot(index='target_cohort',columns='model',values='mse')
    base=wide.baseline.mean();full=wide.full.mean()
    for f,z in zip(F,Z):
        add=wide.baseline-wide['add_'+f];drop=wide['drop_'+f]-wide.full
        for cohort in TEST_COHORTS:
            fold_importance.append([minimum,penalty,cohort,f,float(add[cohort]),float(drop[cohort])])
        importance.append([minimum,penalty,f,float(add.mean()/base),float(drop.mean()/base),int(add.gt(0).sum()),int(drop.gt(0).sum()),float((base-full)/base)])
imp=pd.DataFrame(importance,columns=['minimum_credits','penalty','feature','addition_gain','conditional_drop_gain','addition_positive_folds','drop_positive_folds','full_model_gain'])
imp.to_csv(DEST/'prediction_importance_sensitivity.csv',index=False)
pd.DataFrame(fold_importance,columns=['minimum_credits','penalty','target_cohort','feature','addition_mse_gain','conditional_drop_mse_gain']).to_csv(DEST/'prediction_feature_fold_gains.csv',index=False)
main=imp[imp.minimum_credits.eq(5)&imp.penalty.eq(MAIN_PENALTY)].copy().set_index('feature')
for f,z in zip(F,Z):
    ss=imp[imp.feature.eq(f)]
    main.loc[f,'positive_settings']=int(ss.addition_gain.gt(0).sum())
    b=coef[coef.model.eq('add_'+f)&coef.term.eq(z)].coefficient
    main.loc[f,'positive_coefficients']=int(b.gt(0).sum())
    main.loc[f,'negative_coefficients']=int(b.lt(0).sum())
# Explicit reporting rule, not a significance threshold: gain >0, >=3/4 folds, all five settings >0.
main['stable_candidate']=main.addition_gain.gt(0)&main.addition_positive_folds.ge(3)&main.positive_settings.eq(len(settings))
main['importance_rank']=main.addition_gain.rank(ascending=False,method='min').astype(int)
main=main.sort_values('importance_rank');main.to_csv(DEST/'prediction_feature_importance.csv')

performance=[];cases=[]
leaders=read_csv(DEST/'cohort_leaders.csv')
for cohort in TEST_COHORTS:
    for model in ['baseline','full']:
        q=pred[pred.target_cohort.eq(cohort)&pred.model.eq(model)]
        # Fixed top-3 with artist-ID tie break; scores remain continuous in the main output.
        actual=set(q.sort_values(['actual_coverage','artist_id'],ascending=[False,True]).head(3).artist_id)
        estimated=set(q.sort_values(['predicted_coverage','artist_id'],ascending=[False,True]).head(3).artist_id)
        performance.append([cohort,model,len(q),float(np.sqrt(np.mean((q.actual_coverage-q.predicted_coverage)**2))),float(spearmanr(q.actual_coverage,q.predicted_coverage).statistic),len(actual&estimated)])
    leader=leaders[leaders.target_cohort.eq(cohort)].iloc[0]
    q=pred[pred.target_cohort.eq(cohort)&pred.model.eq('full')]
    p=q[q.artist_id.eq(leader.artist_id)]
    if not p.empty:cases.append(p.iloc[0].to_dict())
perf=pd.DataFrame(performance,columns=['target_cohort','model','test_sources','coverage_rmse','rank_spearman','top3_overlap'])
perf.to_csv(DEST/'prediction_performance.csv',index=False)
pd.DataFrame(cases).to_csv(DEST/'prediction_leader_examples.csv',index=False)

fig,axs=plt.subplots(1,2,figsize=(13,7),layout='constrained')
y=np.arange(len(main));w=.35
axs[0].barh(y-w/2,100*main.addition_gain,w,label='Add one feature to baseline',color='#137f88')
axs[0].barh(y+w/2,100*main.conditional_drop_gain,w,label='Remove one from all-10 model',color='#7654a3')
axs[0].set_yticks(y,main.index);axs[0].invert_yaxis();axs[0].axvline(0,color='gray',lw=.8)
axs[0].set(xlabel='Gain / baseline MSE (%)',title='Feature importance on held-out future cohorts');axs[0].legend(fontsize=8)
for model,c,label in [('baseline','#526b83','Network / opportunity baseline'),('full','#d66139','Baseline + all 10 audio features')]:
    q=perf[perf.model.eq(model)];axs[1].plot(q.target_cohort,100*q.coverage_rmse,'o-',label=label,color=c)
axs[1].set(xticks=TEST_COHORTS,xlabel='Held-out target career-start cohort',ylabel='Coverage RMSE (percentage points)',title='Does audio improve the whole model?');axs[1].legend(fontsize=8)
fig.suptitle('R6 | Earlier catalogue profiles predicting next-cohort recorded coverage',fontsize=14)
export(fig,DEST,'08_predictive_importance')

fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained',gridspec_kw={'width_ratios':[1,1.4]})
case=pd.DataFrame(cases)
y=np.arange(len(case));width=.25
basecase=pred[pred.model.eq('baseline')].set_index(['target_cohort','artist_id'])
axs[0].barh(y-width,100*case.actual_coverage,width,label='Observed coverage',color='#526b83')
axs[0].barh(y,100*np.array([basecase.at[(r.target_cohort,r.artist_id),'predicted_coverage'] for r in case.itertuples()]),width,label='Baseline prediction',color='#b5c5ce')
axs[0].barh(y+width,100*case.predicted_coverage,width,label='All-10 prediction',color='#137f88')
axs[0].set_yticks(y,[f'{r.name} → {r.target_cohort}s' for r in case.itertuples()]);axs[0].invert_yaxis();axs[0].set(xlabel='Country target nodes reached (%)',title='Observed leaders: transparent out-of-time examples');axs[0].legend(fontsize=8)
mat=np.array([contrib[contrib.target_cohort.eq(r.target_cohort)&contrib.artist_id.eq(r.artist_id)].set_index('term').loc[Z,'log_odds_contribution'].to_numpy() for r in case.itertuples()])
v=max(abs(mat).max(),.1);im=axs[1].imshow(mat,cmap='RdBu',vmin=-v,vmax=v,aspect='auto')
axs[1].set_xticks(range(10),F,rotation=50,ha='right',fontsize=8);axs[1].set_yticks(range(len(case)),case.target_cohort.astype(str));axs[1].set_title('Ten-feature contributions to predicted log odds',fontsize=11)
for row in range(len(case)):
    for col in range(10):axs[1].text(col,row,f'{mat[row,col]:+.2f}',ha='center',va='center',fontsize=7,color='white' if abs(mat[row,col])>v*.6 else 'black')
fig.colorbar(im,ax=axs[1],label='Coefficient × training-standardized feature')
export(fig,DEST,'09_prediction_examples')

main_loss=loss[loss.minimum_credits.eq(5)&loss.penalty.eq(MAIN_PENALTY)]
info={'primary_penalty':MAIN_PENALTY,'test_cohorts':TEST_COHORTS,'eligible_panel_rows':len(panel),'unique_sources':int(panel.artist_id.nunique()),'panel_support':panel.groupby('target_cohort').size().to_dict(),'stable_candidates':main.index[main.stable_candidate].tolist(),'full_model_relative_gain':float(main.full_model_gain.iloc[0]),'baseline_cohort_equal_rmse':float(np.sqrt(main_loss[main_loss.model.eq('baseline')].mse.mean())),'full_cohort_equal_rmse':float(np.sqrt(main_loss[main_loss.model.eq('full')].mse.mean())),'criterion':'positive addition gain, >=3/4 positive held-out cohorts, positive aggregate gain in all five penalty/credit settings','baseline_columns':BASE_COLUMNS}
(DEST/'prediction_summary.json').write_text(json.dumps(info,indent=2))
assert len(pred[pred.model.eq('full')])==panel.target_cohort.isin(TEST_COHORTS).sum()
assert pred.predicted_coverage.between(0,1).all()
print(json.dumps(info,indent=2));print(main.to_string());print(perf.to_string(index=False));print(pd.DataFrame(cases).to_string(index=False))

# Explicit missing-data ledger: a leader without >=5 prior-decade credits is not silently omitted.
support_rows=[];leader_eligibility=[]
for cohort in COHORTS:
    all_sources=dynamic[dynamic.target_cohort.eq(cohort)]
    usable=panel[panel.target_cohort.eq(cohort)]
    support_rows.append([cohort,len(all_sources),len(usable),usable.followers.gt(0).sum(),int(all_sources.all_country_nodes.iloc[0])])
for cohort in TEST_COHORTS:
    r=leaders[leaders.target_cohort.eq(cohort)].iloc[0]
    audio=dec[dec.artist_id.eq(r.artist_id)&dec.decade.eq(cohort-10)]
    credits=int(audio.n_songs.iloc[0]) if len(audio) else 0
    usable=panel[panel.artist_id.eq(r.artist_id)&panel.target_cohort.eq(cohort)]
    leader_eligibility.append([cohort,int(r.artist_id),r['name'],cohort-10,cohort-1,credits,bool(len(usable)),'' if len(usable) else 'fewer than 5 prior-decade credits'])
pd.DataFrame(support_rows,columns=['target_cohort','eligible_network_sources','audio_supported_sources','sources_with_positive_reach','target_nodes']).to_csv(DEST/'prediction_cohort_support.csv',index=False)
pd.DataFrame(leader_eligibility,columns=['target_cohort','artist_id','name','source_window_start','source_window_stop','prior_credits','model_eligible','exclusion_reason']).to_csv(DEST/'prediction_leader_eligibility.csv',index=False)
main['conditional_supported']=main.stable_candidate&main.conditional_drop_gain.gt(0)&main.drop_positive_folds.ge(3)
main.to_csv(DEST/'prediction_feature_importance.csv')
info['conditional_supported_candidates']=main.index[main.conditional_supported].tolist()
info['conditional_rule']='stable single-addition candidate, positive drop-one gain, >=3/4 positive drop-one held-out cohorts'
(DEST/'prediction_summary.json').write_text(json.dumps(info,indent=2))
