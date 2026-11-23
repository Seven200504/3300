from csv_store import read_csv
"""R6 cohort indicators, recording trajectories, and exact sample decomposition."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mat3300-mpl')
import matplotlib
matplotlib.use('Agg')
from common import *

DEST=ROOT/'output/r6';DEST.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,'grid.alpha':.18,'axes.titleweight':'bold'})
periods=list(range(1950,2020,10))
colors=['#137f88','#d66139','#7654a3','#b58a17','#3267a8']
country_nodes=nodes[nodes.genre.eq('Country')]
e=edges[edges.follower_main_genre.eq('Country')].copy()
assert not e.duplicated(['influencer_id','follower_id']).any() and len(e)==3248
incoming=e.groupby('follower_active_start').size()
cohort_n=country_nodes.groupby('active_start').size()
linked=e.groupby('follower_active_start').follower_id.nunique()
earlier=e[e.influencer_main_genre.eq('Country')&e.influencer_active_start.lt(e.follower_active_start)]
counts=earlier.groupby(['influencer_id','follower_active_start']).follower_id.nunique()
# Zero is an observed absence only once the source is eligible as an earlier source.
rows=[]
for i,source in country_nodes.iterrows():
    for b in periods:
        if source.active_start>=b:continue
        f=counts.get((i,b),0)
        rows.append([i,source.artist_name,int(source.active_start),b,f,int(incoming[b]),int(cohort_n[b]),int(linked[b]),f/incoming[b],f/cohort_n[b]])
dyn=pd.DataFrame(rows,columns=['artist_id','name','source_cohort','target_cohort','followers','all_incoming_edges','all_country_nodes','targets_with_incoming','incoming_share','cohort_coverage'])
dyn['rank']=dyn.groupby('target_cohort').followers.rank(method='min',ascending=False)
dyn['share_change_pp']=dyn.groupby('artist_id').incoming_share.diff()*100
dyn['coverage_change_pp']=dyn.groupby('artist_id').cohort_coverage.diff()*100
dyn.to_csv(DEST/'dynamic_indicators.csv',index=False)
leaders=dyn.loc[dyn.groupby('target_cohort').followers.idxmax()].sort_values('target_cohort')
leaders.to_csv(DEST/'cohort_leaders.csv',index=False)
ids=leaders.artist_id.drop_duplicates().tolist()
assert ids==[549797,410219,214162,944084,68551]
cohort=pd.DataFrame({'all_country_nodes':cohort_n,'targets_with_incoming':linked,'all_incoming_edges':incoming})
cohort['external_share']=e[e.influencer_main_genre.ne('Country')].groupby('follower_active_start').size()/incoming
cohort.to_csv(DEST/'cohort_support.csv',index_label='target_cohort')
persist=dyn[dyn.artist_id.isin(ids)].groupby(['artist_id','name']).agg(eligible_cohorts=('target_cohort','size'),cohorts_with_followers=('followers',lambda s:int(s.gt(0).sum())),top3_cohorts=('rank',lambda s:int(s.le(3).sum())))
persist.to_csv(DEST/'leader_persistence.csv')

def export(fig,name):
    fig.savefig(DEST/(name+'.png'),dpi=170,bbox_inches='tight')
    plt.close(fig)

def decade_axis(a,label):
    a.set_xticks(periods,[f'{p}s' for p in periods]);a.set_xlabel(label);a.set_xlim(1947,2013)

# Style time axis is recording decade, not follower career cohort.
dec=read_csv(shared/'artist_decade.csv.gz')
country=dec[dec.artist_id.isin(nodes.index[nodes.eligible])&dec.genre.eq('Country')&dec.n_songs.ge(5)&dec.decade.isin(periods)].copy()
g=country.groupby('decade')[Z].mean();gn=country.groupby('decade').size()
g.join(gn.rename('artists')).to_csv(DEST/'genre_trajectories.csv')
fig,axs=plt.subplots(5,2,figsize=(13,14),layout='constrained',sharex=True,sharey=True)
for ax,f in zip(axs.flat,F):
    values=g['z_'+f]
    col='#137f88' if f in ['energy','acousticness'] else '#526b83'
    ax.plot(periods,values,'o-',color=col,lw=2)
    ax.axhline(0,color='gray',lw=.6);ax.axvspan(1990,2000,color='#ead69c',alpha=.25)
    ax.set_title(f'{f} | endpoint change {values.iloc[-1]-values.iloc[0]:+.3f} SD',loc='left',fontsize=11)
    ax.set_ylabel('Reference SD');ax.set_xticks(periods,[str(t) for t in periods],rotation=25)
for ax in axs[-1]:ax.set_xlabel('Recording decade')
fig.suptitle('Country | All 10 features on the same vertical scale\nArtists by decade: '+', '.join(f'{p}s: {gn[p]}' for p in periods),fontsize=14)
export(fig,'01_genre_trends')

fig,axs=plt.subplots(3,1,figsize=(13,11),layout='constrained',sharex=True,gridspec_kw={'height_ratios':[1,1,.65]})
for i,c in zip(ids,colors):
    s=dyn[dyn.artist_id.eq(i)].set_index('target_cohort').reindex(periods)
    axs[0].plot(periods,s.incoming_share*100,'o-',color=c,label=nodes.at[i,'artist_name'],lw=2)
    axs[1].plot(periods,s.cohort_coverage*100,'o-',color=c,lw=2)
axs[0].set_title('Dynamic influencers | Which earlier Country sources reach each new cohort?',loc='left',fontsize=15)
axs[0].set_ylabel('Share of ALL incoming edges (%)');axs[0].legend(ncol=3,fontsize=9)
axs[0].text(.01,.02,'Missing before eligibility ≠ zero. Lines track follower cohorts, not influence-event dates.',transform=axs[0].transAxes,fontsize=9)
axs[0].set_ylim(-.4,10)
axs[1].set_ylabel('Share of ALL Country nodes\nin target cohort reached (%)');axs[1].set_ylim(-1,60)
axs[2].plot(periods,cohort.loc[periods,'all_country_nodes'],'o-',color='#35495c',label='Country nodes')
axs[2].plot(periods,cohort.loc[periods,'targets_with_incoming'],'s--',color='#9aaeb7',label='Nodes with incoming edges')
for p in periods:axs[2].text(p,cohort.at[p,'all_country_nodes']+4,f'n={int(cohort.at[p,"all_country_nodes"])}',ha='center',fontsize=8)
axs[2].set_ylim(0,95);axs[2].set_ylabel('Target nodes');axs[2].legend(fontsize=8,loc='lower left')
decade_axis(axs[2],'Follower career-start cohort')
export(fig,'02_dynamic_influencers')

# Each network leader receives all ten feature trajectories, preserving missing decades.
track=[]
for i,c in zip(ids,colors):
    personal=country[country.artist_id.eq(i)].set_index('decade')
    peers=country[country.artist_id.ne(i)].groupby('decade')
    means=peers[Z].mean();pn=peers.size()
    for p,r in personal.iterrows():
        for f in F:
            track.append([i,nodes.at[i,'artist_name'],p,int(r.n_songs),f,r['z_'+f],means.at[p,'z_'+f],r['z_'+f]-means.at[p,'z_'+f],int(pn[p])])
    fig,axs=plt.subplots(5,2,figsize=(13,14),layout='constrained',sharex=True)
    for ax,f in zip(axs.flat,F):
        ax.plot(periods,means['z_'+f].reindex(periods),'--',color='#95a6b1',label='Other Country artists')
        values=personal['z_'+f].reindex(periods)
        ax.plot(periods,values,'o-',color=c,lw=2,label='Artist catalogue')
        ax.axhline(0,lw=.6,color='gray');ax.set_title(f,loc='left');ax.set_ylabel('Reference SD')
        ax.set_xticks(periods,[str(t) for t in periods],rotation=25)
    axs[0,0].legend(fontsize=8)
    for ax in axs[-1]:ax.set_xlabel('Recording decade')
    counts_text=', '.join(f'{int(p)}s: {int(r.n_songs)}' for p,r in personal.iterrows())
    fig.suptitle(nodes.at[i,'artist_name']+' | All 10 features\nCredits per supported decade: '+counts_text,fontsize=13)
    export(fig,f'03_leader_{i}')
tracks=pd.DataFrame(track,columns=['artist_id','name','decade','credits','feature','artist_mean','other_country_mean','gap','other_artists'])
tracks.to_csv(DEST/'leader_style_profiles.csv',index=False)

# Exact descriptive decomposition using continuing artists as the reference group.
a=country[country.decade.eq(1990)].set_index('artist_id');b=country[country.decade.eq(2000)].set_index('artist_id')
s=a.index.intersection(b.index);left=a.index.difference(b.index);entered=b.index.difference(a.index)
parts=[];paired=[]
fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
for row,f in enumerate(['energy','acousticness']):
    z='z_'+f
    within=b.loc[s,z].mean()-a.loc[s,z].mean()
    exit_term=len(left)/len(a)*(a.loc[s,z].mean()-a.loc[left,z].mean())
    entry_term=len(entered)/len(b)*(b.loc[entered,z].mean()-b.loc[s,z].mean())
    total=b[z].mean()-a[z].mean()
    assert np.isclose(within+exit_term+entry_term,total)
    parts.append([f,len(a),len(b),len(s),len(left),len(entered),a[z].mean(),b[z].mean(),within,exit_term,entry_term,total])
    delta=b.loc[s,z]-a.loc[s,z]
    for i,d in delta.items():paired.append([i,nodes.at[i,'artist_name'],f,a.at[i,z],b.at[i,z],d])
    vals=[within,exit_term,entry_term,total];starts=[0,within,within+exit_term,0]
    ax=axs[row,0]
    for j,(v,base) in enumerate(zip(vals,starts)):
        ax.bar(j,v,bottom=base,width=.6,color=('#35495c' if j==3 else '#137f88' if v>=0 else '#d66139'))
        ax.text(j,base+v/2,f'{v:+.3f}',ha='center',va='center',color='white',fontweight='bold',fontsize=11)
    ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(4),['Continuing\nartists','Leaving-sample\ncomposition','Entering-sample\ncomposition','Observed\ntotal']);ax.set_title(f'{f} | Exact mean-change decomposition',loc='left');ax.set_ylabel('Reference SD')
    ax=axs[row,1]
    for i in s:ax.plot([1990,2000],[a.at[i,z],b.at[i,z]],color='#137f88' if delta[i]>=0 else '#d66139',alpha=.4,lw=1)
    ax.plot([1990,2000],[a.loc[s,z].mean(),b.loc[s,z].mean()],'o-',lw=3,color='#263947',label='Same 34 artists: mean')
    ax.set_xticks([1990,2000],['1990s','2000s']);ax.set_title(f'{f} | {int(delta.gt(0).sum())}/34 artists increase',loc='left');ax.set_ylabel('Reference SD');ax.legend(fontsize=9)
fig.suptitle('What changed in Country, 1990s → 2000s?\n89 artists → 70 artists: 34 continue, 55 leave the comparable sample, 36 enter',fontsize=16)
export(fig,'04_change_decomposition')
parts=pd.DataFrame(parts,columns=['feature','before_n','after_n','continuing_n','left_n','entered_n','before_mean','after_mean','within_change','exit_composition','entry_composition','total_change'])
print('LEADERS\n',leaders.to_string(index=False));print('DECOMPOSITION\n',parts.to_string(index=False));print('PERSISTENCE\n',persist.to_string());print('All assertions passed.')


# Every adjacent supported decade, all ten features, with 3/5/10-credit sensitivity.
all_country=dec[dec.artist_id.isin(nodes.index[nodes.eligible])&dec.genre.eq('Country')&dec.decade.isin(periods)].copy()
decomposed=[];individual=[]
rng=np.random.default_rng(33006)
for minimum in [3,5,10]:
    c=all_country[all_country.n_songs.ge(minimum)]
    for before,after in zip(periods[:-1],periods[1:]):
        a=c[c.decade.eq(before)].set_index('artist_id');b=c[c.decade.eq(after)].set_index('artist_id')
        s=a.index.intersection(b.index);left=a.index.difference(b.index);entered=b.index.difference(a.index)
        for f,z in zip(F,Z):
            total=b[z].mean()-a[z].mean()
            within=exit_term=entry_term=lo=hi=np.nan
            if len(s)>=10:
                d=(b.loc[s,z]-a.loc[s,z]).to_numpy()
                within=d.mean()
                exit_term=len(left)/len(a)*(a.loc[s,z].mean()-a.loc[left,z].mean()) if len(left) else 0.
                entry_term=len(entered)/len(b)*(b.loc[entered,z].mean()-b.loc[s,z].mean()) if len(entered) else 0.
                boot=rng.choice(d,size=(1000,len(d)),replace=True).mean(axis=1)
                lo,hi=np.quantile(boot,[.025,.975])
                assert np.isclose(within+exit_term+entry_term,total)
                if minimum==5:
                    individual.extend([[before,after,int(i),f,float(a.at[i,z]),float(b.at[i,z]),float(dv)] for i,dv in zip(s,d)])
            decomposed.append([before,after,minimum,f,len(a),len(b),len(s),len(left),len(entered),total,within,exit_term,entry_term,lo,hi])
decomposition=pd.DataFrame(decomposed,columns=['before','after','minimum_credits','feature','before_n','after_n','continuing_n','left_n','entered_n','total','within','exit_composition','entry_composition','within_boot_low','within_boot_high'])
decomposition.to_csv(DEST/'all_decade_decomposition.csv',index=False)
pd.DataFrame(individual,columns=['before_decade','after_decade','artist_id','feature','before','after','change']).to_csv(DEST/'all_continuing_changes.csv',index=False)
fig,axs=plt.subplots(5,2,figsize=(14,15),layout='constrained',sharex=True,sharey=True)
for ax,f in zip(axs.flat,F):
    q=decomposition[decomposition.minimum_credits.eq(5)&decomposition.feature.eq(f)]
    x=np.arange(len(q));width=.22
    for offset,key,label,color in [(-1,'within','Continuing group','#137f88'),(0,'exit_composition','Leaving-sample composition','#d66139'),(1,'entry_composition','Entering-sample composition','#7654a3')]:
        ax.bar(x+offset*width,q[key],width,label=label,color=color)
    ax.plot(x,q.total,'ko--',lw=1,label='Observed total')
    ax.axhline(0,color='gray',lw=.7);ax.set_title(f,loc='left');ax.set_ylabel('Change (SD)')
    ax.set_xticks(x,[f'{r.before}→{r.after}' for r in q.itertuples()],rotation=30,fontsize=8)
axs[0,0].legend(fontsize=7,ncol=2)
for ax in axs[-1]:ax.set_xlabel('Recording-decade transition')
fig.suptitle('Country | All 10 mean-change decompositions on the same scale\nContinuing artists per transition: 16, 20, 22, 30, 34, 29',fontsize=14)
export(fig,'05_all_decade_decomposition')

# Retrospective, explicit rationale for two narrative examples; not a pre-registered screen.
# Rank absolute endpoint change in the fixed Phase 2 units. Other metrics remain visible.
metrics=[];sensitivity=[]
for minimum in [3,5,10]:
    sample=all_country[all_country.n_songs.ge(minimum)]
    gm=sample.groupby('decade')[Z].mean().reindex(periods)
    delta=gm.iloc[-1]-gm.iloc[0]
    ranks=delta.abs().rank(ascending=False,method='min')
    for f,z in zip(F,Z):
        sensitivity.append([minimum,f,delta[z],int(ranks[z]),int(sample[sample.decade.eq(1950)].shape[0]),int(sample[sample.decade.eq(2010)].shape[0])])
        if minimum==5:
            steps=gm[z].diff().dropna();path=steps.abs().sum()
            metrics.append([f,gm[z].iloc[0],gm[z].iloc[-1],delta[z],abs(delta[z]),int(ranks[z]),gm[z].max()-gm[z].min(),int((np.sign(steps)==np.sign(delta[z])).sum()),abs(delta[z])/path if path else np.nan,f in ['energy','acousticness']])
metrics=pd.DataFrame(metrics,columns=['feature','mean_1950','mean_2010','endpoint_change','abs_endpoint_change','endpoint_rank','range','same_direction_steps','path_efficiency','detailed_example'])
sens=pd.DataFrame(sensitivity,columns=['minimum_credits','feature','endpoint_change','endpoint_rank','artists_1950','artists_2010'])
metrics.to_csv(DEST/'feature_selection_metrics.csv',index=False)
sens.to_csv(DEST/'feature_selection_sensitivity.csv',index=False)
assert set(metrics.nsmallest(2,'endpoint_rank').feature)=={'energy','acousticness'}
fig,axs=plt.subplots(1,2,figsize=(13,7),layout='constrained')
q=metrics.sort_values('endpoint_rank');y=np.arange(len(q))
axs[0].barh(y,q.abs_endpoint_change,color=['#137f88' if f in ['energy','acousticness'] else '#b1bdc6' for f in q.feature])
axs[0].set_yticks(y,q.feature);axs[0].invert_yaxis();axs[0].set(xlabel='Absolute 1950s–2010s mean change (reference SD)',title='Primary criterion | magnitude of net change')
for j,r in enumerate(q.itertuples()):axs[0].text(r.abs_endpoint_change,j,f' {r.abs_endpoint_change:.3f}',va='center',fontsize=9)
axs[0].set_xlim(0,2.05)
for f,c in zip(['acousticness','energy','log_duration','loudness'],['#137f88','#d66139','#7654a3','#3267a8']):
    z=sens[sens.feature.eq(f)];axs[1].plot(z.minimum_credits,z.endpoint_rank,'o-',label=f,color=c)
axs[1].set(xticks=[3,5,10],yticks=[1,2,3,4],ylim=(4.4,.6),xlabel='Minimum credits per artist per decade',ylabel='Absolute endpoint-change rank',title='Sensitivity | energy is not always second')
axs[1].legend(fontsize=9)
export(fig,'07_feature_selection')

# Frozen early source profiles vs the first five observed years of later followers.
pr=[];fr=[];diag=[]
for i in ids:
    t=int(first_year[i])
    p,f,d=compare_followers(i,t,t+4,target_genre='Country')
    pr.extend(p);fr.extend(f);diag.append(d)
pairs=pd.DataFrame(pr,columns=PAIR_COLUMNS)
features=pd.DataFrame(fr,columns=FEATURE_COLUMNS)
pairs.to_csv(DEST/'leader_follower_pairs.csv',index=False)
features.to_csv(DEST/'leader_follower_features.csv',index=False)
pd.DataFrame(diag).to_csv(DEST/'leader_follower_support.csv',index=False)
summary=pairs.groupby(['artist_id','target_cohort']).agg(matched_followers=('follower_id','size'),mean_excess=('excess_similarity','mean'),median_excess=('excess_similarity','median'),minimum_controls=('controls','min'))
summary['reported_mean']=summary.mean_excess.where(summary.matched_followers.ge(5))
summary.to_csv(DEST/'leader_follower_summary.csv')
fs=features.groupby(['artist_id','target_cohort','feature']).agg(followers=('follower_id','size'),mean_advantage=('distance_advantage','mean'))
fs.to_csv(DEST/'leader_follower_feature_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained',gridspec_kw={'width_ratios':[1.4,1]})
mat=summary.reported_mean.unstack().reindex(index=ids,columns=periods)
vmax=max(.04,float(np.nanmax(np.abs(mat.to_numpy())))) if mat.notna().any().any() else .04
im=axs[0].imshow(mat,aspect='auto',cmap='RdBu',vmin=-vmax,vmax=vmax)
axs[0].set_yticks(range(len(ids)),[nodes.at[i,'artist_name'] for i in ids],fontsize=9);axs[0].set_xticks(range(7),[f'{b}s' for b in periods],rotation=30)
for row,i in enumerate(ids):
    for col,b in enumerate(periods):
        if (i,b) in summary.index:
            r=summary.loc[(i,b)];txt=f'{r.mean_excess:+.3f}\nn={int(r.matched_followers)}' if r.matched_followers>=5 else f'n={int(r.matched_followers)}\nwithheld'
            axs[0].text(col,row,txt,ha='center',va='center',fontsize=8)
axs[0].set(title='Early-profile resemblance across follower cohorts',xlabel='Follower career-start cohort');fig.colorbar(im,ax=axs[0],label='Similarity advantage over matched nonfollowers')
d=pd.DataFrame(diag);y=np.arange(len(ids))
axs[1].barh(y-.24,d.network_targets,.24,label='Eligible network followers',color='#c1cbd0')
axs[1].barh(y,d.audio_targets,.24,label='Time-ordered early solo profiles',color='#7654a3')
axs[1].barh(y+.24,d.matched_targets,.24,label='At least 5 controls',color='#137f88')
axs[1].set_yticks(y,[nodes.at[i,'artist_name'] for i in ids],fontsize=8);axs[1].invert_yaxis();axs[1].set(title='Coverage of the comparison',xlabel='Country followers');axs[1].legend(fontsize=8,loc='lower right')
export(fig,'06_leader_follower_bridge')
assert len(pairs)==sum(d['matched_targets'] for d in diag)
assert pairs.empty or (pairs.controls.ge(5).all() and pairs.target_start.gt(pairs.artist_id.map(first_year)+4).all())
print('R6: all supported decade decompositions and time-ordered follower comparisons verified.')
print(decomposition[decomposition.minimum_credits.eq(5)&decomposition.feature.isin(['energy','acousticness'])].to_string(index=False))
print(pd.DataFrame(diag).to_string(index=False));print(summary.to_string())
# Calendar matching is substantive: show the cost of dropping it as sensitivity only.
calendar_rows=[]
for i in ids:
    t=int(first_year[i])
    pp,_,dd=compare_followers(i,t,t+4,target_genre='Country',match_recording_decade=False)
    q=pd.DataFrame(pp,columns=PAIR_COLUMNS)
    for b,group in q.groupby('target_cohort'):
        strict=pairs[pairs.artist_id.eq(i)&pairs.target_cohort.eq(b)]
        calendar_rows.append([i,nodes.at[i,'artist_name'],b,len(strict),strict.excess_similarity.mean(),len(group),group.excess_similarity.mean()])
pd.DataFrame(calendar_rows,columns=['artist_id','name','target_cohort','strict_n','strict_mean','no_calendar_match_n','no_calendar_match_mean']).to_csv(DEST/'leader_follower_calendar_sensitivity.csv',index=False)
