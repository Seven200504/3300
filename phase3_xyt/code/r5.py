from csv_store import read_csv
from common import *
OUT=ROOT/"output/r5"
OUT.mkdir(parents=True,exist_ok=True)
from scipy.stats import spearmanr
import hashlib,json
files=[shared/'artist_year.csv.gz',shared/'artist_decade.csv.gz',shared/'feature_scaler.csv',shared/'nodes.csv',shared/'edges.csv',BASE/'r1/metrics.csv',BASE/'r3/genre_pairs.csv',BASE/'r4/followers.csv',P2/'code/output/full_music_data_clean.csv']
(OUT.parent/'input_hashes.json').write_text(json.dumps({str(f.relative_to(P2)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
coverage=year.groupby(['year','genre']).agg(artists=('artist_id','size'),credits=('n_songs','sum')).reset_index()
coverage.to_csv(OUT/'coverage.csv',index=False)
annual=year.groupby('year')[Z].mean().join(year.groupby('year').agg(artists=('artist_id','size'),credits=('n_songs','sum')))
annual=annual.reindex(range(int(year.year.min()),int(year.year.max())+1))
print(annual[['artists','credits']].iloc[[0,1,-2,-1]])
print('repeat-year artists',year.groupby('artist_id').size().gt(1).sum(),'of',year.artist_id.nunique())
print('Years with >=20:',annual.index[annual.artists.ge(20)].tolist())

eligible=annual.artists.ge(20)
changes=annual[Z].where(eligible,axis=0).diff()
annual['J']=np.sqrt(changes.pow(2).mean(axis=1))
for k in Z:annual['jump_'+k]=changes[k]
# Complete six decades; include every known genre with >=5 artists EVERY year.
ys=range(1960,2020)
n=coverage.pivot(index='year',columns='genre',values='artists').reindex(ys).fillna(0)
genres=n.columns[(n.min()>=5)&(n.columns!='Unknown')].tolist()
w=n[genres].sum()/n[genres].values.sum()
sub=year[year.year.isin(ys)&year.genre.isin(genres)]
fixed=sub.groupby(['year','genre'])[Z].mean().mul(w,level='genre',axis=0).groupby('year').sum()
subset=sub.groupby('year')[Z].mean()
for key,table in [('fixed',fixed),('same_genres',subset)]:
    annual=annual.join(table.add_prefix(key+'_'))
    annual[key+'_J']=np.sqrt(table.diff().pow(2).mean(axis=1))
prev=year.copy();prev.year+=1
common=year.merge(prev,on=['artist_id','year'],suffixes=('','_prev'),validate='one_to_one')
for k in Z:common[k]-=common[k+'_prev']
cont=common.groupby('year')[Z].mean()
annual['continuing_n']=common.groupby('year').size()
annual['continuing_J']=np.sqrt(cont.pow(2).mean(axis=1)).where(annual.continuing_n>=20)
annual.to_csv(OUT/'annual.csv',index_label='year')
w.rename('weight').to_csv(OUT/'fixed_weights.csv')
bins=[]
for start in range(1930,2020,5):
    p=window(start,start+4)
    bins.append([start,len(p),p.n_songs.sum(),*p[Z].mean()])
five=pd.DataFrame(bins,columns=['start','artists','credits',*Z]).set_index('start')
five['J']=np.sqrt(five[Z].where(five.artists>=20,axis=0).diff().pow(2).mean(axis=1))
five.to_csv(OUT/'five_year.csv')
print('Fixed weights:',w,'Annual largest movements:',annual.J.nlargest(5))

def segment(table,minlen,maxbreaks=3):
    y=table.to_numpy();n,p=y.shape
    loss=np.full((n+1,n+1),np.inf)
    for i in range(n-minlen+1):
        for j in range(i+minlen,n+1):
            design=np.column_stack([np.ones(j-i),np.arange(j-i)])
            fit=design@np.linalg.lstsq(design,y[i:j],rcond=None)[0]
            loss[i,j]=np.square(y[i:j]-fit).sum()
    dp=np.full((maxbreaks+2,n+1),np.inf);dp[0,0]=0
    prev=np.zeros_like(dp,dtype=int);rows=[]
    for s in range(1,maxbreaks+2):
        for j in range(s*minlen,n+1):
            idx=np.arange((s-1)*minlen,j-minlen+1)
            vals=dp[s-1,idx]+loss[idx,j]
            prev[s,j]=idx[vals.argmin()];dp[s,j]=vals.min()
        if n<s*minlen:continue
        j=n;cuts=[]
        for b in range(s,1,-1):
            j=prev[b,j];cuts.append(int(table.index[j]))
        # Working isotropic residual likelihood; partition points count as parameters.
        parameters=2*p*s+(s-1)+1
        bic=n*p*np.log(dp[s,n]/(n*p))+parameters*np.log(n*p)
        rows.append([s-1,dp[s,n],bic,'/'.join(map(str,sorted(cuts)))])
    return pd.DataFrame(rows,columns=['K','SSE','working_BIC','breaks'])
fits=[]
for spec,data,L in [('annual20',annual.loc[1949:2019,Z],5),('annual20_L10',annual.loc[1949:2019,Z],10),('annual50',annual.loc[1953:2019,Z],5),('all_on_common',annual.loc[1960:2019,Z],5),('observed_four',subset,5),('fixed_four',fixed,5),('five_year',five.loc[1950:2015,Z],3)]:
    assert not data.isna().any().any()
    fits.append(segment(data,L).assign(spec=spec,min_periods=L))
fits=pd.concat(fits,ignore_index=True);fits.to_csv(OUT/'segments.csv',index=False)
print(fits.loc[fits.groupby('spec').working_BIC.idxmin()])
first=year.groupby('artist_id').year.min()
incoming=edges.groupby('follower_id').influencer_id.agg(set).to_dict()
outgoing=edges.groupby('influencer_id').follower_id.agg(set).to_dict()
rows=[]
for t,artists in first.groupby(first):
    early=window(t,t+4);prior=window(t-5,t-1);longprior=window(t-10,t-1)
    for artist in artists.index:
        x=early.loc[artist];g=x.genre
        if g=='Unknown' or t+4>2020 or x.n_songs<5:continue
        history=prior[(prior.genre==g)&(prior.n_songs>=3)]
        if len(history)<20:continue
        assert artist not in history.index
        diff=x[Z].to_numpy(float)-history[Z].mean().to_numpy()
        nov=np.sqrt(np.mean(diff**2))
        # Each predecessor compared against its other peers (leave one out).
        v=history[Z].to_numpy();rms=np.sqrt(np.mean(((v-v.mean(0))*len(v)/(len(v)-1))**2,axis=1))
        hist10=longprior[(longprior.genre==g)&(longprior.n_songs>=3)]
        nov10=np.sqrt(np.mean((x[Z].to_numpy(float)-hist10[Z].mean().to_numpy())**2))
        s=v.std(0);local=np.sqrt(np.mean((diff/s)**2)) if np.all(s>0) else np.nan
        low=t+5;high=t+24
        cohorts=list(range(((low+9)//10)*10,high-8,10))
        potential=nodes.index[nodes.active_start.isin(cohorts)&(nodes.index!=artist)]
        complete=high<=year.year.max()
        nreach=len(outgoing.get(artist,set()).intersection(potential)) if complete else np.nan
        rows.append([artist,nodes.loc[artist,'artist_name'],g,t,int(x.n_songs),len(v),nov,(rms<=nov).mean(),nov10,local,
                     low,high,'/'.join(map(str,cohorts)),complete,len(potential) if complete else np.nan,nreach,*diff])
nov=pd.DataFrame(rows,columns=['artist_id','name','genre','start','early_n','background_n','novelty','past_percentile','novelty10','past_only_scale','follow_start','follow_end','cohorts','complete','opportunities','reach',*Z]).set_index('artist_id')
ok=nov[nov.complete].copy()
ok['reach_per_1000']=1000*ok.reach/ok.opportunities
ok['peer_n']=ok.groupby('cohorts').reach.transform('size')
ok['reach_pct']=ok.groupby('cohorts').reach.rank(method='average',pct=True)
nov=nov.join(ok[['reach_per_1000','peer_n','reach_pct']])
nov.to_csv(OUT/'artist_evidence.csv')
ok=ok[ok.peer_n>=10]
picks=[]
for q in [.75,.85]:
    mask=(ok.past_percentile>=q)&(ok.reach_pct>=q)&(ok.reach>0)
    print('Descriptive high/high threshold',q,mask.sum())
both=ok[(ok.past_percentile>=.75)&(ok.reach_pct>=.75)&(ok.reach>0)].sort_values(['reach_per_1000','novelty'],ascending=False)
picks=list(both.head(3).index)
counter=ok[ok.reach.eq(0)].novelty.idxmax()
hub=ok[ok.past_percentile<.5].reach_per_1000.idxmax()
case=nov.loc[picks+[counter,hub]].copy()
case['role']=['joint candidate']*len(picks)+['unusual / zero reach','high reach / typical']
case.to_csv(OUT/'early_cases.csv')
print('artists:',len(nov),'complete:',len(ok),'positive reach:',ok.reach.gt(0).sum())
print(case[['name','genre','start','early_n','background_n','novelty','past_percentile','reach','reach_pct']])
print('Rank associations:',ok.novelty.corr(ok.reach_per_1000,method='spearman'),ok.past_percentile.corr(ok.reach_pct,method='spearman'),ok.novelty.corr(ok.novelty10,method='spearman'),ok.novelty.corr(ok.past_only_scale,method='spearman'))
# Apply the existing Phase 2 eligibility and scaler to corrected solo recordings.
import ast
songs=read_csv(P2/'code/output/full_music_data_clean.csv')
credit=songs.artists_id.map(ast.literal_eval)
solo=songs[credit.str.len().eq(1)].copy()
solo['artist_id']=credit[credit.str.len().eq(1)].str[0].astype(int)
solo=solo.join(nodes[['genre','active_start']],on='artist_id',how='inner')
solo=solo[solo.tempo.gt(0)&solo.duration_ms.gt(0)&solo.year.ge(solo.active_start)&
          solo.release_date.str.extract(r'(\d{4})')[0].astype(int).eq(solo.year)].copy()
solo['log_duration']=np.log(solo.duration_ms/1000)
solo[Z]=((solo[F]-scale['mean'])/scale.sd).to_numpy()
solo['n_songs']=1
def band(n):return np.digitize(n,[10,30,100])
aligned=[];diag=[]
for artist in picks:
    r=nov.loc[artist];t=int(r.start)
    early=profile(solo[solo.year.between(t,t+4)])
    latest=profile(solo[solo.year.between(t+5,t+24)])
    later_cohorts=list(map(int,r.cohorts.split('/')))
    fol=nodes.index[nodes.active_start.isin(later_cohorts)].intersection(list(outgoing.get(artist,set())))
    if artist not in early.index:diag.append([artist,len(fol),0,0,0]);continue
    x=early.loc[artist];n0=x.n_songs
    pool=early[(early.genre==r.genre)&(early.active_start==nodes.at[artist,'active_start'])&(early.n_songs>=5)&(band(early.n_songs)==band(n0))]
    for j in fol.intersection(latest.index):
        ctrl=pool.drop(index=list(incoming[j]|{j,artist}),errors='ignore')
        if n0<5 or latest.at[j,'n_songs']<5 or len(ctrl)<5:continue
        y=latest.loc[j,Z].to_numpy(float);source=x[Z].to_numpy(float)
        real=1/(1+np.sqrt(np.mean((source-y)**2)))
        baseline=np.mean(1/(1+np.sqrt(np.mean((ctrl[Z].to_numpy()-y)**2,axis=1))))
        aligned.append([artist,j,len(ctrl),real,baseline,real-baseline])
    diag.append([artist,len(fol),n0,len(pool),sum(i[0]==artist for i in aligned)])
al=pd.DataFrame(aligned,columns=['artist_id','follower','controls','real','baseline','delta'])
al.to_csv(OUT/'alignment.csv',index=False)
pd.DataFrame(diag,columns=['artist_id','later_edges','early_solo','pool_including_source','matched_followers']).to_csv(OUT/'alignment_support.csv',index=False)
print(diag,al.groupby('artist_id').delta.agg(['count','mean']))

flow=[('artist with valid recording year',len(first))]
known=first[nodes.loc[first.index,'genre'].ne('Unknown').to_numpy()]
early_ok=[]
for t,group in known.groupby(known):
    if t+4<=2020:
        counts=window(t,t+4).loc[group.index,'n_songs']
        early_ok.extend(counts[counts>=5].index)
flow.extend([('known genre and complete early window',int((known<=2016).sum())),
             ('at least 5 early credits',len(early_ok)),('at least 20 historical peers',len(nov)),
             ('complete 20-year follow-up',int(nov.complete.sum())),('at least 10 same-opportunity peers',len(ok))])
pd.DataFrame(flow,columns=['rule','artists']).to_csv(OUT/'artist_flow.csv',index=False)
assert np.isclose(w.sum(),1) and np.allclose(subset.loc[2015],np.average(
    sub[sub.year==2015].groupby('genre')[Z].mean(),axis=0,weights=n.loc[2015,genres]))
for i in case.index:
    r=case.loc[i];t=int(r.start)
    ss=year[(year.artist_id==i)&year.year.between(t,t+4)]
    direct=np.average(ss[Z],axis=0,weights=ss.n_songs)
    assert np.allclose(direct,window(t,t+4).loc[i,Z].to_numpy(float))
    targets=edges[edges.influencer_id.eq(i)]
    assert targets.follower_active_start.between(t+5,t+24-9).sum()==r.reach
# Quantify how much each selected candidate's deviation coincides with subsequent genre movement.
# All three periods exclude the candidate from peer means, using disjoint recording years.
directions=[]
for i in picks:
    r=nov.loc[i];t=int(r.start)
    old=window(t-5,t-1);new=window(t+5,t+9)
    g0=old[(old.genre==r.genre)&old.n_songs.ge(3)].drop(index=i,errors='ignore')
    g1=new[(new.genre==r.genre)&new.n_songs.ge(3)].drop(index=i,errors='ignore')
    d=g1[Z].mean().to_numpy()-g0[Z].mean().to_numpy()
    v=r[Z].to_numpy(float)
    directions.append([i,len(g0),len(g1),np.dot(v,d)/(np.linalg.norm(v)*np.linalg.norm(d)),np.sqrt(np.mean(d*d))])
pd.DataFrame(directions,columns=['artist_id','past_artists','later_artists','direction_cosine','genre_shift_RMS']).to_csv(OUT/'candidate_directions.csv',index=False)

annual = read_csv(OUT/'annual.csv').set_index('year').loc[1949:2019]
five = read_csv(OUT/'five_year.csv').set_index('start').loc[1950:2015]
cases = read_csv(OUT/'early_cases.csv').set_index('artist_id')
nov = read_csv(OUT/'artist_evidence.csv').set_index('artist_id')
complete = nov[nov.complete & nov.peer_n.ge(10)]
al = read_csv(OUT/'alignment.csv')
support = read_csv(OUT/'alignment_support.csv').set_index('artist_id')
colors = ['#087e8b', '#d45a35', '#7654a3']

def export(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

def display_name(name):
    # Repair reversible UTF-8 mojibake for display only; CSV keeps source names.
    try:
        return name.encode('latin1').decode('utf8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name

# Raw annual observations remain visible; five-year points use bin midpoints.
fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True, layout='constrained',
                       gridspec_kw={'height_ratios':[1,1,1,1.15,.65]})
for a, feature, color in zip(ax[:3], ['acousticness','energy','log_duration'], colors):
    a.axvspan(1960, 1974, color='#ead69c', alpha=.35)
    a.plot(annual.index, annual['z_'+feature], color=color, alpha=.3, lw=1, label='Annual mean')
    a.plot(five.index+2, five['z_'+feature], 'o-', color=color, lw=2.3, ms=4, label='Five-year mean (bin midpoint)')
    a.set_ylabel(feature+'\n(reference SD)')
    a.axhline(0, color='gray', lw=.6)
ax[0].set_title('R5 | A sustained change in the observed musical profile', loc='left', fontsize=17)
ax[0].legend(loc='upper right', fontsize=9)
ax[0].text(1962, .1, '1960–74: candidate transition interval', fontsize=9)
a=ax[3]
a.plot(annual.index, annual.J, color='#35495c', alpha=.55, label='Annual displacement J')
a.plot(annual.index, annual.continuing_J, color='#087e8b', alpha=.8, label='Continuing artists: annual net displacement')
a.plot(five.index[1:]+2, five.J.iloc[1:], 'o-', color='#d45a35', label='Adjacent five-year displacement J')
a.axvspan(1960,1974,color='#ead69c',alpha=.35)
for t in [1965,1970]:
    a.annotate(f'{five.at[t,"J"]:.3f}',(t+2,five.at[t,'J']),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
a.set_ylabel('RMS displacement'); a.legend(loc='upper right',fontsize=8)
a.text(.01,.02,'Annual and five-year J use different time steps; compare peaks within each series.',transform=a.transAxes,fontsize=8)
ax[4].plot(annual.index,annual.artists,color='#35495c',label='Observed artists')
ax[4].plot(annual.index,annual.continuing_n,color='#087e8b',label='Also observed in preceding year')
ax[4].set(xlabel='Recording year',ylabel='Artists',xlim=(1949,2019));ax[4].legend(fontsize=8)
export(fig,'01_time_trends')

fig, axs=plt.subplots(5,2,figsize=(14,13),sharex=True,layout='constrained')
for a,f in zip(axs.flat,F):
    a.axvspan(1960,1974,color='#ead69c',alpha=.35)
    a.plot(annual.index,annual['z_'+f],color='#bac2ca',lw=.9,label='All artists: annual')
    a.plot(five.index+2,five['z_'+f],'o-',color='#35495c',ms=3,label='All artists: five-year')
    a.plot(annual.index,annual['same_genres_z_'+f],color='#d45a35',alpha=.85,lw=1,label='Four genres: observed weights')
    a.plot(annual.index,annual['fixed_z_'+f],color='#087e8b',ls='--',lw=1.2,label='Same four genres: fixed weights')
    a.set_title(f,loc='left');a.set_ylabel('Reference SD');a.set_xlim(1949,2019)
axs[0,0].legend(fontsize=7)
for a in axs[-1]:a.set_xlabel('Recording year')
fig.suptitle('All 10 features | Composition sensitivity uses the same four genres',fontsize=17)
export(fig,'02_all_features')

# Paired five-year changes and their rankings are directly reproducible.
changes=five[Z].diff().iloc[1:]
changes.columns=F
changes.insert(0,'comparison',[f'{t-5}–{t-1} → {t}–{t+4}' for t in changes.index])
changes['J']=np.sqrt(changes[F].pow(2).mean(axis=1))
changes['rank_by_J']=changes.J.rank(ascending=False,method='min').astype(int)
assert np.allclose(changes.J,five.J.iloc[1:])
changes.to_csv(OUT/'period_comparisons.csv',index_label='later_period_start')

# Candidate and future-genre changes share a past baseline; this is descriptive.
feature_rows=[]; history_rows=[]; roster=[]
fig,axs=plt.subplots(3,2,figsize=(15,12),layout='constrained',gridspec_kw={'width_ratios':[1.6,1]})
for row,(i,r) in enumerate(cases.iloc[:3].iterrows()):
    t=int(r.start)
    past=window(t-5,t-1);past=past[(past.genre==r.genre)&past.n_songs.ge(3)].drop(index=i,errors='ignore')
    early=window(t,t+4).loc[i]
    later=window(t+5,t+9);later=later[(later.genre==r.genre)&later.n_songs.ge(3)].drop(index=i,errors='ignore')
    p=past[Z].mean();e=early[Z].astype(float);l=later[Z].mean()
    assert np.allclose((e-p).to_numpy(),r[Z].to_numpy(float))
    for f,z in zip(F,Z):
        feature_rows.append([i,r['name'],t,f,p[z],e[z],l[z],e[z]-p[z],l[z]-p[z],len(past),len(later)])
    a=axs[row,0];y=np.arange(10)
    a.barh(y-.18,(e-p).values,.34,color=colors[row],label=f'{r["name"]}: {t}–{t+4}')
    a.barh(y+.18,(l-p).values,.34,color='#bdc6cd',label=f'Later same-genre peers: {t+5}–{t+9}')
    a.set_yticks(y,F);a.invert_yaxis();a.axvline(0,color='gray',lw=.8)
    a.set_title(f'{r["name"]} | deviation from {t-5}–{t-1}',loc='left')
    a.set_xlabel('Difference from past same-genre mean (reference SD)');a.legend(fontsize=8)
    v=past[Z].to_numpy();dist=np.sqrt(np.mean(((v-v.mean(0))*len(v)/(len(v)-1))**2,axis=1))
    for j,d in zip(past.index,dist):history_rows.append([i,int(j),d])
    a=axs[row,1];a.hist(dist,bins=16,color='#c4ced5',edgecolor='white')
    a.axvline(r.novelty,color=colors[row],lw=2,label=f'Candidate N = {r.novelty:.3f}')
    a.set_title(f'Historical novelty percentile: {100*r.past_percentile:.1f}%',loc='left')
    a.set(xlabel='RMS distance from historical peer center',ylabel='Past artists');a.legend(fontsize=8)
    a.text(.97,.95,f'Past artists: {len(past)}\nEarly credits: {int(r.early_n)}\nLater followers: {int(r.reach)} / {int(r.opportunities)}\nReach percentile: {100*r.reach_pct:.1f}%',transform=a.transAxes,ha='right',va='top',fontsize=9)
    followers=edges[edges.influencer_id.eq(i)&edges.follower_active_start.between(t+5,t+15)].copy()
    assert len(followers)==r.reach
    followers['source_early_window']=f'{t}–{t+4}'
    followers['followup_window']=f'{t+5}–{t+24}'
    roster.append(followers)
fig.suptitle('Candidate evidence | Distinctive early profiles and later recorded connections',fontsize=17)
export(fig,'03_artist_profiles')
pd.DataFrame(feature_rows,columns=['artist_id','name','start','feature','past_mean','early_mean','later_genre_mean','early_minus_past','later_minus_past','past_artists','later_artists']).to_csv(OUT/'artist_feature_comparisons.csv',index=False)
pd.DataFrame(history_rows,columns=['candidate_id','past_artist_id','leave_one_out_novelty']).to_csv(OUT/'historical_novelty.csv',index=False)
followers=pd.concat(roster,ignore_index=True)
followers.to_csv(OUT/'recorded_followers.csv',index=False)
al['follower_name']=al.follower.map(nodes.artist_name)
al.to_csv(OUT/'alignment.csv',index=False)

# Feature-level extension of the existing solo-recording matched comparison.
songs=read_csv(P2/'code/output/full_music_data_clean.csv')
credit=songs.artists_id.map(ast.literal_eval)
solo=songs[credit.str.len().eq(1)].copy();solo['artist_id']=credit[credit.str.len().eq(1)].str[0].astype(int)
solo=solo.join(nodes[['genre','active_start']],on='artist_id',how='inner')
solo=solo[solo.tempo.gt(0)&solo.duration_ms.gt(0)&solo.year.ge(solo.active_start)&solo.release_date.str.extract(r'(\d{4})')[0].astype(int).eq(solo.year)].copy()
solo['log_duration']=np.log(solo.duration_ms/1000);solo[Z]=((solo[F]-scale['mean'])/scale.sd).to_numpy();solo['n_songs']=1
incoming=edges.groupby('follower_id').influencer_id.agg(set).to_dict()
feature_match=[]
for i,group in al.groupby('artist_id'):
    r=cases.loc[i];t=int(r.start)
    e=profile(solo[solo.year.between(t,t+4)]);l=profile(solo[solo.year.between(t+5,t+24)])
    x=e.loc[i];band=lambda n:np.digitize(n,[10,30,100])
    pool=e[(e.genre==r.genre)&(e.active_start==nodes.at[i,'active_start'])&e.n_songs.ge(5)&(band(e.n_songs)==band(x.n_songs))]
    for q in group.itertuples():
        ctrl=pool.drop(index=list(incoming[q.follower]|{q.follower,i}),errors='ignore')
        y=l.loc[q.follower,Z].to_numpy(float);source=x[Z].to_numpy(float);cv=ctrl[Z].to_numpy(float)
        real=1/(1+np.sqrt(np.mean((source-y)**2)));base=np.mean(1/(1+np.sqrt(np.mean((cv-y)**2,axis=1))))
        assert len(ctrl)==q.controls and np.isclose(real-base,q.delta)
        for k,f in enumerate(F):feature_match.append([i,q.follower,q.follower_name,f,abs(source[k]-y[k]),np.abs(cv[:,k]-y[k]).mean()])
fm=pd.DataFrame(feature_match,columns=['artist_id','follower_id','follower_name','feature','source_abs_distance','control_abs_distance'])
fm['distance_advantage']=fm.control_abs_distance-fm.source_abs_distance
fm.to_csv(OUT/'matched_feature_evidence.csv',index=False)
summary=fm.groupby('feature').agg(mean_advantage=('distance_advantage','mean'),positive_followers=('distance_advantage',lambda s:int(s.gt(0).sum())),followers=('follower_id','size')).reindex(F)
summary.to_csv(OUT/'matched_feature_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(15,9),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
ordered=al.sort_values('delta');y=np.arange(len(ordered))
axs[0].barh(y,ordered.delta,color=np.where(ordered.delta>0,'#087e8b','#d45a35'))
axs[0].set_yticks(y,[display_name(n) for n in ordered.follower_name],fontsize=8);axs[0].axvline(0,color='gray')
axs[0].set_title(f'Metallica: {int(al.delta.gt(0).sum())}/{len(al)} followers above controls',loc='left')
axs[0].set_xlabel('Similarity to Metallica − mean similarity to matched controls')
axs[1].barh(F,summary.mean_advantage,color=np.where(summary.mean_advantage>0,'#087e8b','#d45a35'))
axs[1].axvline(0,color='gray');axs[1].invert_yaxis()
axs[1].set_title('Which features are closer to Metallica?',loc='left');axs[1].set_xlabel('Mean absolute-distance advantage (SD)\nPositive = closer to Metallica than controls')
fig.suptitle(f'Source-side matched check | Mean excess similarity = {al.delta.mean():+.5f}\n34 of 52 eligible followers; Bob Dylan and Kiss lack sufficient controls',fontsize=15)
export(fig,'04_follower_comparison')

# Five different bin origins: temporal aggregation stability, not a significance test.
shift_rows=[]
for offset in range(5):
    starts=list(range(1950+offset,2016,5))
    means=pd.DataFrame([window(t,t+4)[Z].mean() for t in starts],index=starts)
    delta=means.diff().iloc[1:]
    js=np.sqrt(delta.pow(2).mean(axis=1))
    ranks=js.rank(ascending=False,method='min')
    for t in js.index:
        shift_rows.append([offset,t-5,t,float(js[t]),int(ranks[t]),len(window(t-5,t-1)),len(window(t,t+4))])
shift=pd.DataFrame(shift_rows,columns=['origin_offset','before_start','after_start','J','rank','before_artists','after_artists'])
shift.to_csv(OUT/'bin_origin_stability.csv',index=False)
fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
for offset,g in shift.groupby('origin_offset'):
    ax.plot(g.after_start,g.J,'o-',ms=3,label=f'Origin +{offset} years')
ax.axvspan(1960,1974,color='#ead69c',alpha=.3)
ax.set(xlabel='Start of later five-year bin',ylabel='RMS displacement',title='Five-year shifts under all five bin origins')
ax.legend(ncol=3,fontsize=8)
export(fig,'05_bin_origin_stability')

# Local, period-linked candidates allow an established artist to change mid-career.
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
macro=(five.loc[1970,Z]-five.loc[1960,Z]).to_numpy(float)
no_live=[k for k,f in enumerate(F) if f!='liveness']
local_rows=[];neighbors=[]
for t in [1960,1965,1970]:
    current=window(t,t+4);past=window(t-5,t-1);future=window(t+5,t+9)
    allowed_cohorts=list(range(((t+5+9)//10)*10,t+24-8,10))
    opportunities=nodes[nodes.active_start.isin(allowed_cohorts)]
    for i,x in current[current.n_songs.ge(5)&current.genre.ne('Unknown')].iterrows():
        h=past[past.genre.eq(x.genre)&past.n_songs.ge(3)].drop(index=i,errors='ignore')
        if len(h)<20:continue
        a=h[Z].to_numpy(float);v=x[Z].to_numpy(float);diff=v-a.mean(0)
        olddist=np.sqrt(np.mean(((a-a.mean(0))*len(a)/(len(a)-1))**2,axis=1))
        dist=np.linalg.norm(diff)/np.sqrt(10)
        pair=cdist(a,a)/np.sqrt(10);np.fill_diagonal(pair,np.inf)
        hd=np.sort(pair,axis=1)[:,:3].mean(axis=1)
        nd=np.linalg.norm(a-v,axis=1)/np.sqrt(10);nearest=np.argsort(nd)[:3]
        nn=nd[nearest].mean()
        h9=a[:,no_live];d9=diff[no_live]
        old9=np.sqrt(np.mean(((h9-h9.mean(0))*len(h9)/(len(h9)-1))**2,axis=1))
        later=future[future.genre.eq(x.genre)&future.n_songs.ge(3)].drop(index=i,errors='ignore')
        movement=later[Z].mean().to_numpy()-a.mean(0)
        cosine=np.dot(diff,macro)/(np.linalg.norm(diff)*np.linalg.norm(macro))
        later_cos=np.dot(diff,movement)/(np.linalg.norm(diff)*np.linalg.norm(movement)) if len(later)>=20 else np.nan
        reach=len(outgoing.get(i,set()).intersection(opportunities.index))
        local_rows.append([i,nodes.at[i,'artist_name'],x.genre,t,t+4,int(first_year[i]),int(x.n_songs),len(h),dist,(olddist<=dist).mean(),nn,(hd<=nn).mean(),np.linalg.norm(d9)/3,(old9<=np.linalg.norm(d9)/3).mean(),cosine,later_cos,len(later),reach,len(opportunities),1000*reach/len(opportunities),int((first_year[i]<t))])
        for rank,k in enumerate(nearest,1):neighbors.append([i,t,rank,int(h.index[k]),nodes.at[h.index[k],'artist_name'],nd[k]])
local=pd.DataFrame(local_rows,columns=['artist_id','name','genre','start','stop','first_observed','credits','past_artists','novelty','novelty_pct','nearest3_distance','nearest3_pct','novelty_no_liveness','pct_no_liveness','macro_direction_cosine','later_genre_cosine','later_artists','later_followers','eligible_nodes','reach_per_1000','established_before_window'])
local['reach_pct']=local.groupby('start').later_followers.rank(pct=True,method='average')
local['legacy_selected']=False
for t,g in local.groupby('start'):
    choices=g[g.novelty_pct.ge(.75)&g.reach_pct.ge(.75)&g.later_followers.gt(0)&g.macro_direction_cosine.gt(0)]
    chosen=choices.sort_values(['reach_per_1000','artist_id'],ascending=[False,True]).head(2).index
    local.loc[chosen,'legacy_selected']=True
# Revolution-linked evidence: individual deviation is projected onto the observed
# system transition, excluding that artist from BOTH system endpoint means.
# This is retrospective correspondence, not an individual's causal contribution.
link_rows=[];detail=[]
for r in local.itertuples():
    i=int(r.artist_id);t=int(r.start)
    h=window(t-5,t-1);h=h[h.genre.eq(r.genre)&h.n_songs.ge(3)].drop(index=i,errors='ignore')
    a=window(t,t+4).loc[i,Z].to_numpy(float);baseline=h[Z].mean().to_numpy()
    d=a-baseline
    m=window(1970,1974).drop(index=i,errors='ignore')[Z].mean().to_numpy()-window(1960,1964).drop(index=i,errors='ignore')[Z].mean().to_numpy()
    u=m/np.linalg.norm(m)
    components=d*u/np.sqrt(10)
    projection=components.sum()
    cosine=d@u/np.linalg.norm(d) if np.linalg.norm(d)>0 else np.nan
    projection9=d[no_live]@m[no_live]/np.linalg.norm(m[no_live])/3
    later=window(t+5,t+9);later=later[later.genre.eq(r.genre)&later.n_songs.ge(3)].drop(index=i,errors='ignore')
    l=later[Z].mean().to_numpy()-baseline
    link_rows.append([i,t,projection,cosine,projection9])
    for k,f in enumerate(F):
        detail.append([i,r.name,t,f,baseline[k],a[k],d[k],m[k],u[k],components[k],d[k]**2/np.sum(d*d),l[k],len(h),len(later)])
link=pd.DataFrame(link_rows,columns=['artist_id','start','transition_projection','transition_cosine','projection_no_liveness'])
local=local.merge(link,on=['artist_id','start'],validate='one_to_one')
local['selected']=False
for t,g in local.groupby('start'):
    pool=g[g.reach_pct.ge(.75)&g.later_followers.gt(0)&g.transition_projection.gt(0)]
    idx=pool.sort_values(['transition_projection','artist_id'],ascending=[False,True]).head(2).index
    local.loc[idx,'selected']=True
local['phase']=local.start.map({1960:'early transition',1965:'transition formation',1970:'late transition'})
# Show selection sensitivity instead of pretending the retrospective cutoff is unique.
selection_rows=[]
for q in [.5,.75,.9]:
    for metric in ['transition_projection','projection_no_liveness']:
        for t,g in local.groupby('start'):
            pool=g[g.reach_pct.ge(q)&g.later_followers.gt(0)&g[metric].gt(0)]
            pick=pool.sort_values([metric,'artist_id'],ascending=[False,True]).head(2)
            for r in pick.itertuples():selection_rows.append([q,metric,t,r.artist_id,r.name])
pd.DataFrame(selection_rows,columns=['reach_cutoff','projection_metric','start','artist_id','name']).to_csv(OUT/'transition_selection_sensitivity.csv',index=False)
pd.DataFrame(detail,columns=['artist_id','name','start','feature','historical_mean','artist_mean','deviation','macro_shift_excluding_artist','unit_direction','projection_component','novelty_squared_share','later_peer_deviation','past_artists','later_artists']).to_csv(OUT/'transition_feature_links.csv',index=False)

local['benchmark']=local.artist_id.eq(754032)
local.to_csv(OUT/'transition_candidates.csv',index=False)
pd.DataFrame(neighbors,columns=['artist_id','start','neighbor_rank','past_artist_id','past_artist_name','RMS_distance']).to_csv(OUT/'historical_neighbors.csv',index=False)
chosen=local[local.selected|local.benchmark].sort_values(['start','selected','reach_per_1000'],ascending=[True,False,False])
chosen.to_csv(OUT/'transition_cases.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
for t,col in zip([1960,1965,1970],colors):
    g=local[local.start.eq(t)]
    axs[0].scatter(g.novelty_pct,g.reach_per_1000,alpha=.25,s=17,color=col,label=f'{t}–{t+4}')
for r in chosen.itertuples():
    axs[0].annotate(f'{r.name} ({r.start})',(r.novelty_pct,r.reach_per_1000),xytext=(3,5),textcoords='offset points',fontsize=7)
axs[0].set(xlabel='Distance-to-center percentile',ylabel='Later followers / 1,000 eligible nodes',title='Period-linked candidates and The Beatles benchmark')
axs[0].legend(fontsize=8)
labels=[f'{r.name}\n{r.start}–{r.stop}' for r in chosen.itertuples()]
y=np.arange(len(chosen))
axs[1].barh(y-.22,chosen.novelty_pct,.22,label='All 10 features',color='#35495c')
axs[1].barh(y,chosen.pct_no_liveness,.22,label='Without liveness',color='#d45a35')
axs[1].barh(y+.22,chosen.nearest3_pct,.22,label='Distance to 3 historical neighbors',color='#087e8b')
axs[1].set_yticks(y,labels,fontsize=8);axs[1].invert_yaxis();axs[1].set(xlim=(0,1.05),xlabel='Historical percentile',title='What kind of distinctiveness?');axs[1].legend(fontsize=8,loc='lower right')
export(fig,'06_transition_candidates')

# Check the original early-career selection on the same no-liveness and near-neighbor scales.
robust=[]
for (t,g),q in nov.reset_index().groupby(['start','genre']):
    h=window(int(t)-5,int(t)-1);h=h[h.genre.eq(g)&h.n_songs.ge(3)]
    a=h[Z].to_numpy(float);a9=a[:,no_live]
    d9=np.sqrt(np.mean(((a9-a9.mean(0))*len(a9)/(len(a9)-1))**2,axis=1))
    pair=cdist(a,a)/np.sqrt(10);np.fill_diagonal(pair,np.inf)
    nearest_hist=np.sort(pair,axis=1)[:,:3].mean(axis=1)
    for r in q.itertuples():
        diff=np.array([getattr(r,z) for z in Z]);v=diff+a.mean(0)
        n9=np.linalg.norm(diff[no_live])/3
        nn=np.sort(np.linalg.norm(a-v,axis=1)/np.sqrt(10))[:3].mean()
        robust.append([r.artist_id,n9,(d9<=n9).mean(),nn,(nearest_hist<=nn).mean()])
robust=pd.DataFrame(robust,columns=['artist_id','novelty_no_liveness','pct_no_liveness','nearest3_distance','nearest3_pct']).set_index('artist_id')
robust.to_csv(OUT/'early_novelty_sensitivity.csv')

# Direction-specific resemblance under the original source-side matching design.
projection=[]
for i,group in al.groupby('artist_id'):
    r=cases.loc[i];t=int(r.start)
    early=profile(solo[solo.year.between(t,t+4)]);later=profile(solo[solo.year.between(t+5,t+24)])
    x=early.loc[i];source=x[Z].to_numpy(float)
    pool=early[(early.genre==r.genre)&(early.active_start==nodes.at[i,'active_start'])&early.n_songs.ge(5)&(np.digitize(early.n_songs,[10,30,100])==np.digitize(x.n_songs,[10,30,100]))]
    u=r[Z].to_numpy(float);u=u/np.linalg.norm(u)
    for q in group.itertuples():
        cv=pool.drop(index=list(incoming[q.follower]|{q.follower,i}),errors='ignore')[Z].to_numpy()
        y=later.loc[q.follower,Z].to_numpy(float)
        advantage=(np.abs((cv-y)@u).mean()-abs((source-y)@u))/np.sqrt(10)
        projection.append([i,q.follower,len(cv),advantage])
pd.DataFrame(projection,columns=['artist_id','follower_id','controls','direction_distance_advantage']).to_csv(OUT/'source_matched_direction.csv',index=False)

# Alternate follower-side comparison includes both original and period-linked cases.
comparisons=cases.iloc[:3].reset_index()[['artist_id','start']].copy()
comparisons=pd.concat([comparisons,local[local.selected|local.benchmark|local.legacy_selected][['artist_id','start']]],ignore_index=True).drop_duplicates()
pairs=[];feat=[];diags=[]
for r in comparisons.itertuples(index=False):
    pr,fr,d=compare_followers(int(r.artist_id),int(r.start),int(r.start)+4,follow_end=int(r.start)+24)
    pairs.extend([row+[int(r.start)] for row in pr]);feat.extend([row+[int(r.start)] for row in fr]);diags.append(d)
pd.DataFrame(pairs,columns=PAIR_COLUMNS+['source_window_start']).to_csv(OUT/'follower_matched_pairs.csv',index=False)
pd.DataFrame(feat,columns=FEATURE_COLUMNS+['source_window_start']).to_csv(OUT/'follower_matched_features.csv',index=False)
pd.DataFrame(diags).to_csv(OUT/'follower_matched_support.csv',index=False)
assert np.allclose(shift[shift.origin_offset.eq(0)].J,changes.J)
assert local.nearest3_pct.between(0,1).all() and local.novelty_pct.between(0,1).all()
print('R5: bin-origin stability, period candidates, historical neighbors and both matching designs verified.')
print(chosen[['name','start','novelty_pct','nearest3_pct','pct_no_liveness','later_followers','macro_direction_cosine']].to_string(index=False))
alt=pd.DataFrame(pairs,columns=PAIR_COLUMNS+['source_window_start'])
agg=alt.groupby(['artist_id','source_window_start']).agg(matched=('follower_id','size'),mean_excess=('excess_similarity','mean'),median_excess=('excess_similarity','median'),mean_direction_advantage=('direction_advantage','mean'))
alt_summary=pd.DataFrame(diags).merge(agg,left_on=['artist_id','source_start'],right_index=True,how='left')
alt_summary['reported_mean']=alt_summary.mean_excess.where(alt_summary.matched_targets.ge(5))
alt_summary.to_csv(OUT/'follower_matched_summary.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(14,7),layout='constrained',sharey=True)
y=np.arange(len(alt_summary));labels=[f'{r.source_name} ({r.source_start}–{r.source_stop})' for r in alt_summary.itertuples()]
for a,key,title in [(axs[0],'mean_excess','Full 10-feature resemblance'),(axs[1],'mean_direction_advantage','Resemblance along the source deviation direction')]:
    values=alt_summary[key].where(alt_summary.matched_targets.ge(5))
    a.barh(y,values,color=np.where(values.ge(0),'#137f88','#d66139'))
    a.axvline(0,color='gray',lw=.8);a.set_title(title,fontsize=11)
    for j,r in enumerate(alt_summary.itertuples()):
        if r.matched_targets<5:a.text(0,j,f'n={r.matched_targets}: withheld',va='center',fontsize=8)
        elif key=='mean_excess':a.text(getattr(r,key),j,f'  {r.matched_targets}/{r.network_targets}',va='center',fontsize=8)
    a.margins(x=.3)
axs[0].set_yticks(y,labels,fontsize=8);axs[0].invert_yaxis()
axs[0].set_xlabel('Mean similarity advantage over matched nonfollowers')
axs[1].set_xlabel('Mean projected distance advantage (reference SD)')
fig.suptitle('Follower-side comparison | Different controls answer a different question',fontsize=15)
export(fig,'07_follower_side_comparison')

# Main figures now concern the identified transition, not global novelty quadrants.
chosen=local[local.selected].sort_values(['start','transition_projection'],ascending=[True,False])
detail=pd.DataFrame(detail,columns=['artist_id','name','start','feature','historical_mean','artist_mean','deviation','macro_shift_excluding_artist','unit_direction','projection_component','novelty_squared_share','later_peer_deviation','past_artists','later_artists'])
fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
for t,col in zip([1960,1965,1970],colors):
    g=local[local.start.eq(t)]
    axs[0].scatter(g.transition_projection,g.reach_per_1000,s=15,alpha=.3,color=col,label=f'{t}-{t+4}')
for r in chosen.itertuples():
    axs[0].scatter(r.transition_projection,r.reach_per_1000,marker='*',s=85,color='#173e4c',edgecolor='white',linewidth=.4)
axs[0].axvline(0,color='gray',lw=.7);axs[0].legend(fontsize=8)
axs[0].set(xlabel='Projection onto transition direction (reference SD)',ylabel='Later followers / 1,000 eligible nodes',title='All eligible period profiles; no novelty cutoff')
y=np.arange(len(chosen));labels=[f'{r.name} ({r.start})' for r in chosen.itertuples()]
axs[1].barh(y-.16,chosen.transition_projection,.32,label='All 10 features',color='#087e8b')
axs[1].barh(y+.16,chosen.projection_no_liveness,.32,label='Without liveness',color='#d45a35')
axs[1].set_yticks(y,labels,fontsize=9);axs[1].invert_yaxis();axs[1].legend(fontsize=8)
axs[1].set(xlabel='Projection (SD; dimension-adjusted)',title='Selected cases: direction-specific sensitivity')
export(fig,'06_transition_candidates')

fig,axs=plt.subplots(2,1,figsize=(13,8),layout='constrained')
for ax,key,title in zip(axs,['deviation','projection_component'],['Artist minus preceding same-genre peers (reference SD)','Signed components of transition projection (sum = projection)']):
    matrix=np.array([detail[(detail.artist_id==r.artist_id)&(detail.start==r.start)].set_index('feature').loc[F,key] for r in chosen.itertuples()])
    bound=np.max(np.abs(matrix));im=ax.imshow(matrix,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_xticks(range(10),F,rotation=20,ha='right',fontsize=8);ax.set_yticks(y,labels,fontsize=8);ax.set_title(title,fontsize=11)
    for j in range(len(matrix)):
        for k in range(10):ax.text(k,j,f'{matrix[j,k]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[j,k])>.6*bound else 'black')
    fig.colorbar(im,ax=ax,shrink=.8)
export(fig,'08_transition_feature_links')

# Timing profiles use one fixed, candidate-excluded 1955-59 same-genre baseline.
# Missing catalogue periods stay missing; no claims of invention or exact lead time.
trajectory=[]
fig,axs=plt.subplots(3,2,figsize=(13,9),layout='constrained',sharex=True)
for ax,r in zip(axs.flat,chosen.itertuples()):
    i=int(r.artist_id);h=window(1955,1959);h=h[h.genre.eq(r.genre)&h.n_songs.ge(3)].drop(index=i,errors='ignore')
    baseline=h[Z].mean().to_numpy();u=detail[(detail.artist_id==i)&(detail.start==r.start)].set_index('feature').loc[F,'unit_direction'].to_numpy()
    for t in range(1950,1980,5):
        p=window(t,t+4);peers=p[p.genre.eq(r.genre)&p.n_songs.ge(3)].drop(index=i,errors='ignore')
        credits=p.at[i,'n_songs'] if i in p.index else 0
        value=(p.loc[i,Z].to_numpy(float)-baseline)@u/np.sqrt(10) if credits>=5 and len(h)>=20 else np.nan
        peer=(peers[Z].mean().to_numpy()-baseline)@u/np.sqrt(10) if len(peers)>=20 and len(h)>=20 else np.nan
        trajectory.append([i,r.name,r.start,t,int(credits),len(h),len(peers),value,peer])
    tr=pd.DataFrame(trajectory[-6:],columns=['artist_id','name','selected_start','period_start','credits','baseline_n','peer_n','artist_projection','peer_projection'])
    ax.plot(tr.period_start+2,tr.artist_projection,'o-',label='Artist (>=5 credits)',color='#087e8b')
    ax.plot(tr.period_start+2,tr.peer_projection,'s--',label='Same-genre peers, artist excluded',color='#677782')
    ax.axvspan(r.start,r.stop,color='#ead69c',alpha=.3);ax.axhline(0,color='gray',lw=.5)
    ax.set_title(f'{r.name} | selected {r.start}-{r.stop}',fontsize=10);ax.set_ylabel('Projection from 1955-59 (SD)')
axs[0,0].legend(fontsize=7)
for ax in axs[-1]:ax.set_xlabel('Recording-period midpoint')
export(fig,'09_transition_timing')
pd.DataFrame(trajectory,columns=['artist_id','name','selected_start','period_start','credits','baseline_n','peer_n','artist_projection','peer_projection']).to_csv(OUT/'transition_timing.csv',index=False)

# Reuse the exact stored matched targets/controls, but project onto the system
# transition instead of each artist's arbitrary historical deviation.
early_profiles=early_solo_profiles();transition_pairs=[]
for r in chosen.itertuples():
    i=int(r.artist_id);t=int(r.start)
    u=detail[(detail.artist_id==i)&(detail.start==t)].set_index('feature').loc[F,'unit_direction'].to_numpy()
    source=solo[solo.artist_id.eq(i)&solo.year.between(t,t+4)][Z].mean().to_numpy()
    for q in alt[(alt.artist_id==i)&(alt.source_window_start==t)].itertuples():
        ids=list(map(int,q.control_ids.split(';')));target=early_profiles.loc[q.follower_id,Z].to_numpy(float)
        cv=early_profiles.loc[ids,Z].to_numpy()
        real=abs((target-source)@u)/np.sqrt(10);control=np.abs((cv-source)@u).mean()/np.sqrt(10)
        transition_pairs.append([i,t,q.follower_id,q.controls,q.control_ids,real,control,control-real])
tp=pd.DataFrame(transition_pairs,columns=['artist_id','start','follower_id','controls','control_ids','source_projection_distance','control_projection_distance','transition_distance_advantage'])
tp.to_csv(OUT/'transition_followers.csv',index=False)
agg=tp.groupby(['artist_id','start']).agg(matched=('follower_id','size'),transition_advantage=('transition_distance_advantage','mean')).reset_index()
linked=chosen.merge(agg,on=['artist_id','start'],how='left');linked['matched']=linked.matched.fillna(0).astype(int)
linked['reported_transition_advantage']=linked.transition_advantage.where(linked.matched.ge(5))
linked.to_csv(OUT/'transition_evidence.csv',index=False)
# Endpoint persistence is descriptive: retention does not by itself reject a pre-existing trend.
m=five.loc[1970,Z].to_numpy()-five.loc[1960,Z].to_numpy();u=m/np.linalg.norm(m)
persistence=[]
for t in range(1950,1985,5):
    d=five.loc[t,Z].to_numpy()-five.loc[1960,Z].to_numpy()
    persistence.append([t,*d,d@u/np.sqrt(10),d@m/(m@m)])
pd.DataFrame(persistence,columns=['start',*Z,'transition_projection','retention_vs_1970']).to_csv(OUT/'transition_persistence.csv',index=False)
print('Revolution-linked selection:',linked[['name','start','transition_projection','novelty_pct','later_followers','matched','reported_transition_advantage']].to_string(index=False))
