from csv_store import read_csv
"""R7 descriptive cohort signatures, grounded in the formal Phase 2 network."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mat3300-mpl')
import matplotlib
matplotlib.use('Agg')
from common import *
DEST=ROOT/'output/r7';DEST.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,'grid.alpha':.2})
e=edges[edges.influencer_main_genre.ne('Unknown')&edges.follower_main_genre.ne('Unknown')].copy()
assert len(edges)==42770 and len(e)==42759 and not e.duplicated(['influencer_id','follower_id']).any()
e['cross']=e.influencer_main_genre.ne(e.follower_main_genre)
periods=list(range(1950,2020,10));rows=[];fr=[]
for b,g in e.groupby('follower_active_start'):
    m=len(g);so=g.groupby('influencer_main_genre').size();ta=g.groupby('follower_main_genre').size()
    expected=m-sum(so.get(k,0)*n/m for k,n in ta.items())
    counts=g.groupby('influencer_id').size()
    gaps=(g.follower_active_start-g.influencer_active_start)/10
    for j,q in g.groupby('follower_id'):
        ps=q.influencer_main_genre.value_counts(normalize=True)
        fr.append([j,b,q.follower_main_genre.iloc[0],len(q),q.cross.mean(),np.exp(-(ps*np.log(ps)).sum())])
    rows.append([b,m,g.follower_id.nunique(),g.influencer_id.nunique(),g.cross.mean(),expected/m,g.cross.sum()/expected if expected else np.nan,counts.nlargest(10).sum()/m,(counts*(counts-1)).sum()/(m*(m-1)) if m>1 else np.nan,gaps[gaps>=0].median(),(gaps<0).mean()])
c=pd.DataFrame(rows,columns=['cohort','edges','followers','sources','cross_edge_share','degree_expected_cross_share','cross_observed_expected','top10_share','source_pair_concentration','median_nonnegative_gap_decades','negative_gap_share']).set_index('cohort')
f=pd.DataFrame(fr,columns=['follower_id','cohort','genre','known_incoming','cross_share','effective_source_genres'])
support=f.groupby(['cohort','genre']).size().unstack(fill_value=0).reindex(periods,fill_value=0)
genres=support.columns[support.min()>=10].tolist()
assert len(genres)>=2
weights=support[genres].sum()/support[genres].to_numpy().sum()
selected=f[f.cohort.isin(periods)&f.genre.isin(genres)]
means=selected.groupby(['cohort','genre']).cross_share.mean().unstack()
c['follower_equal_cross_share']=f.groupby('cohort').cross_share.mean()
c['common_genres_observed']=selected.groupby('cohort').cross_share.mean()
c['common_genres_fixed']=means.mul(weights).sum(axis=1)
c['source_diversity_3to5']=f[f.known_incoming.between(3,5)].groupby('cohort').effective_source_genres.mean()
c['source_diversity_n']=f[f.known_incoming.between(3,5)].groupby('cohort').size()
c.to_csv(DEST/'cohort_signatures.csv');f.to_csv(DEST/'follower_signatures.csv',index=False)
weights.rename('fixed_weight').to_csv(DEST/'fixed_genre_weights.csv');support.to_csv(DEST/'genre_cohort_support.csv')
assert np.isclose(weights.sum(),1) and c.edges.sum()==len(e)
assert c.cross_edge_share.between(0,1).all() and c.source_pair_concentration.between(0,1).all()
v=c.loc[periods]
fig,ax=plt.subplots(3,2,figsize=(14,12),layout='constrained',sharex=True)
a=ax[0,0];a.plot(periods,100*v.cross_edge_share,'o-',label='Observed cross-genre edges');a.plot(periods,100*v.degree_expected_cross_share,'s--',label='Degree-based reference');a.set_title('Cross-genre mixing vs connection opportunity');a.set_ylabel('Edges (%)');a.legend(fontsize=8)
a=ax[0,1];a.plot(periods,100*v.common_genres_observed,'o-',label='Observed target-genre weights');a.plot(periods,100*v.common_genres_fixed,'s--',label='Fixed target-genre weights');a.set_title(f'Mixing among the same {len(genres)} target genres');a.set_ylabel('Mean follower cross-genre share (%)');a.legend(fontsize=8)
a=ax[1,0];a.plot(periods,v.source_diversity_3to5,'o-',color='#137f88');a.set_title('Source diversity | followers with 3–5 sources');a.set_ylabel('Mean effective number of source genres')
for b,r in v.iterrows():a.annotate(f'n={int(r.source_diversity_n)}',(b,r.source_diversity_3to5),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
a.margins(y=.3)
a=ax[1,1];a.plot(periods,100*v.source_pair_concentration,'o-',color='#7654a3');a.set_title('Concentration of recorded source connections');a.set_ylabel('Two distinct edges share a source (%)')
a=ax[2,0];a.plot(periods,100*v.top10_share,'o-',color='#d66139');a.set_title('Share of edges from the top 10 sources');a.set_ylabel('Incoming edges (%)')
a=ax[2,1];a.plot(periods,v.followers,'o-',label='Followers with known-genre edges');a.plot(periods,v.sources,'s--',label='Distinct sources');a.set_title('Coverage changes across cohorts');a.set_ylabel('Artists');a.legend(fontsize=8)
for a in ax.flat:a.set_xticks(periods,[f'{b}s' for b in periods],rotation=30)
for a in ax[-1]:a.set_xlabel('Follower career-start cohort')
fig.suptitle('R7 | Observable network signatures, not identified effects of external events',fontsize=17)
for ext in ['png']:fig.savefig(DEST/f'network_signatures.{ext}',dpi=170,bbox_inches='tight')
plt.close(fig)
print('Common genres:',genres);print(v.round(4).to_string());print('All assertions passed.')
# Sensitivity to non-later source eligibility, keeping the cohort comparison explicit.
chron=e[e.influencer_active_start.le(e.follower_active_start)]
cs=chron.groupby('follower_active_start').agg(edges=('cross','size'),cross_share=('cross','mean'))
cs.to_csv(DEST/'nonlater_source_sensitivity.csv',index_label='cohort')
r4=read_csv(BASE/'r4/followers.csv').join(nodes[['active_start','genre']],on='follower')
r4.groupby('active_start').agg(matchable_followers=('follower','size'),mean_lifetime_excess_similarity=('similarity','mean')).to_csv(DEST/'r4_cohort_context.csv',index_label='cohort')
assert len(r4)==2600 and np.isclose(r4.similarity.mean(),.034126,atol=.000001)


# Country: attributed source traditions and career-cohort gaps, two distinct weightings.
ce=edges[edges.follower_main_genre.eq('Country')].copy()
ce['gap']=(ce.follower_active_start-ce.influencer_active_start)//10
ce['gap_group']=np.select([ce.gap.lt(0),ce.gap.eq(0),ce.gap.eq(1),ce.gap.eq(2)],['Later source cohort','Same cohort','One cohort earlier','Two cohorts earlier'],default='Three+ cohorts earlier')
source_counts=ce.groupby(['follower_active_start','influencer_main_genre']).size().unstack(fill_value=0).reindex(periods,fill_value=0)
source_share=source_counts.div(source_counts.sum(axis=1),axis=0)
source_counts.to_csv(DEST/'country_source_counts.csv',index_label='target_cohort')
source_share.to_csv(DEST/'country_source_shares.csv',index_label='target_cohort')
follower_genres=ce.groupby(['follower_active_start','follower_id','influencer_main_genre']).size().unstack(fill_value=0)
follower_genres=follower_genres.div(follower_genres.sum(axis=1),axis=0)
follower_mean=follower_genres.groupby(level=0).mean().reindex(periods)
follower_mean.to_csv(DEST/'country_follower_equal_sources.csv',index_label='target_cohort')
gap_counts=ce.groupby(['follower_active_start','gap_group']).size().unstack(fill_value=0).reindex(periods,fill_value=0)
gap_order=['Later source cohort','Same cohort','One cohort earlier','Two cohorts earlier','Three+ cohorts earlier']
gap_share=gap_counts.reindex(columns=gap_order,fill_value=0).div(gap_counts.sum(axis=1),axis=0)
gap_share.to_csv(DEST/'country_generation_gaps.csv',index_label='target_cohort')
country_support=pd.DataFrame({'network_nodes':nodes[nodes.genre.eq('Country')].groupby('active_start').size(),'linked_targets':ce.groupby('follower_active_start').follower_id.nunique(),'edges':ce.groupby('follower_active_start').size()}).reindex(periods)
country_support['edge_external']=1-source_share.Country
country_support['follower_equal_external']=1-follower_mean.Country
country_support['two_or_more_cohorts_earlier']=gap_share['Two cohorts earlier']+gap_share['Three+ cohorts earlier']
country_support.to_csv(DEST/'country_cultural_support.csv',index_label='target_cohort')
assert np.allclose(source_share.sum(axis=1),1) and np.allclose(follower_mean.sum(axis=1),1) and np.allclose(gap_share.sum(axis=1),1)
fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
main=source_counts.sum().sort_values(ascending=False).head(5).index.tolist()
cols=['#137f88','#d66139','#7654a3','#b58a17','#3267a8','#b7c2c9']
for a,df,title in [(axs[0,0],source_share,'Source traditions | edge-weighted'),(axs[0,1],follower_mean,'Source traditions | equal weight per linked follower')]:
    plot=df[main].copy();plot['Other']=1-plot.sum(axis=1)
    bottom=np.zeros(len(periods))
    for g,col in zip(plot.columns,cols):
        a.bar(periods,100*plot[g],bottom=100*bottom,width=7,color=col,label=g);bottom+=plot[g].to_numpy()
    a.set(title=title,ylabel='Source share (%)',ylim=(0,105))
axs[0,0].legend(fontsize=8,ncol=3,loc='lower left')
bottom=np.zeros(len(periods))
for g,col in zip(gap_order,['#b7c2c9','#b58a17','#3267a8','#137f88','#7654a3']):
    axs[1,0].bar(periods,100*gap_share[g],bottom=100*bottom,width=7,color=col,label=g);bottom+=gap_share[g].to_numpy()
axs[1,0].set(title='Transmission across career-start cohorts',ylabel='Incoming edge share (%)',ylim=(0,105));axs[1,0].legend(fontsize=7,ncol=2,loc='lower left')
axs[1,1].plot(periods,100*country_support.edge_external,'o-',label='External sources: edge-weighted')
axs[1,1].plot(periods,100*country_support.follower_equal_external,'s--',label='External sources: follower-equal')
for b,r in country_support.iterrows():axs[1,1].annotate(f'{int(r.linked_targets)}/{int(r.network_nodes)} linked',(b,100*r.follower_equal_external),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=7)
axs[1,1].set(title='External connections and target coverage',ylabel='Source share (%)');axs[1,1].legend(fontsize=8);axs[1,1].margins(y=.35)
for a in axs.flat:a.set_xticks(periods,[f'{p}s' for p in periods],rotation=30);a.set_xlabel('Country follower career-start cohort')
fig.savefig(DEST/'country_cultural_structure.png',dpi=175,bbox_inches='tight');plt.close(fig)

# Sonic variance: exact within/between genre decomposition, common genre support.
# Unlike the old mean RMS radius, mean squared distance admits an additive identity.
ps={t:window(t,t+4) for t in range(1960,2020,5)}
ps={t:p[p.n_songs.ge(5)&p.genre.ne('Unknown')&p.index.isin(nodes.index[nodes.eligible])] for t,p in ps.items()}
counts=pd.DataFrame({t:p.groupby('genre').size() for t,p in ps.items()}).T.fillna(0)
common_genres=counts.columns[counts.min().ge(10)].tolist()
assert len(common_genres)>=2
fixed_w=counts[common_genres].sum()/counts[common_genres].to_numpy().sum()
fixed_w.rename('fixed_weight').to_csv(DEST/'audio_fixed_weights.csv')
counts.to_csv(DEST/'audio_genre_period_support.csv',index_label='period_start')
vr=[];gr=[]
for t,p in ps.items():
    for kind,selected in [('all_eligible',p),('common_observed',p[p.genre.isin(common_genres)]),('common_fixed',p[p.genre.isin(common_genres)])]:
        mu=selected.groupby('genre')[Z].mean()
        weights=fixed_w if kind=='common_fixed' else selected.groupby('genre').size()/len(selected)
        center=mu.mul(weights,axis=0).sum().to_numpy()
        within=between=total=0.
        for genre,q in selected.groupby('genre'):
            x=q[Z].to_numpy();m=mu.loc[genre].to_numpy();w=float(weights[genre])
            vg=np.mean((x-m)**2);bg=np.mean((m-center)**2)
            within+=w*vg;between+=w*bg;total+=w*np.mean((x-center)**2)
            gr.append([t,kind,genre,len(q),w,vg,bg])
        assert np.isclose(total,within+between,atol=1e-12)
        vr.append([t,kind,len(selected),len(weights),within,between,total,between/total])
variance=pd.DataFrame(vr,columns=['period_start','population','artists','genres','within','between','total','between_fraction'])
variance.to_csv(DEST/'sonic_variance_decomposition.csv',index=False)
pd.DataFrame(gr,columns=['period_start','population','genre','artists','weight','within_variance','center_squared_distance']).to_csv(DEST/'sonic_genre_components.csv',index=False)
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained',sharex=True)
for a,key,title in [(axs[0,0],'total','Total squared feature dispersion'),(axs[0,1],'within','Within-genre dispersion'),(axs[1,0],'between','Between-genre dispersion')]:
    for pop,label,col,style in [('common_observed','Common genres: observed weights','#d66139','-'),('common_fixed','Same genres: fixed weights','#137f88','--')]:
        q=variance[variance.population.eq(pop)];a.plot(q.period_start+2,q[key],'o'+style,color=col,label=label,ms=4)
    a.set(title=title,ylabel='Mean squared reference SD')
axs[0,0].legend(fontsize=8)
q=variance[variance.population.eq('common_observed')];axs[1,1].bar(q.period_start+2,q.artists,width=3,color='#b7c2c9');axs[1,1].set(title=f'Coverage | {len(common_genres)} common genres',ylabel='Eligible artists')
for a in axs.flat:a.set_xlabel('Recording five-year bin midpoint')
fig.suptitle('Sonic differences within and between genres | Total = Within + Between',fontsize=15)
fig.savefig(DEST/'sonic_variance_decomposition.png',dpi=175,bbox_inches='tight');plt.close(fig)

# Systematic genre-pair comparison with an exact ordered-cohort-pair opportunity baseline.
known=edges[edges.influencer_main_genre.ne('Unknown')&edges.follower_main_genre.ne('Unknown')].copy()
observed=known.groupby(['influencer_main_genre','follower_main_genre']).size()
expected={}
for (a,b),q in known.groupby(['influencer_active_start','follower_active_start']):
    so=q.influencer_main_genre.value_counts();ta=q.follower_main_genre.value_counts();m=len(q)
    for g,k in so.items():
        for h,l in ta.items():expected[(g,h)]=expected.get((g,h),0)+k*l/m
so=known.influencer_main_genre.value_counts();ta=known.follower_main_genre.value_counts()
audio=read_csv(BASE/'r3/genre_pairs.csv')
pg=[]
for r in audio.itertuples(index=False):
    m=int(observed.get((r.source,r.target),0));ex=expected.get((r.source,r.target),0.)
    un=so.get(r.source,0)*ta.get(r.target,0)/len(known)
    reliable=(r.source!=r.target and r.n_audio_g>=20 and r.n_audio_h>=20 and m>=10 and ex>=5)
    pg.append([r.source,r.target,r.n_audio_g,r.n_audio_h,r.similarity,m,un,ex,m/un if un else np.nan,m/ex if ex else np.nan,reliable])
pairtable=pd.DataFrame(pg,columns=['source','target','source_audio_artists','target_audio_artists','audio_similarity','edges','unstratified_expected','cohort_stratified_expected','unstratified_ratio','cohort_stratified_ratio','reliable'])
pairtable.to_csv(DEST/'genre_pair_comparison.csv',index=False)
assert np.isclose(sum(expected.values()),len(known))
rel=pairtable[pairtable.reliable]
from scipy.stats import spearmanr
rho=float(spearmanr(rel.audio_similarity,rel.cohort_stratified_ratio).statistic)
fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
ax.scatter(rel.audio_similarity,rel.cohort_stratified_ratio,s=15+3*np.sqrt(rel.edges),alpha=.55,c='#137f88',edgecolors='white',linewidth=.5)
for source,target in [('Country','Folk'),('Blues','Country')]:
    q=pairtable[(pairtable.source==source)&(pairtable.target==target)].iloc[0]
    ax.scatter(q.audio_similarity,q.cohort_stratified_ratio,s=70,marker='o' if q.reliable else 'x',color='#d66139')
    ax.annotate(f'{source} → {target}\n{q.edges} edges'+(' | below support threshold' if not q.reliable else ''),(q.audio_similarity,q.cohort_stratified_ratio),xytext=(8,8),textcoords='offset points',fontsize=8)
ax.axhline(1,color='gray',ls='--',lw=1)
ax.set(yscale='log',xlabel='Genre audio similarity (fixed 10-feature metric)',ylabel='Observed / career-cohort-stratified expected edges',title=f'Recorded lineage and sonic proximity | {len(rel)} supported directed pairs, Spearman r={rho:.3f}')
fig.savefig(DEST/'genre_lineage_vs_sound.png',dpi=175,bbox_inches='tight');plt.close(fig)
import json
(DEST/'summary.json').write_text(json.dumps({'common_audio_genres':common_genres,'reliable_directed_genre_pairs':len(rel),'spearman_audio_vs_stratified_enrichment':rho},indent=2))
print('R7: source shares, cohort gaps, exact variance decomposition and cohort-stratified reference verified.')
print(country_support.round(4).to_string());print(variance[variance.population.eq('common_fixed')].round(4).to_string(index=False));print('Supported pair correlation:',rho)
