"""Original vector illustrations and data figures generated from saved results."""
from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrowPatch, Arc

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
 'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#74808c','axes.labelcolor':'#243646','text.color':'#243646',
 'xtick.color':'#243646','ytick.color':'#243646','pdf.fonttype':42,'svg.fonttype':'none'})
INK='#243646'; TEAL='#148486'; GOLD='#dd9a32'; BLUE='#4080b6'; RED='#b65558'; PALE='#e8f2f4'; PURPLE='#7965a4'

def load(path):return json.loads(Path(path).read_text())
def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def arrow(ax,a,b,color=GOLD,lw=1.7,style='-|>',rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=12,lw=lw,color=color,connectionstyle=f'arc3,rad={rad}'))
def person(ax,x,y,scale=.6):
    ax.add_patch(Circle((x,y),.12*scale,color=INK))
    ax.plot([x,x],[y-.16*scale,y-.75*scale],color=INK,lw=2)
    ax.plot([x-.25*scale,x,x+.23*scale],[y-.45*scale,y-.28*scale,y-.45*scale],color=INK,lw=1.5)
    ax.plot([x-.23*scale,x,x+.23*scale],[y-1.2*scale,y-.75*scale,y-1.2*scale],color=INK,lw=1.5)
def finish(ax,xlab=None):
    ax.set_axisbelow(True);ax.grid(axis='x',color='#e4e9ee',lw=.7)
    if xlab:ax.set_xlabel(xlab)

def street(data,out):
    route=load(data/'route.json'); modes=load(data/'modes.json')['rows']; design=load(data/'designs.json')['rows']
    fig,axes=plt.subplots(1,5,figsize=(12,4.5),gridspec_kw={'width_ratios':[1,1.2,1.55,1.4,1.25]})
    short=['Residential lane','Mixed use collector','Wide arterial','Concrete underpass','Office canyon']
    for i,(ax,s) in enumerate(zip(axes,route['sections'])):
        w=s['width_m'];hl=s['left_height_m'];hr=s['right_height_m'];x=s['eye_offset_m']
        ax.set_facecolor('#f2f8fb');ax.add_patch(Rectangle((-w/2-5,0),5,hl,fc='#bfcad3',ec=INK,lw=.7))
        ax.add_patch(Rectangle((w/2,0),5,hr,fc='#91a8b7',ec=INK,lw=.7))
        for h in np.arange(4,max(hl,hr),6):
            if h<hl: ax.plot([-w/2-4,-w/2-1],[h,h],color='white',lw=1)
            if h<hr: ax.plot([w/2+1,w/2+4],[h,h],color='white',lw=1)
        ax.plot([-w/2,w/2],[0,0],color=INK,lw=2)
        canopy=s['canopy'];opaque=canopy.get('transmittance')=='opaque'
        if opaque: ax.add_patch(Rectangle((-w/2,canopy['base_m']),w,3,fc='#89959b',alpha=.9))
        else:
            ax.add_patch(Rectangle((-w/2,canopy['base_m']),w,canopy['top_m']-canopy['base_m'],fc='#77ab85',alpha=.35))
            arrow(ax,(0,40),(x,4),BLUE,lw=1.3)
        ax.scatter([x],[1.55],c=RED,s=25,zorder=5)
        ax.annotate(f'{w:g} m',xy=(0,-3),ha='center',va='top',fontsize=10,fontweight='bold')
        ax.text(0,-10,f"{s['length_m']/1000:g} km\n{hl:g} / {hr:g} m height",ha='center',va='top',fontsize=9)
        ax.text(0,59,f'{i+1}  {short[i]}',ha='center',fontsize=10,fontweight='bold')
        ax.text(0,52,f"{canopy['cover']:.0%} "+('opaque cover' if opaque else 'foliage cover'),ha='center',fontsize=8.5)
        ax.set_xlim(-w/2-6,w/2+6);ax.set_ylim(-22,65);ax.axis('off')
    fig.text(.5,.025,'Section dimensions are model inputs • red point = eye position • blue arrow = diffuse sky path\nFive sequential street types, total 4.97 km; this is a schematic, not a surveyed route map.',ha='center',fontsize=10)
    fig.subplots_adjust(wspace=.28,bottom=.2)
    save(fig,out,'figure_1_street_sections')

    labels={'walk':'Walk','rickshaw':'Rickshaw','cng':'CNG auto-rickshaw','car_clear':'Clear-glazed car','car_tinted':'Bronze-glazed car','bus':'Bus window seat','car_film':'Car with film'}
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True)
    y=np.arange(len(modes));colors=[TEAL if r['key']=='walk' else BLUE for r in modes]
    for ax,key,title,unit in [(axes[0],'mean_medi_travel_lx','A  Average light level at the eye','Mean melanopic EDI (lx)'),(axes[1],'dose_travel_lxh','B  Light accumulated during the trip','Melanopic dose (lx h)')]:
        vals=[r['summary'][key] for r in modes]
        ax.barh(y,vals,color=colors,height=.64)
        for yy,v in zip(y,vals):ax.text(v+max(vals)*.025,yy,f'{v:,.0f}',va='center',fontsize=9)
        ax.set_xlim(0,max(vals)*1.23);ax.set_title(title,loc='left',pad=14);finish(ax,unit)
    axes[0].set_yticks(y,[labels[r['key']] for r in modes]);axes[0].invert_yaxis()
    fig.tight_layout(w_pad=2);save(fig,out,'figure_2_mode_exposure')

    ds=[r for r in design if r['key']!='baseline']
    ds.sort(key=lambda r:r['delta_vs_baseline_pct'])
    dn={'pale_facades':'Whitewashed façades','setback_low':'Building height capped at 18 m','widen_18':'Minimum street width 18 m','pale_paving':'Asphalt replaced by pale concrete','canopy_40':'Foliage cover 40%','canopy_70':'Foliage cover 70%'}
    fig,ax=plt.subplots(figsize=(10,4.1));vals=[r['delta_vs_baseline_pct'] for r in ds];y=np.arange(len(ds))
    ax.barh(y,vals,color=[TEAL if v>=0 else GOLD for v in vals],height=.65)
    ax.set_yticks(y,[dn[r['key']] for r in ds]);ax.axvline(0,color=INK,lw=.9)
    for yy,v in zip(y,vals):ax.text(v+3 if v>0 else v/2,yy,f'{v:+.1f}%',ha='left' if v>0 else 'center',va='center',fontweight='bold',color=INK if v>0 else 'white')
    ax.set_xlim(-57,170);finish(ax,'Change in walking melanopic dose relative to baseline (%)')
    fig.tight_layout();save(fig,out,'figure_3_design_changes')

    rows=[r for r in modes if r['key']!='walk'];y=np.arange(len(rows))
    dur=[100*r['decomposition']['duration_share_of_gap'] for r in rows]
    fig,ax=plt.subplots(figsize=(10,3.8));ax.barh(y,dur,color=GOLD,label='Duration contribution')
    ax.barh(y,100-np.array(dur),left=dur,color=BLUE,label='Exposure rate contribution')
    for i,v in enumerate(dur):ax.text(v/2,i,f'{v:.1f}%',ha='center',va='center',color=INK);ax.text(v+(100-v)/2,i,f'{100-v:.1f}%',ha='center',va='center',color='white')
    ax.set_yticks(y,[labels[r['key']] for r in rows]);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel('Share of the dose deficit relative to walking (%)');ax.legend(ncol=2,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False)
    fig.tight_layout();save(fig,out,'figure_4_dose_decomposition')

def room_diagram(out,name='figure_1_office_geometry',spec=False):
    fig,(ax,plan)=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1.6,1]})
    ax.add_patch(Rectangle((0,0),8,3,fc='#f5f7f7',ec=INK,lw=1.5))
    ax.plot([0,0],[.8,2.7],color='#63abc9',lw=6);ax.plot([0,0],[1.75,2.7],color=PURPLE,lw=8,alpha=.8)
    ax.add_patch(Polygon([(0,2.7),(5.8,0),(0,0)],fc='#f8d899',alpha=.38))
    ax.add_patch(Rectangle((2.1,.72),1.0,.08,color='#998273'))
    ax.plot([2.2,2.2],[0,.72],color='#998273',lw=2)
    person(ax,3.15,1.2,.6);ax.plot([2.95,3.45],[.52,.52],color=INK,lw=1.7)
    arrow(ax,(-1,2.5),(2.8,1.22),GOLD);arrow(ax,(.1,1.1),(5.4,2.9),BLUE)
    arrow(ax,(5.4,2.9),(3.25,1.25),BLUE)
    ax.text(-.7,3.25,'Daylight',fontsize=11,color=BLUE)
    ax.annotate('Clear glazing\n0.8–2.7 m',xy=(0,1.0),xytext=(-1.55,.05),fontsize=9,arrowprops={'arrowstyle':'-','color':INK})
    ax.annotate('Roller shade\n50% illustrated',xy=(0,2.3),xytext=(.65,3.35),fontsize=9,arrowprops={'arrowstyle':'-','color':INK})
    ax.annotate('Eye at 1.2 m',xy=(3.15,1.2),xytext=(4.9,1.1),fontsize=9,arrowprops={'arrowstyle':'-','color':INK})
    ax.text(4,-.43,'8 m room depth',ha='center',fontsize=10)
    arrow(ax,(0,-.2),(8,-.2),INK,lw=.8,style='<->')
    ax.text(5.15,2.35,'Reflected light',color=BLUE,fontsize=9)
    ax.text(1,.22,'Direct / transmitted path',color='#96681b',fontsize=9)
    ax.set_xlim(-1.8,8.5);ax.set_ylim(-.8,4);ax.set_aspect('equal');ax.axis('off');ax.set_title('A  Room section',loc='left')
    plan.add_patch(Rectangle((0,0),6,8,fc='#f3f7f8',ec=INK,lw=1.5))
    plan.plot([.45,5.55],[0,0],lw=5,color='#63abc9');plan.text(3,-.9,'5.1 m wide window',ha='center',fontsize=9)
    depths=[3] if spec else [2.5,4.5,6.5]
    for d in depths:
        plan.add_patch(Rectangle((2.3,d-.25),1.4,.5,fc='#d2bca5',ec='white'))
        plan.scatter([3],[d],color=RED,s=40,zorder=5)
        if spec:
            for th in np.arange(0,360,45):
                rad=np.radians(th);arrow(plan,(3,d),(3+.85*np.sin(rad),d+.85*np.cos(rad)),TEAL,lw=.9)
        else:arrow(plan,(3,d),(3.8,d-.8),TEAL,lw=1.5)
        plan.text(6.2,d,f'{d:g} m',va='center',fontsize=9)
    plan.text(3,8.45,'6 m room width',ha='center',fontsize=10)
    plan.text(3,7.75,'Eight view directions' if spec else 'Three desk depths',ha='center',fontsize=9)
    plan.set_xlim(-.5,7.7);plan.set_ylim(-1.2,9.2);plan.set_aspect('equal');plan.axis('off');plan.set_title('B  Plan and eye position',loc='left')
    fig.text(.5,.015,'Model schematic; ray paths illustrate mechanisms and are not a luminance rendering.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1]);save(fig,out,name)

def shade(data,out):
    room_diagram(out)
    e1=load(data/'e1.json');e3=load(data/'e3.json');e2=load(data/'e2.json')
    keys=['always_open','thermal_only','glare_only','thermal_and_glare','circadian_aware','haldi_stochastic']
    labels=['Always open','Solar trigger','Glare trigger','Glare or solar trigger','Melanopic-aware','Illustrative stochastic']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.6),sharey=True);y=np.arange(len(keys))
    a=[e1['by_controller'][k]['medi_retained'] for k in keys]
    vals=np.array([v['mean'] for v in a])*100;lo=vals-np.array([v['lo'] for v in a])*100;hi=np.array([v['hi'] for v in a])*100-vals
    axes[0].barh(y,vals,color=TEAL,height=.62);axes[0].errorbar(vals,y,xerr=[lo,hi],fmt='none',ecolor=INK,capsize=3,lw=1)
    axes[0].set_xlim(0,110);axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_title('A  Mean mEDI retained',loc='left');finish(axes[0],'% of the always-open value')
    for i,v in enumerate(vals):axes[0].text(3,i,f'{v:.1f}%',va='center',color='white',fontsize=9)
    glare=[e1['by_controller'][k]['dgp_delta_pp']['mean'] for k in keys]
    threshold=[e1['by_controller'][k]['compliance_delta_pp']['mean'] for k in keys]
    axes[1].barh(y-.16,glare,height=.29,color=BLUE,label='DGP ≥ 0.35 time')
    axes[1].barh(y+.16,threshold,height=.29,color=GOLD,label='mEDI ≥ 250 lx time')
    axes[1].set_xlim(-10.5,1);axes[1].set_title('B  Change in threshold time',loc='left');finish(axes[1],'Difference from always open (percentage points)')
    axes[1].legend(frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(0,-.16))
    fig.tight_layout();save(fig,out,'figure_2_blind_tradeoffs')

    fig,axes=plt.subplots(1,2,figsize=(10,4.8),subplot_kw={'projection':'polar'})
    names={'always_open':'Always open','thermal_and_glare':'Glare or solar trigger','haldi_stochastic':'Illustrative stochastic','circadian_aware':'Melanopic-aware'}
    for r,c in zip(e3['rows'],[BLUE,TEAL,GOLD,PURPLE]):
        vals=sorted(r['values'],key=lambda v:v['heading_deg']);theta=np.radians([v['heading_deg'] for v in vals]+[vals[0]['heading_deg']])
        for ax,k in [(axes[0],'medi_mean'),(axes[1],'mean_coverage')]:
            yy=[v[k] for v in vals]+[vals[0][k]];ax.plot(theta,yy,color=c,lw=1.8,marker='o',ms=2.5,label=names[r['controller']])
    for ax in axes:ax.set_theta_zero_location('N');ax.set_theta_direction(-1);ax.grid(color='#d7e0e5');ax.tick_params(labelsize=9)
    axes[0].set_title('A  Mean melanopic EDI (lx)',pad=20);axes[0].set_ylim(0,4500);axes[0].set_yticks([1000,2000,3000,4000])
    axes[1].set_title('B  Mean shade coverage',pad=20);axes[1].set_ylim(0,.75);axes[1].set_yticks([.25,.5,.75]);axes[1].set_yticklabels(['25%','50%','75%'])
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='lower center',frameon=False,fontsize=9)
    fig.tight_layout(rect=[0,.13,1,1]);save(fig,out,'figure_3_view_direction')

    fig,(ax,b)=plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'width_ratios':[1.5,1]})
    grid=np.array([[next(c['aggregates']['medi_mean'] for c in e2['cells'] if c['encoding']==e and c['optimiser']==o) for o in e2['optimisers']] for e in e2['encodings']])
    ax.imshow(grid,cmap=matplotlib.colors.ListedColormap([PALE]),aspect='auto')
    ax.set_yticks(range(len(e2['encodings'])),[s.replace('_',' ') for s in e2['encodings']],fontsize=9)
    ax.set_xticks(range(len(e2['optimisers'])),['Random','Latin hypercube','Coordinate','Annealing'],fontsize=9,rotation=15)
    for (i,j),v in np.ndenumerate(grid):ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=9)
    ax.set_title('A  Mean mEDI in every optimisation cell (lx)',loc='left',fontsize=11)
    b.axis('off');b.text(.05,.86,'24 cells • one outcome',fontsize=15,fontweight='bold')
    b.text(.05,.56,'Algorithm dispersion = 0\nEncoding dispersion = 0',fontsize=12,linespacing=1.8)
    b.text(.05,.25,'Ratio 0 / 0 is undefined\nNo comparative effect is identifiable',fontsize=11,color=RED,linespacing=1.6)
    fig.tight_layout();save(fig,out,'figure_4_unidentifiable_comparison')

def specdebt(data,out):
    room_diagram(out,'figure_1_measurement_choices',spec=True)
    s=load(data/'specdebt_results.json')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for name,ax,col,title in [('melanopic',axes[0],TEAL,'A  Daylight threshold test'),('benchmark',axes[1],PURPLE,'B  Synthetic scorer comparison')]:
        with (data/(name+'_specifications.csv')).open() as f:rows=list(csv.DictReader(f))
        v=np.sort([float(r['value']) for r in rows]);ax.plot(np.arange(1,len(v)+1),v,color=col,lw=2)
        ax.axhline(250 if name=='melanopic' else 0,ls='--',color=RED,lw=1,label='Decision boundary')
        ax.set_title(title,loc='left');ax.set_xlabel('Specifications ordered by value');ax.set_ylabel('Melanopic EDI (lx)' if name=='melanopic' else 'Accuracy A − accuracy B');ax.grid(color='#e5e9ec',lw=.6);ax.legend(frameon=False,fontsize=9)
    fig.tight_layout(w_pad=2);save(fig,out,'figure_2_specification_envelopes')
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    aliases={'view_heading_deg':'View direction','blind_coverage':'Blind coverage','eye_height_m':'Eye height','hour':'Hour','month':'Month','cloud_cover':'Cloud cover','action_spectrum':'Legacy scale','extraction':'Extraction','abstention':'Abstention','subsample_seed':'Subsample seed','normalisation':'Normalisation','max_chars':'Truncation'}
    for name,ax,col,title in [('melanopic',axes[0],TEAL,'A  Daylight factors'),('benchmark',axes[1],PURPLE,'B  Scoring factors')]:
        d=s[name];keys=list(d['attribution_value']);y=np.arange(len(keys))
        ax.barh(y-.16,[d['attribution_value'][k] for k in keys],height=.3,color=col,label='Value variance')
        ax.barh(y+.16,[d['attribution_decision'][k] for k in keys],height=.3,color=GOLD,label='Decision variance')
        ax.set_yticks(y,[aliases[k] for k in keys],fontsize=9);ax.invert_yaxis();ax.set_title(title,loc='left');finish(ax,'First-order variance fraction');ax.set_xlim(0,1);ax.legend(frameon=False,loc='lower right',fontsize=8)
    fig.tight_layout(w_pad=2);save(fig,out,'figure_3_factor_attribution')
    fig,ax=plt.subplots(figsize=(8.5,4))
    for name,col,label in [('melanopic',TEAL,'Daylight grid'),('benchmark',PURPLE,'Synthetic scoring grid')]:
        c=s[name]['sufficiency_curve'];ax.plot([r['n'] for r in c],[r['residual_flip_rate']*100 for r in c],'-o',color=col,label=label,lw=2)
    ax.axhline(1,ls='--',color=RED,lw=1,label='1% weighted tolerance');ax.set_xticks(range(8));ax.set_ylim(-1,45);ax.set_xlabel('Number of factors declared');ax.set_ylabel('Minimum weighted residual minority share (%)');ax.grid(color='#e5e9ec');ax.legend(frameon=False,fontsize=9)
    fig.tight_layout();save(fig,out,'figure_4_declaration_curve')
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),gridspec_kw={'width_ratios':[1,1.2,1]})
    a,b,c=axes
    a.add_patch(Rectangle((.15,.28),.68,.48,fc=PALE,ec=INK));a.plot([.15,.15],[.36,.65],color=BLUE,lw=5);a.scatter([.53],[.45],s=70,c=RED)
    for th in range(0,360,60):
        z=np.radians(th);arrow(a,(.53,.45),(.53+.17*np.cos(z),.45+.17*np.sin(z)),TEAL,lw=1)
    a.text(.5,.86,'One object',ha='center',fontweight='bold',fontsize=15);a.text(.5,.12,'Room or fixed answer corpus',ha='center',fontsize=10)
    b.text(.5,.86,'Many declared choices',ha='center',fontweight='bold',fontsize=15)
    for i in range(6):
        for j in range(8):b.add_patch(Rectangle((.09+j*.1,.28+i*.07),.08,.052,fc=TEAL if (i+j)%5 else GOLD,ec='none'))
    b.text(.5,.12,'Evaluate every specification',ha='center',fontsize=10)
    c.text(.5,.86,'A reporting requirement',ha='center',fontweight='bold',fontsize=15)
    for name,y,col,n in [('Daylight',.64,TEAL,5),('Scoring',.46,PURPLE,3)]:
        c.text(.02,y+.065,name,fontsize=10)
        for i in range(n):c.add_patch(Rectangle((.05+i*.13,y-.04),.09,.075,fc=col))
    c.text(.46,.27,'5 factors  /  3 factors',ha='center',fontsize=12,fontweight='bold')
    c.text(.46,.12,'Residual minority share ≤ 1%',ha='center',fontsize=10)
    for ax in axes:ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    fig.subplots_adjust(wspace=.23);save(fig,out,'graphical_abstract')

def main(study,data=None,out=None):
    data=Path(data or ROOT/'results'/study);out=Path(out or ROOT/'figures'/study)
    {'streetlux':street,'shadeparadox':shade,'specdebt':specdebt}[study](data,out)
    print('Figures written:',study)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('--data');p.add_argument('--out');a=p.parse_args();main(a.study,a.data,a.out)
