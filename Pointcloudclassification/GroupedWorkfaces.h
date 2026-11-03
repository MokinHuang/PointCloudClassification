#pragma once
#include "AutoWorkfaces.h"
#include <numeric>

// Experimental physical-bench topology. Original feature geometry is never edited.
namespace grouped {
using namespace sections;
struct Options {
    double neighbor=15, elevation=2.5, directionCos=.85, observed=.9, slopeSupport=.65;
    double transverse=4, spanPadding=5, platformReach=10;
    int minSamples=3, minVotes=3, maxCandidates=2048;
};
struct Span {int line;double begin,end;};
struct Group {std::vector<const Sample*> samples;std::vector<Span> crests,toes;double high=0,low=0;};
struct Join {int a,b;};
struct Link {int upper,lower,votes;};
struct Candidate {std::string name;std::vector<int> chain;bool upstreamWeak=false,downstreamWeak=false,upstreamLink=false,downstreamLink=false;};
struct Result {std::vector<Group> groups;std::vector<Join> joins;std::vector<Link> links,weak;std::vector<Candidate> candidates;bool truncated=false;Options options;};
inline const GridCell* lookup(const std::vector<std::vector<GridCell>>& grid,const V& p,double cell,const Eigen::Vector4f& origin){
    int r=int(std::floor((p.y()-origin.y())/cell)),c=int(std::floor((p.x()-origin.x())/cell));
    if(r<0||r>=int(grid.size())||c<0||c>=int(grid[r].size()))return nullptr;return &grid[r][c];
}
inline bool observed(const GridCell* g){return g&&!g->is_empty&&g->count>0&&std::isfinite(g->avg_z);}
inline bool adjacent(const Sample& a,const Sample& b,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin,const Options& o){
    XY ac=xy(a.crest),bc=xy(b.crest),at=xy(a.toe),bt=xy(b.toe),da=at-ac,db=bt-bc;
    if((ac-bc).norm()>o.neighbor||(at-bt).norm()>o.neighbor||da.norm()<cell||db.norm()<cell)return false;
    if(std::abs(a.crest.z()-b.crest.z())>o.elevation||std::abs(a.toe.z()-b.toe.z())>o.elevation)return false;
    da.normalize();db.normalize();if(da.dot(db)<o.directionCos)return false;
    XY displacement=(bc+bt-ac-at)*.5,down=(da+db).normalized();
    if(std::abs(displacement.dot(down))>std::max(o.transverse,displacement.norm()*.35))return false;
    if(crossing(a,b))return false;
    V p=(a.crest+a.toe)*.5,q=(b.crest+b.toe)*.5;int steps=std::max(1,int(std::ceil((xy(p)-xy(q)).norm()/cell))),seen=0,slope=0;
    for(int i=0;i<=steps;++i){auto g=lookup(grid,p+(q-p)*(double(i)/steps),cell,origin);if(observed(g)){++seen;if(g->type==2)++slope;}}
    return double(seen)/(steps+1)>=o.observed&&double(slope)/(steps+1)>=o.slopeSupport;
}
inline std::vector<Span> spans(const std::vector<const Sample*>& samples,bool crest,const std::map<int,const Line*>& lines,const Options& o){
    std::map<int,std::vector<double>> stations;for(auto s:samples)stations[crest?s->crestLine:s->toeLine].push_back(crest?s->crestS:s->toeS);
    std::vector<Span> out;for(auto& entry:stations){auto it=lines.find(entry.first);if(it==lines.end())continue;auto& v=entry.second;std::sort(v.begin(),v.end());
        // Keep separate intervals across unsupported gaps even on the same original ID.
        double start=v[0],end=start;auto emit=[&](){out.push_back({entry.first,std::max(0.,start-o.spanPadding),std::min(it->second->s.back(),end+o.spanPadding)});};
        for(size_t i=1;i<v.size();++i){if(v[i]-end>o.neighbor){emit();start=v[i];}end=v[i];}emit();
    }return out;
}
inline Result discover(const std::vector<Sample>& slopes,const std::vector<platforms::Measurement>& platforms,const std::vector<FeatureLine>& source,
 const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin,const Options& o={}){
    if(cell<=0||o.neighbor<=0||o.minSamples<1||o.minVotes<1||o.maxCandidates<1)throw std::runtime_error("Invalid grouped options");
    Result out;out.options=o;std::vector<const Sample*> s;for(auto& a:slopes)if(a.status=="matched")s.push_back(&a);
    std::vector<int> parent(s.size());std::iota(parent.begin(),parent.end(),0);
    std::function<int(int)> root=[&](int i){return parent[i]==i?i:parent[i]=root(parent[i]);};
    std::map<std::pair<int,int>,std::vector<int>> buckets;
    for(int i=0;i<int(s.size());++i){int x=int(std::floor((s[i]->crest.x()-origin.x())/o.neighbor)),y=int(std::floor((s[i]->crest.y()-origin.y())/o.neighbor));
        for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy){auto it=buckets.find({x+dx,y+dy});if(it==buckets.end())continue;
            for(int j:it->second)if(adjacent(*s[i],*s[j],grid,cell,origin,o)){int a=root(i),b=root(j);if(a!=b){parent[a]=b;out.joins.push_back({s[i]->id,s[j]->id});}}
        }buckets[{x,y}].push_back(i);
    }
    auto local=overall::localLines(source,origin);std::map<int,const Line*> lines;for(auto& l:local)lines[l.id]=&l;
    std::map<int,std::vector<const Sample*>> components;for(int i=0;i<int(s.size());++i)components[root(i)].push_back(s[i]);
    std::map<int,int> owner;for(auto& component:components){if(component.second.size()<size_t(o.minSamples))continue;Group g;g.samples=component.second;
        for(auto a:g.samples){g.high+=a->crest.z();g.low+=a->toe.z();owner[a->id]=int(out.groups.size());}g.high/=g.samples.size();g.low/=g.samples.size();
        g.crests=spans(g.samples,true,lines,o);g.toes=spans(g.samples,false,lines,o);out.groups.push_back(g);
    }
    std::map<std::pair<int,int>,std::set<int>> votes;
    for(auto& pm:platforms){auto& p=pm.geometry;auto own=owner.find(p.id);if(p.status!="matched"||own==owner.end())continue;int d=own->second;
        const Sample* ds=nullptr;for(auto a:out.groups[d].samples)if(a->id==p.id){ds=a;break;}if(!ds)continue;
        XY dd=(xy(ds->toe)-xy(ds->crest)).normalized(),across=xy(p.crest)-xy(p.toe);if(across.norm()<cell)continue;across.normalize();
        int best=-1;double distance=o.platformReach;bool ambiguous=false;
        for(int u=0;u<int(out.groups.size());++u){if(u==d||out.groups[u].high<=out.groups[d].high+1)continue;
            for(auto us:out.groups[u].samples){if(us->toeLine!=p.toeLine)continue;double delta=(xy(us->toe)-xy(p.toe)).norm();if(delta>=o.platformReach)continue;
                XY ud=(xy(us->toe)-xy(us->crest)).normalized();if(ud.dot(dd)<.7||ud.dot(across)<.7||dd.dot(across)<.7)continue;
                if(delta<distance-.001){distance=delta;best=u;ambiguous=false;}else if(best!=u&&std::abs(delta-distance)<.001)ambiguous=true;
            }
        }if(best>=0&&!ambiguous)votes[{best,d}].insert(p.id);
    }
    std::vector<std::vector<int>> next(out.groups.size());std::vector<int> indegree(out.groups.size());
    for(auto& v:votes){Link l{v.first.first,v.first.second,int(v.second.size())};if(l.votes>=o.minVotes){out.links.push_back(l);next[l.upper].push_back(l.lower);++indegree[l.lower];}else out.weak.push_back(l);}
    std::vector<int> path;std::function<void(int)> visit=[&](int n){if(out.candidates.size()>=size_t(o.maxCandidates)){out.truncated=true;return;}
        if(std::find(path.begin(),path.end(),n)!=path.end())throw std::runtime_error("Grouped graph cycle");path.push_back(n);
        // Retain subchains: a link supported elsewhere along a bench cannot force
        // every cross-section to extend through that link.
        if(path.size()>=2){Candidate c;c.name="GROUP_"+std::to_string(out.candidates.size()+1);c.chain=path;c.upstreamLink=indegree[path.front()]>0;c.downstreamLink=!next[n].empty();
            for(auto& w:out.weak){if(w.lower==path.front())c.upstreamWeak=true;if(w.upper==n)c.downstreamWeak=true;}out.candidates.push_back(c);
        }for(int k:next[n])visit(k);path.pop_back();};
    for(int i=0;i<int(next.size());++i)if(!next[i].empty())visit(i);return out;
}
struct Intersection {bool found=false;double distance=0,station=0;int line=-1;V point=V::Zero();};
inline Intersection intersect(const std::vector<Span>& spans,const std::map<int,const Line*>& lines,const XY& start,const XY& direction,double after,double limit){
    Intersection best;best.distance=limit;
    for(auto& span:spans){auto it=lines.find(span.line);if(it==lines.end())continue;auto& l=*it->second;
        for(size_t j=1;j<l.p.size();++j){double t,u;if(!hit(start,direction,xy(l.p[j-1]),xy(l.p[j]),t,u)||t<=after||t>best.distance)continue;
            double station=l.s[j-1]+u*(l.s[j]-l.s[j-1]);if(station<span.begin-1e-6||station>span.end+1e-6)continue;
            best={true,t,station,l.id,l.p[j-1]+u*(l.p[j]-l.p[j-1])};
        }
    }return best;
}
inline std::vector<overall::Preview> measure(const Result& graph,const std::vector<FeatureLine>& source,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin,double rotationDegrees=0,const std::set<int>* only=nullptr){
    auto local=overall::localLines(source,origin);std::map<int,const Line*> lines;for(auto& l:local)lines[l.id]=&l;V offset(origin.x(),origin.y(),0);
    std::vector<overall::Preview> out;
    for(auto& c:graph.candidates){size_t first=out.size();for(auto s:graph.groups[c.chain.front()].samples){overall::Preview a;a.id=int(out.size());a.name=c.name;a.upper=s->crestLine;a.upperType=0;a.lowerType=1;a.station=s->crestS;a.top=s->crest;
        auto finish=[&](){out.push_back(a);};if(only&&!only->count(a.id)){a.status="not_requested";finish();continue;}auto line=lines.find(s->crestLine);if(line==lines.end()){a.status="missing_upper_line";finish();continue;}
        auto& l=*line->second;XY tangent=xy(at(l,std::min(l.s.back(),s->crestS+15)))-xy(at(l,std::max(0.,s->crestS-15)));
        if(tangent.norm()<cell*.25){a.status="unstable_upper_direction";finish();continue;}tangent.normalize();XY dir(-tangent.y(),tangent.x()),down=xy(s->toe)-xy(s->crest);if(dir.dot(down)<0)dir=-dir;
        double angle=rotationDegrees*std::acos(-1.)/180;dir=XY(dir.x()*std::cos(angle)-dir.y()*std::sin(angle),dir.x()*std::sin(angle)+dir.y()*std::cos(angle));
        XY start=xy(a.top-offset);double last=0;bool complete=true;Intersection bottom;
        for(size_t k=0;k<c.chain.size();++k){auto& g=graph.groups[c.chain[k]];
            if(k){auto crest=intersect(g.crests,lines,start,dir,last,1500);if(!crest.found){complete=false;break;}last=crest.distance;}
            bottom=intersect(g.toes,lines,start,dir,last,1500);if(!bottom.found){complete=false;break;}last=bottom.distance;
        }
        if(!complete){a.status="incomplete_chain_intersection";finish();continue;}
        a.hasTarget=true;a.bottom=bottom.point+offset;a.lower=bottom.line;a.width=bottom.distance;
        auto top=lookup(grid,a.top,cell,origin),bot=lookup(grid,a.bottom,cell,origin);
        if(!observed(top)||!observed(bot)){a.status="unobserved_endpoint";finish();continue;}a.top.z()=top->avg_z;a.bottom.z()=bot->avg_z;
        if(a.top.z()-a.bottom.z()<1){a.status="lower_not_below_upper";finish();continue;}
        int seen=0;double gap=0,maxGap=0,runStart=0,minZ=a.top.z();bool on=false,reverse=false;int steps=int(std::ceil(a.width/cell));
        for(int k=0;k<=steps;++k){double d=std::min(a.width,k*cell);V p=a.top;p.head<2>()+=dir*d;auto g=lookup(grid,p,cell,origin);bool valid=observed(g);
            a.profile.push_back({d,valid?double(g->avg_z):0,g?g->type:0,g?g->count:0,valid});if(valid){++seen;gap=0;}else{gap+=cell;maxGap=std::max(gap,maxGap);}
            bool slope=valid&&g->type==2;if(slope&&!on)runStart=d;if(!slope&&on&&d-runStart>=2*cell)++a.slopeRuns;on=slope;
            if(valid&&g->type!=4){if(g->avg_z>minZ+3)reverse=true;minZ=std::min(minZ,double(g->avg_z));}
        }if(on&&a.width-runStart>=2*cell)++a.slopeRuns;a.coverage=double(seen)/a.profile.size();
        if(a.coverage<.95||maxGap>3*cell)a.status="incomplete_observed_profile";
        else if(reverse)a.status="possible_opposite_wall_or_uphill_crossing";
        else if(a.slopeRuns<2)a.status="insufficient_multiple_bench_evidence";
        else {a.status="extent_unconfirmed";a.height=a.top.z()-a.bottom.z();a.angle=std::atan2(a.height,a.width)*180/std::acos(-1.);a.geometryAvailable=true;}finish();
    }
        for(size_t i=first;i<out.size();++i)for(size_t j=i+1;j<out.size();++j){if(!out[i].hasTarget||!out[j].hasTarget)continue;Sample a,b;a.crest=out[i].top;a.toe=out[i].bottom;b.crest=out[j].top;b.toe=out[j].bottom;
            if(crossing(a,b))for(size_t k:{i,j})if(out[k].geometryAvailable){out[k].geometryAvailable=false;out[k].status="crossing_sections";}}
    }return out;
}
inline std::string spanText(const std::vector<Span>& spans){std::ostringstream s;s<<std::fixed<<std::setprecision(2);for(size_t i=0;i<spans.size();++i){if(i)s<<';';s<<spans[i].line<<':'<<spans[i].begin<<':'<<spans[i].end;}return s.str();}
inline void save(const Result& graph,const std::vector<overall::Preview>& rows,const std::string& prefix){
    overall::save(rows,prefix+"grouped_");std::ofstream groups(prefix+"grouped_benches.csv"),members(prefix+"grouped_members.csv"),joins(prefix+"grouped_join_evidence.csv"),links(prefix+"grouped_links.csv"),candidates(prefix+"grouped_candidates.csv"),settings(prefix+"grouped_settings.csv");
    if(!groups||!members||!joins||!links||!candidates||!settings)throw std::runtime_error("Cannot save grouped output");
    auto& o=graph.options;settings<<"parameter,value\nneighbor_m,"<<o.neighbor<<"\nelevation_difference_m,"<<o.elevation<<"\ndirection_cos,"<<o.directionCos<<"\nobserved_fraction,"<<o.observed<<"\nslope_fraction,"<<o.slopeSupport<<"\ntransverse_m,"<<o.transverse<<"\nspan_padding_m,"<<o.spanPadding<<"\nplatform_reach_m,"<<o.platformReach<<"\nminimum_samples,"<<o.minSamples<<"\nminimum_votes,"<<o.minVotes<<"\nmax_candidates,"<<o.maxCandidates<<'\n';
    groups<<"group_id,samples,crest_spans_line_start_end,toe_spans_line_start_end,mean_high,mean_low\n";members<<"group_id,pair_id,crest_line,toe_line\n";
    for(size_t i=0;i<graph.groups.size();++i){auto& g=graph.groups[i];groups<<i<<','<<g.samples.size()<<','<<spanText(g.crests)<<','<<spanText(g.toes)<<','<<g.high<<','<<g.low<<'\n';for(auto s:g.samples)members<<i<<','<<s->id<<','<<s->crestLine<<','<<s->toeLine<<'\n';}
    joins<<"pair_a,pair_b,evidence\n";for(auto& j:graph.joins)joins<<j.a<<','<<j.b<<",proximity_elevation_direction_observed_slope\n";
    links<<"upper_group,lower_group,platform_votes,status\n";for(auto e:graph.links)links<<e.upper<<','<<e.lower<<','<<e.votes<<",supported\n";for(auto e:graph.weak)links<<e.upper<<','<<e.lower<<','<<e.votes<<",insufficient_support\n";
    candidates<<"candidate,chain,upper_spans,lower_spans,sampled_sections,geometry_sections,upstream_weak,downstream_weak,upstream_link_elsewhere,downstream_link_elsewhere,extent_status,search_truncated\n";
    for(auto& c:graph.candidates){candidates<<c.name<<',';for(size_t k=0;k<c.chain.size();++k){if(k)candidates<<';';candidates<<c.chain[k];}int n=0,v=0;for(auto& a:rows)if(a.name==c.name){++n;if(a.geometryAvailable)++v;}
        candidates<<','<<spanText(graph.groups[c.chain.front()].crests)<<','<<spanText(graph.groups[c.chain.back()].toes)<<','<<n<<','<<v<<','<<c.upstreamWeak<<','<<c.downstreamWeak<<','<<c.upstreamLink<<','<<c.downstreamLink<<",unconfirmed_graph_end_not_physical_boundary,"<<graph.truncated<<'\n';}
}
}
