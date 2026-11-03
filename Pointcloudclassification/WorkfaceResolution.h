#pragma once
#include "GroupedWorkfaces.h"

namespace resolved {
using namespace sections;
struct Options {double localReach=15, stationGap=7.5, bottomJump=20, directionCos=.965925826, probe=30;int minRun=3;};
struct Decision {std::string state="raw_rejected",reason,topEnd,bottomEnd;int dominator=-1,zone=-1,upstreamGroup=-1;bool local=false,unresolvedExtension=false,unresolvedUpstream=false;};
struct Zone {int id,upper;double start,end;std::vector<int> rows;std::string state;};
struct Result {std::vector<Decision> decisions;std::vector<Zone> zones;Options options;};
inline bool prefix(const std::vector<int>& a,const std::vector<int>& b){return a.size()<b.size()&&std::equal(a.begin(),a.end(),b.begin());}
inline bool suffix(const std::vector<int>& a,const std::vector<int>& b){return a.size()<b.size()&&std::equal(a.rbegin(),a.rend(),b.rbegin());}
// The same graph suffix is not sufficient: both rays must cover the same local strip.
inline bool upstreamCovers(const overall::Preview& child,const overall::Preview& parent,double tolerance=7.5){
    XY cd=xy(child.bottom)-xy(child.top),pd=xy(parent.bottom)-xy(parent.top);double length=pd.norm();
    if(cd.norm()<1e-6||length<1e-6||length<=cd.norm()+1||parent.top.z()<=child.top.z()+1)return false;
    if(cd.normalized().dot(pd.normalized())<.985||(xy(child.bottom)-xy(parent.bottom)).norm()>tolerance||std::abs(child.bottom.z()-parent.bottom.z())>3)return false;
    XY delta=xy(child.top)-xy(parent.top),direction=pd/length;double along=delta.dot(direction);
    return along>1&&along<length&&(delta-direction*along).norm()<=tolerance;
}
inline bool compatible(const overall::Preview& a,const overall::Preview& b,const Options& o){
    if(a.upper!=b.upper||b.station-a.station<=0||b.station-a.station>o.stationGap)return false;
    XY da=xy(a.bottom)-xy(a.top),db=xy(b.bottom)-xy(b.top);if(da.norm()<1e-6||db.norm()<1e-6)return false;
    return da.normalized().dot(db.normalized())>=o.directionCos&&(xy(a.bottom)-xy(b.bottom)).norm()<=o.bottomJump&&std::abs(a.bottom.z()-b.bottom.z())<=3;
}
inline std::string probe(const V& p,const XY& direction,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin,const Options& o){
    int seen=0,flat=0,n=0;double slopeRun=0,maxRun=0;
    for(double d=2*cell;d<=o.probe;d+=cell){V q=p;q.head<2>()+=d*direction;auto g=grouped::lookup(grid,q,cell,origin);++n;
        if(!grouped::observed(g))return "missing_or_cropped_surface";
        ++seen;if(g->type==1||g->type==4)++flat;if(g->type==2){slopeRun+=cell;maxRun=std::max(slopeRun,maxRun);}else slopeRun=0;
    }
    if(!n||seen!=n)return "insufficient_probe";
    if(maxRun>=3*cell)return "possible_continuing_slope";
    return double(flat)/n>=.9?"observed_flat_tail_not_boundary_proof":"unclear_surface";
}
inline Result run(const grouped::Result& graph,const std::vector<overall::Preview>& rows,const std::vector<platforms::Measurement>& platforms,
 const std::vector<FeatureLine>& source,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin,const Options& o={}){
    if(cell<=0||o.localReach<=0||o.minRun<2||o.stationGap<=0||o.probe<3*cell)throw std::runtime_error("Invalid resolution options");
    Result out;out.options=o;out.decisions.resize(rows.size());auto local=overall::localLines(source,origin);std::map<int,const Line*> lines;for(auto& l:local)lines[l.id]=&l;
    std::map<std::string,const grouped::Candidate*> candidates;for(auto& c:graph.candidates)candidates[c.name]=&c;
    std::map<int,int> owner;for(size_t k=0;k<graph.groups.size();++k)for(auto s:graph.groups[k].samples)owner[s->id]=int(k);
    std::map<int,std::vector<const Sample*>> widths;for(auto& pm:platforms)if(pm.geometry.status=="matched"&&owner.count(pm.geometry.id))widths[owner[pm.geometry.id]].push_back(&pm.geometry);
    V offset(origin.x(),origin.y(),0);
    // A platform vote elsewhere in a group is not evidence for this ray.
    for(size_t i=0;i<rows.size();++i){auto& a=rows[i];auto& d=out.decisions[i];d.reason=a.status;if(!a.geometryAvailable)continue;
        auto ci=candidates.find(a.name);if(ci==candidates.end())throw std::runtime_error("Unknown candidate in resolution");auto& c=*ci->second;
        XY start=xy(a.top-offset),dir=(xy(a.bottom)-xy(a.top)).normalized();double previous=0;bool ok=true;
        for(size_t k=0;k<c.chain.size();++k){const auto& g=graph.groups[c.chain[k]];double crestDistance=0;
            if(k){auto crest=grouped::intersect(g.crests,lines,start,dir,previous,a.width+cell);if(!crest.found){d.reason="missing_local_crest";ok=false;break;}crestDistance=crest.distance;
                V upperToe=a.top,lowerCrest=a.top;upperToe.head<2>()+=dir*previous;lowerCrest.head<2>()+=dir*crestDistance;
                bool support=false;for(auto p:widths[c.chain[k]]){bool belongs=false;for(auto& span:graph.groups[c.chain[k-1]].toes)if(span.line==p->toeLine)belongs=true;
                    if(belongs&&(xy(p->toe)-xy(upperToe)).norm()<=o.localReach&&(xy(p->crest)-xy(lowerCrest)).norm()<=o.localReach){support=true;break;}}
                if(!support){d.reason="no_local_platform_support";ok=false;break;}
                int total=0,seen=0,flat=0;for(double t=previous+cell*.5;t<crestDistance;t+=cell){V p=a.top;p.head<2>()+=dir*t;auto gcell=grouped::lookup(grid,p,cell,origin);++total;if(grouped::observed(gcell)){++seen;if(gcell->type==1||gcell->type==4)++flat;}}
                if(!total||double(seen)/total<.9||double(flat)/total<.6){d.reason="unsupported_local_platform_profile";ok=false;break;}
            }
            auto toe=grouped::intersect(g.toes,lines,start,dir,crestDistance,a.width+cell);if(!toe.found){d.reason="missing_local_toe";ok=false;break;}previous=toe.distance;
        }
        if(!ok){d.state="local_evidence_rejected";continue;}d.local=true;d.state="pending";d.reason="local_chain_supported";
        d.topEnd=probe(a.top,-dir,grid,cell,origin,o);d.bottomEnd=probe(a.bottom,dir,grid,cell,origin,o);
    }
    // Quantization is only an identity tolerance (1 mm), not a spatial merge radius.
    using Key=std::pair<int,long long>;std::map<Key,std::vector<int>> buckets;
    for(size_t i=0;i<rows.size();++i)buckets[{rows[i].upper,std::llround(rows[i].station*1000)}].push_back(int(i));
    std::map<int,std::vector<int>> uniqueByLine;
    for(auto& entry:buckets){std::vector<int> supported;for(int i:entry.second)if(out.decisions[i].local)supported.push_back(i);
        for(int i:supported)for(int j:supported){if(i==j)continue;auto& a=rows[i];auto& b=rows[j];auto& ca=*candidates.at(a.name);auto& cb=*candidates.at(b.name);
            XY da=xy(a.bottom)-xy(a.top),db=xy(b.bottom)-xy(b.top);
            if(prefix(ca.chain,cb.chain)&&(xy(a.top)-xy(b.top)).norm()<.01&&da.normalized().dot(db.normalized())>.99999&&b.width>a.width+cell){
                auto& d=out.decisions[i];if(d.dominator<0||cb.chain.size()>candidates.at(rows[d.dominator].name)->chain.size()){d.state="subchain";d.reason="locally_supported_extension_exists";d.dominator=j;}
            }
        }
        std::vector<int> maxima;for(int i:supported)if(out.decisions[i].state=="pending")maxima.push_back(i);
        if(maxima.size()>1){for(int i:maxima){out.decisions[i].state="uncertain_branch";out.decisions[i].reason="multiple_non_dominated_chains";}continue;}
        if(maxima.size()==1){int i=maxima[0];out.decisions[i].state="unique_local";
            for(int j:entry.second)if(!out.decisions[j].local&&prefix(candidates.at(rows[i].name)->chain,candidates.at(rows[j].name)->chain))out.decisions[i].unresolvedExtension=true;
            uniqueByLine[rows[i].upper].push_back(i);}
    }
    // Cross-crest upward ownership: a lower starting crest can be inside a longer wall.
    // This does not synthesize a new ray or repair a missing boundary.
    for(size_t i=0;i<rows.size();++i){auto& d=out.decisions[i];if(!d.local||d.state=="subchain")continue;
        auto& a=rows[i];const auto& chain=candidates.at(a.name)->chain;int best=-1;double bestDistance=1e30;
        for(size_t j=0;j<rows.size();++j){if(i==j||!out.decisions[j].local)continue;auto& b=rows[j];auto& parentChain=candidates.at(b.name)->chain;
            if(!suffix(chain,parentChain)||!upstreamCovers(a,b))continue;
            double distance=(xy(a.bottom)-xy(b.bottom)).norm();
            // Prefer the fullest supported ancestor, then the closest local ray.
            if(best<0||parentChain.size()>candidates.at(rows[best].name)->chain.size()||(parentChain.size()==candidates.at(rows[best].name)->chain.size()&&(distance<bestDistance-1e-8||(std::abs(distance-bestDistance)<1e-8&&b.id<rows[best].id)))){best=int(j);bestDistance=distance;}
        }
        if(best>=0){d.state="subchain";d.reason="supported_upper_chain_contains_local_strip";d.dominator=best;continue;}
        // Even if the higher full ray is missing, local platform evidence prevents
        // interpreting a flat 30 m probe as proof of the highest crest.
        XY uphill=(xy(a.top)-xy(a.bottom)).normalized();std::set<int> parents;
        auto inspect=[&](const grouped::Link& link){if(link.lower!=chain.front())return;
            for(auto p:widths[chain.front()]){if((xy(p->crest)-xy(a.top)).norm()>o.localReach)continue;
                XY up=xy(p->toe)-xy(p->crest);if(up.norm()<cell||up.normalized().dot(uphill)<.7)continue;
                bool supported=false;for(auto s:graph.groups[link.upper].samples)if(s->toeLine==p->toeLine&&(xy(s->toe)-xy(p->toe)).norm()<graph.options.platformReach){supported=true;break;}
                if(supported){parents.insert(link.upper);break;}
            }
        };
        for(auto& link:graph.links)inspect(link);for(auto& link:graph.weak)inspect(link);
        if(!parents.empty()){d.unresolvedUpstream=true;d.upstreamGroup=parents.size()==1?*parents.begin():-2;}
    }
    // No neighbour vote invents geometry. Gaps, chain switches and endpoint jumps split runs.
    for(auto& entry:uniqueByLine){auto& ids=entry.second;std::sort(ids.begin(),ids.end(),[&](int a,int b){return rows[a].station<rows[b].station;});std::vector<int> run;
        auto flush=[&](){if(run.empty())return;Zone z;z.id=int(out.zones.size());z.upper=entry.first;z.start=rows[run.front()].station;z.end=rows[run.back()].station;z.rows=run;
            bool stable=run.size()>=size_t(o.minRun),terminal=true,blockedExtension=false,blockedUpstream=false;
            for(int i:run){auto& d=out.decisions[i];if(d.topEnd!="observed_flat_tail_not_boundary_proof"||d.bottomEnd!="observed_flat_tail_not_boundary_proof")terminal=false;if(d.unresolvedExtension)blockedExtension=true;}
            for(int i:run)if(out.decisions[i].unresolvedUpstream)blockedUpstream=true;
            z.state=stable&&terminal&&!blockedExtension&&!blockedUpstream&&!graph.truncated?"recommended_unconfirmed":"uncertain_extent";
            for(int i:run){auto& d=out.decisions[i];d.zone=z.id;d.state=z.state;d.reason=graph.truncated?"candidate_search_truncated":blockedUpstream?"upper_extent_unresolved_local_parent":blockedExtension?"unresolved_longer_chain":!stable?"insufficient_adjacent_sections":!terminal?"termination_not_supported":"unique_continuous_local_chain";}
            out.zones.push_back(z);run.clear();};
        auto boundaryClass=[&](int i){auto& d=out.decisions[i];return std::make_tuple(d.unresolvedUpstream,d.unresolvedExtension,d.topEnd=="observed_flat_tail_not_boundary_proof"&&d.bottomEnd=="observed_flat_tail_not_boundary_proof");};
        for(int i:ids){if(out.decisions[i].state!="unique_local"){flush();continue;}if(!run.empty()&&(!compatible(rows[run.back()],rows[i],o)||candidates.at(rows[run.back()].name)->chain!=candidates.at(rows[i].name)->chain||boundaryClass(run.back())!=boundaryClass(i)))flush();run.push_back(i);}flush();
    }
    // Cross-candidate geometric conflicts revoke recommendations symmetrically.
    std::set<int> conflicted;
    for(size_t i=0;i<rows.size();++i)if(out.decisions[i].state=="recommended_unconfirmed")for(size_t j=i+1;j<rows.size();++j)if(out.decisions[j].state=="recommended_unconfirmed"){
        Sample a,b;a.crest=rows[i].top;a.toe=rows[i].bottom;b.crest=rows[j].top;b.toe=rows[j].bottom;if(crossing(a,b)){conflicted.insert(out.decisions[i].zone);conflicted.insert(out.decisions[j].zone);}}
    for(int z:conflicted){out.zones[z].state="uncertain_extent";for(int i:out.zones[z].rows){out.decisions[i].state="uncertain_extent";out.decisions[i].reason="cross_zone_section_conflict";}}
    return out;
}
inline void save(const Result& result,const std::vector<overall::Preview>& rows,const std::string& prefix){
    std::ofstream settings(prefix+"resolved_settings.csv");if(!settings)throw std::runtime_error("Cannot save resolution settings");auto& o=result.options;
    settings<<"parameter,value\nlocal_platform_reach_m,"<<o.localReach<<"\nstation_gap_m,"<<o.stationGap<<"\nbottom_jump_m,"<<o.bottomJump<<"\ndirection_cos,"<<o.directionCos<<"\nterminal_probe_m,"<<o.probe<<"\nminimum_adjacent_sections,"<<o.minRun<<"\nstation_identity_tolerance_m,0.001\n";
    std::ofstream audit(prefix+"resolved_decisions.csv"),zones(prefix+"resolved_zones.csv"),selected(prefix+"resolved_recommended.csv");if(!audit||!zones||!selected)throw std::runtime_error("Cannot save resolved workfaces");
    audit<<"section_id,candidate,upper_line,station,lower_line,state,reason,dominator_section,zone,top_probe,bottom_probe,unresolved_extension,unresolved_upstream,upstream_group\n";
    selected<<"section_id,candidate,zone,upper_line,station,lower_line,height_m,span_m,angle_deg,scope\n";
    for(size_t i=0;i<rows.size();++i){auto& a=rows[i];auto& d=result.decisions[i];audit<<a.id<<','<<a.name<<','<<a.upper<<','<<a.station<<','<<a.lower<<','<<d.state<<','<<d.reason<<','<<(d.dominator<0?-1:rows[d.dominator].id)<<','<<d.zone<<','<<d.topEnd<<','<<d.bottomEnd<<','<<d.unresolvedExtension<<','<<d.unresolvedUpstream<<','<<d.upstreamGroup<<'\n';
        if(d.state=="recommended_unconfirmed")selected<<a.id<<','<<a.name<<','<<d.zone<<','<<a.upper<<','<<a.station<<','<<a.lower<<','<<a.height<<','<<a.width<<','<<a.angle<<",geometry_extent_unconfirmed\n";
    }
    zones<<"zone,upper_line,start_m,end_m,sections,state,scope\n";for(auto& z:result.zones)zones<<z.id<<','<<z.upper<<','<<z.start<<','<<z.end<<','<<z.rows.size()<<','<<z.state<<",sampled_interval_not_complete_workface_polygon\n";
}
}
