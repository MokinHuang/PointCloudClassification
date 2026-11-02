#pragma once
#include "WorkfaceOwnership.h"
#include "ResolvedWorkfaceReview.h"
namespace through {
using namespace sections;
struct Audit {int wall=-1;double station=0;std::string reason,chain,missing,platformEvidence,upperState="not_checked",upperEvidence;int originalUpper=-1;};
struct Result {std::vector<overall::Preview> rows;std::vector<Audit> audit;};
struct Interval {grouped::Intersection crest,toe;};
// A branch elsewhere along the wall is not evidence of a lower bench here.
// Test its original span footprint, including a small neighbouring station band.
inline bool downstreamFootprint(const grouped::Group& group,const std::map<int,const Line*>& lines,const XY& dir,const XY& across,double station,double after){
    for(auto spans:{&group.crests,&group.toes})for(auto& span:*spans){auto found=lines.find(span.line);if(found==lines.end())continue;const auto& l=*found->second;
        double begin=std::clamp(span.begin-3,0.,l.s.back()),end=std::clamp(span.end+3,0.,l.s.back());
        std::vector<V> points{at(l,begin)};for(size_t k=0;k<l.p.size();++k)if(l.s[k]>begin&&l.s[k]<end)points.push_back(l.p[k]);points.push_back(at(l,end));
        for(size_t k=1;k<points.size();++k){XY a=xy(points[k-1]),b=xy(points[k]);double u=a.dot(across),v=b.dot(across);
            double t0=0,t1=1;if(std::abs(v-u)<1e-9){if(std::abs(u-station)>7.5)continue;}else{t0=(station-7.5-u)/(v-u);t1=(station+7.5-u)/(v-u);if(t0>t1)std::swap(t0,t1);t0=std::max(0.,t0);t1=std::min(1.,t1);if(t0>t1)continue;}
            if(std::max((a+(b-a)*t0).dot(dir),(a+(b-a)*t1).dot(dir))>after+1)return true;
        }
    }return false;
}
inline bool continuousBoundary(const overall::Preview& a,const overall::Preview& b){
    return a.name==b.name&&std::abs(a.station-b.station)<=5.01&&(xy(a.top)-xy(b.top)).norm()<=20&&(xy(a.bottom)-xy(b.bottom)).norm()<=20&&std::abs(a.top.z()-b.top.z())<=3&&std::abs(a.bottom.z()-b.bottom.z())<=3;
}
inline bool surface(const V& a,const V& b,int type,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin){
    double length=(xy(b)-xy(a)).norm();int n=std::max(1,int(std::ceil(length/cell))),seen=0,kind=0;for(int k=0;k<=n;++k){auto p=a+(b-a)*(double(k)/n);auto g=grouped::lookup(grid,p,cell,origin);if(grouped::observed(g)){++seen;if(g->type==type||(type==1&&g->type==4))++kind;}}
    return double(seen)/(n+1)>=.95&&double(kind)/(n+1)>=.6;
}
// Revalidate a known topological link on this exact ray when nearby formal
// platform samples are absent. This does not create a formal platform width.
inline bool measuredPlatform(const V& a,const V& b,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin){
    auto ga=grouped::lookup(grid,a,cell,origin),gb=grouped::lookup(grid,b,cell,origin);if(!grouped::observed(ga)||!grouped::observed(gb))return false;V from=a,to=b;from.z()=ga->avg_z;to.z()=gb->avg_z;
    double length=(xy(b)-xy(a)).norm();if(length<5*cell||length>150||std::abs(from.z()-to.z())>2.5)return false;
    int seen=0,flat=0,n=0;double badRun=0;for(double d=2*cell;d<length-2*cell;d+=cell){V p=from+(to-from)*(d/length);auto g=grouped::lookup(grid,p,cell,origin);++n;if(!grouped::observed(g))return false;++seen;
        if(g->type==1||g->type==4){++flat;badRun=0;}else badRun+=cell;
        if(badRun>cell|| (g->type!=4&&std::abs(g->avg_z-p.z())>2.0))return false;
    }return n>=3&&seen==n&&double(flat)/n>=.85;
}
inline Result run(const grouped::Result& graph,const ownership::Result& owned,const std::vector<overall::Preview>& seeds,const std::vector<platforms::Measurement>& widths,
 const std::vector<FeatureLine>& source,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin){
    if(cell<=0||grid.empty())throw std::runtime_error("Invalid through-section grid");Result out;auto local=overall::localLines(source,origin);std::map<int,const Line*> lines;for(auto& l:local)lines[l.id]=&l;V offset(origin.x(),origin.y(),0);
    std::map<int,int> sampleOwner;for(size_t i=0;i<graph.groups.size();++i)for(auto s:graph.groups[i].samples)sampleOwner[s->id]=int(i);
    std::map<int,std::vector<const Sample*>> platforms;for(auto& w:widths)if(w.geometry.status=="matched"&&sampleOwner.count(w.geometry.id))platforms[sampleOwner[w.geometry.id]].push_back(&w.geometry);
    for(size_t wi=0;wi<owned.walls.size();++wi){const auto& wall=owned.walls[wi];XY dir=XY::Zero();for(int i:wall.members)dir+=(xy(seeds[i].bottom)-xy(seeds[i].top)).normalized();if(dir.norm()<1e-6)continue;dir.normalize();XY across(-dir.y(),dir.x());
        std::set<int> searchGroups=wall.groups;bool expanded=true;while(expanded){expanded=false;for(auto& link:graph.links)if(searchGroups.count(link.lower)&&!searchGroups.count(link.upper)){searchGroups.insert(link.upper);expanded=true;}}
        double lo=1e30,hi=-1e30,left=1e30,right=-1e30;for(int i:wall.members)for(auto p:{seeds[i].top,seeds[i].bottom}){XY q=xy(p-offset);lo=std::min(lo,q.dot(dir));hi=std::max(hi,q.dot(dir));left=std::min(left,q.dot(across));right=std::max(right,q.dot(across));}
        // A short seed must not clip an existing lower boundary out of the ray.
        // Keep the sampled lateral footprint, but bound the search longitudinally
        // using the actual owned feature spans (not a guessed extension radius).
        for(int g:searchGroups)for(auto spans:{&graph.groups[g].crests,&graph.groups[g].toes})for(auto& span:*spans){auto found=lines.find(span.line);if(found==lines.end())continue;auto& l=*found->second;
            double begin=std::clamp(span.begin-3,0.,l.s.back()),end=std::clamp(span.end+3,0.,l.s.back());auto include=[&](const V& p){double d=xy(p).dot(dir);lo=std::min(lo,d);hi=std::max(hi,d);};include(at(l,begin));include(at(l,end));for(size_t k=0;k<l.p.size();++k)if(l.s[k]>=begin&&l.s[k]<=end)include(l.p[k]);}
        std::vector<const grouped::Candidate*> candidates;for(auto& c:graph.candidates){if(!wall.heads.count(c.chain.front())||!wall.tails.count(c.chain.back()))continue;bool inside=true;for(int g:c.chain)if(!wall.groups.count(g))inside=false;if(inside)candidates.push_back(&c);}
        for(double station=std::ceil(left/5)*5;station<=right+1e-6;station+=5){overall::Preview row;row.id=int(out.rows.size());row.name="WALL_"+std::to_string(wi);row.station=station;row.upperType=0;row.lowerType=1;
            Audit audit; audit.wall=int(wi);audit.station=station;XY start=dir*(lo-5)+across*station;double limit=hi-lo+10;std::map<int,Interval> hits;std::map<int,std::string> missing;
            for(int g:searchGroups){auto spansC=graph.groups[g].crests,spansT=graph.groups[g].toes;for(auto spans:{&spansC,&spansT})for(auto& s:*spans){s.begin=std::max(0.,s.begin-3);s.end+=3;}
                auto c=grouped::intersect(spansC,lines,start,dir,0,limit);if(!c.found){missing[g]="crest";continue;}auto t=grouped::intersect(spansT,lines,start,dir,c.distance,limit);if(!t.found){missing[g]="toe";continue;}hits[g]={c,t};}
            std::vector<const grouped::Candidate*> good;std::map<std::string,std::string> evidence;std::string failure=candidates.empty()?"no_complete_topology_path":"missing_intermediate_boundary";
            size_t fewestMissing=graph.groups.size()+1;for(auto candidate:candidates){bool ok=true;size_t nMissing=0;std::string details;for(int g:candidate->chain)if(!hits.count(g)){ok=false;++nMissing;if(!details.empty())details+=';';details+=std::to_string(g)+":"+missing[g];}if(nMissing<fewestMissing){fewestMissing=nMissing;audit.missing=details;}if(!ok)continue;
                for(size_t k=0;k<candidate->chain.size();++k){int g=candidate->chain[k];auto& h=hits[g];V c=h.crest.point+offset,t=h.toe.point+offset;
                    if(c.z()<=t.z()+.5||!surface(c,t,2,grid,cell,origin)){ok=false;failure="unsupported_bench_surface";break;}
                    if(!k)continue;auto& previous=hits[candidate->chain[k-1]];if(h.crest.distance<=previous.toe.distance+cell){ok=false;failure="invalid_boundary_order";break;}
                    V upperToe=previous.toe.point+offset;bool support=false;for(auto p:platforms[g])if(p->toeLine==previous.toe.line&&(xy(p->crest)-xy(c)).norm()<=15&&(xy(p->toe)-xy(upperToe)).norm()<=15){support=true;break;}
                    if(!support){bool known=false;for(auto& link:graph.links)if(link.upper==candidate->chain[k-1]&&link.lower==g)known=true;
                        if(!known||!measuredPlatform(upperToe,c,grid,cell,origin)){ok=false;failure="no_local_platform_support";break;}
                        if(!evidence[candidate->name].empty())evidence[candidate->name]+=';';evidence[candidate->name]+=std::to_string(candidate->chain[k-1])+">"+std::to_string(g)+":observed_grid";
                    }if(!surface(upperToe,c,1,grid,cell,origin)){ok=false;failure="unsupported_platform_surface";break;}
                }if(ok)good.push_back(candidate);
            }
            if(graph.truncated)audit.reason="candidate_search_truncated";
            else if(good.size()!=1)audit.reason=good.empty()?failure:"competing_complete_paths";
            else{audit.platformEvidence=evidence[good.front()->name];auto chain=good.front()->chain;
                audit.originalUpper=hits[chain.front()].crest.line;bool upperUnresolved=false;int added=0;
                // Trace immediate predecessors, validating the actual same-station
                // slope and platform before adding each bench. Never skip a failed tier.
                for(size_t step=0;step<=graph.groups.size();++step){int head=chain.front();auto& current=hits[head];std::set<int> localParents;double before=xy(current.crest.point).dot(dir);
                    for(auto& link:graph.links)if(link.lower==head&&downstreamFootprint(graph.groups[link.upper],lines,-dir,across,station,-before))localParents.insert(link.upper);
                    if(localParents.empty())break;
                    if(localParents.size()!=1){upperUnresolved=true;audit.upperEvidence+="competing_parents;";break;}
                    int parent=*localParents.begin();audit.upperEvidence+=std::to_string(parent)+">"+std::to_string(head)+":";
                    if(std::find(chain.begin(),chain.end(),parent)!=chain.end()){upperUnresolved=true;audit.upperEvidence+="cycle;";break;}
                    if(!hits.count(parent)){upperUnresolved=true;audit.upperEvidence+="missing_"+missing[parent]+";";break;}
                    auto& up=hits[parent];V crest=up.crest.point+offset,toe=up.toe.point+offset,lowerCrest=current.crest.point+offset;
                    bool support=false;for(auto p:platforms[head])if(p->toeLine==up.toe.line&&(xy(p->crest)-xy(lowerCrest)).norm()<=15&&(xy(p->toe)-xy(toe)).norm()<=15){support=true;break;}
                    if(up.toe.distance+cell>=current.crest.distance||crest.z()<=toe.z()+.5||!surface(crest,toe,2,grid,cell,origin)){upperUnresolved=true;audit.upperEvidence+="invalid_bench;";break;}
                    if((!support&&!measuredPlatform(toe,lowerCrest,grid,cell,origin))||!surface(toe,lowerCrest,1,grid,cell,origin)){upperUnresolved=true;audit.upperEvidence+="unsupported_platform;";break;}
                    chain.insert(chain.begin(),parent);++added;audit.upperEvidence+=support?"matched_platform;":"observed_grid;";
                }
                // A disconnected higher tier in this wall is evidence of an
                // incomplete graph, not proof that the current crest is outermost.
                for(int g:wall.groups)if(std::find(chain.begin(),chain.end(),g)==chain.end()&&hits.count(g)){auto& h=hits[g];if(h.toe.distance+cell<hits[chain.front()].crest.distance&&h.crest.point.z()>hits[chain.front()].crest.point.z()+.5){upperUnresolved=true;audit.upperEvidence+=std::to_string(g)+":higher_disconnected_bench;";}}
                audit.upperState=upperUnresolved?"upper_extent_unresolved":added?"extended_no_known_local_upstream":"no_known_local_upstream";
                auto& first=hits[chain.front()].crest;auto& last=hits[chain.back()].toe;row.top=first.point+offset;row.bottom=last.point+offset;row.upper=first.line;row.lower=last.line;row.width=last.distance-first.distance;row.hasTarget=true;
                auto top=grouped::lookup(grid,row.top,cell,origin),bottom=grouped::lookup(grid,row.bottom,cell,origin);bool valid=grouped::observed(top)&&grouped::observed(bottom);double minimum=valid?top->avg_z:0;int seen=0;double gap=0,maxGap=0;bool reverse=false;
                for(double d=0;d<row.width+cell;d+=cell){double distance=std::min(d,row.width);V p=row.top;p.head<2>()+=dir*distance;auto g=grouped::lookup(grid,p,cell,origin);bool observed=grouped::observed(g);row.profile.push_back({distance,observed?double(g->avg_z):0,g?g->type:0,g?g->count:0,observed});if(observed){++seen;gap=0;if(g->type!=4){if(g->avg_z>minimum+3)reverse=true;minimum=std::min(minimum,double(g->avg_z));}}else{gap+=cell;maxGap=std::max(maxGap,gap);}}
                row.coverage=double(seen)/row.profile.size();if(!valid||row.coverage<.95||maxGap>3*cell)audit.reason="incomplete_observed_profile";else if(reverse)audit.reason="uphill_profile";else{row.top.z()=top->avg_z;row.bottom.z()=bottom->avg_z;row.height=row.top.z()-row.bottom.z();if(row.height<1)audit.reason="insufficient_drop";else{row.angle=std::atan2(row.height,row.width)*180/std::acos(-1.);row.geometryAvailable=true;row.slopeRuns=int(chain.size());audit.reason="through_verified_unconfirmed";}}
                if(row.geometryAvailable){bool downstream=false;double after=xy(last.point).dot(dir);
                    for(auto& link:graph.links)if(link.upper==chain.back()&&downstreamFootprint(graph.groups[link.lower],lines,dir,across,station,after))downstream=true;
                    // An intersected lower bench cannot be ignored just because its connecting path failed.
                    for(auto& item:hits)if(std::find(chain.begin(),chain.end(),item.first)==chain.end()&&item.second.crest.distance>last.distance+cell&&item.second.toe.point.z()<last.point.z()-.5)downstream=true;
                    if(downstream){row.geometryAvailable=false;audit.reason="bottom_extent_unresolved";}}
                if(upperUnresolved){row.geometryAvailable=false;audit.reason="upper_extent_unresolved";}
                for(int g:chain){if(!audit.chain.empty())audit.chain+=';';audit.chain+=std::to_string(g);}
            }row.status=audit.reason;out.rows.push_back(row);out.audit.push_back(audit);
        }
    }
    // Multi-branch owners require a run of at least three locally continuous
    // measured sections. Do not interpolate across gaps or force boundary jumps.
    std::set<int> stable;for(size_t i=0;i<out.rows.size();){if(!out.rows[i].geometryAvailable){++i;continue;}size_t end=i+1;while(end<out.rows.size()&&out.rows[end].geometryAvailable&&continuousBoundary(out.rows[end-1],out.rows[end]))++end;if(end-i>=3)for(size_t k=i;k<end;++k)stable.insert(int(k));i=end;}
    for(size_t i=0;i<out.rows.size();++i)if(out.rows[i].geometryAvailable&&owned.walls[out.audit[i].wall].tails.size()>1&&!stable.count(int(i))){out.rows[i].geometryAvailable=false;out.rows[i].status=out.audit[i].reason="boundary_continuity_unconfirmed";}
    // Different wall frames may overlap at corners; never publish intersecting rays.
    std::set<int> conflict;for(size_t i=0;i<out.rows.size();++i)if(out.rows[i].geometryAvailable)for(size_t j=i+1;j<out.rows.size();++j)if(out.rows[j].geometryAvailable){Sample a,b;a.crest=out.rows[i].top;a.toe=out.rows[i].bottom;b.crest=out.rows[j].top;b.toe=out.rows[j].bottom;if(crossing(a,b)){conflict.insert(int(i));conflict.insert(int(j));}}
    for(int i:conflict){out.rows[i].geometryAvailable=false;out.rows[i].status=out.audit[i].reason="cross_workface_conflict";}return out;
}
inline void process(const grouped::Result& graph,const ownership::Result& owned,const std::vector<overall::Preview>& seeds,const std::vector<platforms::Measurement>& widths,
 const std::vector<FeatureLine>& source,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin,const std::string& prefix){
    auto result=run(graph,owned,seeds,widths,source,grid,cell,origin);overall::save(result.rows,prefix+"through_");std::ofstream audit(prefix+"through_audit.csv");if(!audit)throw std::runtime_error("Cannot save through-section audit");audit<<"section_id,workface,transverse_station_m,chain,status,missing_group_boundary_on_best_path,platform_revalidation,upper_state,original_upper_line,upper_trace,scope\n";resolved::Result display;display.decisions.resize(result.rows.size());
    for(size_t i=0;i<result.rows.size();++i){auto& a=result.audit[i];audit<<result.rows[i].id<<','<<a.wall<<','<<a.station<<','<<a.chain<<','<<a.reason<<','<<a.missing<<','<<a.platformEvidence<<','<<a.upperState<<','<<a.originalUpper<<','<<a.upperEvidence<<",candidate_full_extent_unconfirmed\n";auto& d=display.decisions[i];d.zone=a.wall;d.state="owned_geometry";d.reason=a.reason;}
    resolved_review::write(source,grid,cell,origin,graph,result.rows,display,prefix+"through_",true);
    auto traceRows=result.rows;for(size_t i=0;i<traceRows.size();++i)if(traceRows[i].hasTarget&&(traceRows[i].status=="upper_extent_unresolved"||result.audit[i].upperState=="extended_no_known_local_upstream"))traceRows[i].geometryAvailable=true;
    resolved_review::write(source,grid,cell,origin,graph,traceRows,display,prefix+"upper_trace_",false,true);
}
}
