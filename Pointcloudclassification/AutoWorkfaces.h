#pragma once
#include "OverallSlopePreview.h"
#include "PlatformMeasurement.h"
#include <functional>

namespace autowalls {
using namespace sections;
struct Node {int crest,toe;double high=0,low=0,start=0,end=0;std::vector<const Sample*> samples;};
struct Edge {int upper,lower,support;};
struct Candidate {overall::Selection selection;std::vector<int> chain;};
struct Result {std::vector<Node> nodes;std::vector<Edge> edges;std::vector<Candidate> candidates;bool truncated=false;};
inline Result discover(const std::vector<Sample>& slopes,const std::vector<platforms::Measurement>& platforms){
    Result out;std::map<std::pair<int,int>,std::vector<const Sample*>> groups;
    for(const auto& s:slopes)if(s.status=="matched")groups[{s.crestLine,s.toeLine}].push_back(&s);
    // A line-pair is a graph node, not a globally unique physical bench.
    for(const auto& entry:groups){if(entry.second.size()<3)continue;Node n;n.crest=entry.first.first;n.toe=entry.first.second;n.samples=entry.second;
        n.start=n.end=n.samples[0]->crestS;
        for(auto s:n.samples){n.high+=s->crest.z();n.low+=s->toe.z();n.start=std::min(n.start,s->crestS);n.end=std::max(n.end,s->crestS);}
        n.high/=n.samples.size();n.low/=n.samples.size();out.nodes.push_back(n);
    }
    std::map<std::pair<int,int>,std::set<int>> votes;
    for(const auto& pm:platforms){const auto& p=pm.geometry;if(p.status!="matched")continue;
        // The platform goes from the lower bench crest uphill to the upper bench toe.
        for(size_t d=0;d<out.nodes.size();++d){auto& down=out.nodes[d];if(down.crest!=p.crestLine)continue;
            const Sample* ds=nullptr;for(auto s:down.samples)if(s->id==p.id){ds=s;break;}if(!ds)continue;
            XY dd=xy(ds->toe)-xy(ds->crest);if(dd.norm()<1e-6)continue;dd.normalize();
            for(size_t u=0;u<out.nodes.size();++u){auto& up=out.nodes[u];if(u==d||up.toe!=p.toeLine||up.high<=down.high+1)continue;
                const Sample* us=nullptr;double distance=10;
                for(auto s:up.samples){double delta=(xy(s->toe)-xy(p.toe)).norm();if(delta<distance){distance=delta;us=s;}}
                if(!us)continue;XY ud=xy(us->toe)-xy(us->crest);if(ud.norm()<1e-6)continue;ud.normalize();
                XY across=xy(p.crest)-xy(p.toe);if(across.norm()<1e-6)continue;across.normalize();
                if(ud.dot(dd)<.7||ud.dot(across)<.7||dd.dot(across)<.7)continue;
                votes[{int(u),int(d)}].insert(p.id);
            }
        }
    }
    std::vector<std::vector<int>> next(out.nodes.size());std::vector<int> indegree(out.nodes.size());
    for(auto& v:votes)if(v.second.size()>=3){out.edges.push_back({v.first.first,v.first.second,int(v.second.size())});next[v.first.first].push_back(v.first.second);++indegree[v.first.second];}
    std::set<std::pair<int,int>> emitted;std::vector<int> path;
    std::function<void(int)> visit=[&](int n){
        if(out.candidates.size()>=256){out.truncated=true;return;}
        path.push_back(n);
        if(next[n].empty()&&path.size()>=2){auto& top=out.nodes[path.front()];auto& bottom=out.nodes[n];
            if(emitted.insert({top.crest,bottom.toe}).second){Candidate c;c.chain=path;c.selection={"AUTO_"+std::to_string(out.candidates.size()+1),top.crest,bottom.toe,top.start,top.end,5,1500};out.candidates.push_back(c);}
        }else for(int child:next[n])visit(child);
        path.pop_back();
    };
    for(size_t i=0;i<out.nodes.size();++i)if(!indegree[i]&&!next[i].empty())visit(int(i));
    return out;
}
inline std::vector<overall::Preview> measure(const Result& graph,const std::vector<FeatureLine>& lines,
 const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin){
    std::vector<overall::Preview> out;
    auto local=overall::localLines(lines,origin);V offset(origin.x(),origin.y(),0);
    for(const auto& candidate:graph.candidates){auto rows=overall::measure(lines,grid,cell,origin,{candidate.selection});
        for(auto& a:rows){
            // Require the straight section to actually cross every claimed bench in order.
            if(a.geometryAvailable){XY start=xy(a.top-offset),dir=(xy(a.bottom)-xy(a.top)).normalized();double previous=-cell;bool supported=true;
                for(int nodeId:candidate.chain){const auto& node=graph.nodes[nodeId];double distances[2]={-1,-1};int ids[2]={node.crest,node.toe};
                    if(nodeId==candidate.chain.front())distances[0]=0;
                    if(nodeId==candidate.chain.back())distances[1]=a.width;
                    for(int k=0;k<2;++k)if(distances[k]<0)for(const auto& l:local)if(l.id==ids[k])for(size_t j=1;j<l.p.size();++j){double t,u;
                        if(hit(start,dir,xy(l.p[j-1]),xy(l.p[j]),t,u)&&t>=-cell*.25&&t<=a.width+cell*.25&&(distances[k]<0||t<distances[k]))distances[k]=t;
                    }
                    if(distances[0]<-.001||distances[1]<0||distances[0]<previous-cell*.25||distances[1]<=distances[0]){supported=false;break;}
                    previous=distances[1];
                }
                if(!supported){a.status="automatic_chain_not_crossed_in_order";a.geometryAvailable=false;}
            }
            a.id=int(out.size());out.push_back(a);
        }
    }return out;
}
inline void save(const Result& graph,const std::vector<overall::Preview>& rows,const std::string& prefix){
    overall::save(rows,prefix+"auto_");
    std::ofstream cfg(prefix+"auto_workfaces.csv"),nodes(prefix+"auto_bench_groups.csv"),edges(prefix+"auto_platform_links.csv"),summary(prefix+"auto_workface_summary.csv");
    if(!cfg||!nodes||!edges||!summary)throw std::runtime_error("Cannot write automatic candidate files");
    cfg<<"workface_id,upper_line_id,lower_line_id,start_m,end_m,spacing_m,max_distance_m\n";
    nodes<<"node_id,crest_line,toe_line,matched_sections\n";
    for(size_t i=0;i<graph.nodes.size();++i){auto& n=graph.nodes[i];nodes<<i<<','<<n.crest<<','<<n.toe<<','<<n.samples.size()<<'\n';}
    edges<<"upper_node,lower_node,matched_platform_support\n";for(auto& e:graph.edges)edges<<e.upper<<','<<e.lower<<','<<e.support<<'\n';
    summary<<"candidate_id,upper_crest,lower_toe,chain_nodes,sampled_sections,geometry_sections,height_min_m,height_max_m,span_min_m,span_max_m,angle_min_deg,angle_max_deg,boundary_status,production_status,search_truncated\n";
    cfg<<std::setprecision(12);summary<<std::fixed<<std::setprecision(4);
    for(auto& c:graph.candidates){auto& s=c.selection;cfg<<s.name<<','<<s.upper<<','<<s.lower<<','<<s.start<<','<<s.end<<','<<s.spacing<<','<<s.maxDistance<<'\n';
        int count=0,valid=0;double hmin=1e30,hmax=0,lmin=1e30,lmax=0,amin=1e30,amax=0;
        for(auto& a:rows)if(a.name==s.name){++count;if(a.geometryAvailable){++valid;hmin=std::min(hmin,a.height);hmax=std::max(hmax,a.height);lmin=std::min(lmin,a.width);lmax=std::max(lmax,a.width);amin=std::min(amin,a.angle);amax=std::max(amax,a.angle);}}
        summary<<s.name<<','<<s.upper<<','<<s.lower<<',';for(size_t i=0;i<c.chain.size();++i){if(i)summary<<';';summary<<c.chain[i];}
        summary<<','<<count<<','<<valid<<',';if(valid)summary<<hmin<<','<<hmax<<','<<lmin<<','<<lmax<<','<<amin<<','<<amax;else summary<<",,,,,";
        summary<<",observed_graph_extent_not_confirmed,unknown,"<<graph.truncated<<'\n';
    }
}
}
