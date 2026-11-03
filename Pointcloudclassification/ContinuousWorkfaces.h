#pragma once
#include "ResolvedWorkfaceReview.h"
#include "WorkfaceOwnership.h"
#include "ThroughWorkfaces.h"

// Independent geometry layer. Neither candidate membership nor a measured
// section proves the production boundary of a working wall.
namespace continuous {
using namespace sections;
struct Evidence {std::string method="original";double rotation=0,padding=0;int area=-1;bool selected=false;};
struct Result {std::vector<overall::Preview> rows;resolved::Result resolution;std::vector<Evidence> evidence;int recovered=0,areas=0;};
inline bool contained(const std::vector<int>& a,const std::vector<int>& b){return a.size()<=b.size()&&std::search(b.begin(),b.end(),a.begin(),a.end())!=b.end();}
inline bool strip(const V& a,const V& b,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin){
    int steps=std::max(1,int(std::ceil((xy(a)-xy(b)).norm()/cell)));
    for(int i=0;i<=steps;++i){V p=a+(b-a)*(double(i)/steps);auto g=grouped::lookup(grid,p,cell,origin);if(!grouped::observed(g)||std::abs(g->avg_z-p.z())>3)return false;}return true;
}
inline bool neighbors(const overall::Preview& a,const overall::Preview& b,const std::vector<int>& ca,const std::vector<int>& cb,
 const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin){
    XY da=xy(a.bottom)-xy(a.top),db=xy(b.bottom)-xy(b.top);if(da.norm()<cell||db.norm()<cell||da.normalized().dot(db.normalized())<.965925826)return false;
    if(!contained(ca,cb)&&!contained(cb,ca))return false;
    Sample sa,sb;sa.crest=a.top;sa.toe=a.bottom;sb.crest=b.top;sb.toe=b.bottom;if(crossing(sa,sb))return false;
    // Shared end level supplies the spatial witness; matching IDs alone do not.
    bool top=ca.front()==cb.front(),bottom=ca.back()==cb.back();
    auto endpointsSupported=[&](const V& p,const V& q){double distance=(xy(p)-xy(q)).norm();return distance<=7.5&&std::abs(p.z()-q.z())<=2.5&&std::abs((xy(q)-xy(p)).dot(da.normalized()))<=3&&strip(p,q,grid,cell,origin);};
    if(top&&bottom)return endpointsSupported(a.top,b.top)&&endpointsSupported(a.bottom,b.bottom);
    return (top&&endpointsSupported(a.top,b.top))||(bottom&&endpointsSupported(a.bottom,b.bottom));
}
inline Result run(const grouped::Result& graph,const std::vector<overall::Preview>& raw,const std::vector<platforms::Measurement>& widths,
 const std::vector<FeatureLine>& source,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin){
    Result out;out.rows=raw;out.evidence.resize(raw.size());auto expanded=graph;
    // Only 3 m beyond each assigned span, on the existing original polyline.
    for(auto& g:expanded.groups)for(auto spans:{&g.crests,&g.toes})for(auto& s:*spans){s.begin=std::max(0.,s.begin-3);s.end+=3;}
    std::set<int> pending;for(auto& a:raw)if(!a.geometryAvailable&&a.status=="incomplete_chain_intersection")pending.insert(a.id);
    for(int pass=0;pass<2;++pass)for(double angle:{0.,-3.,3.,-6.,6.}){
        if(pending.empty()||(!pass&&angle==0))continue;auto& searchGraph=pass?expanded:graph;
        auto trial=grouped::measure(searchGraph,source,grid,cell,origin,angle,&pending);
        std::vector<overall::Preview> usable;for(auto& a:trial)if(a.geometryAvailable)usable.push_back(a);
        auto checked=resolved::run(searchGraph,usable,widths,source,grid,cell,origin);
        for(size_t k=0;k<usable.size();++k){auto& a=usable[k];if(!checked.decisions[k].local)continue;
            bool conflict=false;for(auto& b:out.rows)if(b.geometryAvailable&&a.name==b.name){Sample x,y;x.crest=a.top;x.toe=a.bottom;y.crest=b.top;y.toe=b.bottom;if(crossing(x,y)){conflict=true;break;}}
            if(conflict)continue;int index=a.id;if(index<0||index>=int(raw.size())||raw[index].id!=index)throw std::runtime_error("Recovery requires stable section IDs");
            out.rows[index]=a;out.evidence[index].method=pass?"existing_line_span_extension":"small_angle_existing_lines";out.evidence[index].rotation=angle;out.evidence[index].padding=pass?3:0;pending.erase(index);++out.recovered;
        }
    }
    out.resolution=resolved::run(expanded,out.rows,widths,source,grid,cell,origin);
    // Recheck original rows against original spans. Enlarged search windows may
    // validate recovered geometry, but must not silently relax old evidence.
    auto baseline=resolved::run(graph,raw,widths,source,grid,cell,origin);
    std::map<std::string,std::vector<int>> chains;for(auto& c:graph.candidates)chains[c.name]=c.chain;
    std::vector<int> selected;
    for(size_t i=0;i<raw.size();++i){auto& d=out.resolution.decisions[i];if(out.evidence[i].method=="original"&&!baseline.decisions[i].local)continue;
        if(!d.local||d.state=="subchain"||d.state=="uncertain_branch")continue;
        out.evidence[i].selected=true;selected.push_back(int(i));}
    std::set<int> conflicts;for(size_t x=0;x<selected.size();++x)for(size_t y=x+1;y<selected.size();++y){int i=selected[x],j=selected[y];Sample a,b;a.crest=out.rows[i].top;a.toe=out.rows[i].bottom;b.crest=out.rows[j].top;b.toe=out.rows[j].bottom;if(crossing(a,b)){conflicts.insert(i);conflicts.insert(j);}}
    for(int i:conflicts){out.evidence[i].selected=false;out.resolution.decisions[i].reason="cross_section_geometry_conflict";}selected.erase(std::remove_if(selected.begin(),selected.end(),[&](int i){return conflicts.count(i)>0;}),selected.end());
    std::vector<int> parent(raw.size());std::iota(parent.begin(),parent.end(),0);std::function<int(int)> root=[&](int i){return parent[i]==i?i:parent[i]=root(parent[i]);};
    for(size_t x=0;x<selected.size();++x)for(size_t y=x+1;y<selected.size();++y){int i=selected[x],j=selected[y];if(neighbors(out.rows[i],out.rows[j],chains[out.rows[i].name],chains[out.rows[j].name],grid,cell,origin))parent[root(j)]=root(i);}
    std::map<int,int> ids;for(int i:selected){int r=root(i);if(!ids.count(r))ids[r]=int(ids.size());out.evidence[i].area=ids[r];}out.areas=int(ids.size());return out;
}
inline void save(const Result& result,const std::vector<overall::Preview>& raw,const std::string& prefix){
    std::ofstream audit(prefix+"continuous_sections.csv"),areas(prefix+"continuous_areas.csv");if(!audit||!areas)throw std::runtime_error("Cannot save continuous geometry");
    audit<<"section_id,candidate,area,selected,method,rotation_deg,span_extension_m,original_status,resolution_state,reason,upper_line,lower_line,height_m,span_m,angle_deg,extent_status\n";
    std::map<int,std::pair<int,int>> counts;
    for(size_t i=0;i<result.rows.size();++i){auto& a=result.rows[i];auto& e=result.evidence[i];auto& d=result.resolution.decisions[i];
        audit<<a.id<<','<<a.name<<','<<e.area<<','<<e.selected<<','<<e.method<<','<<e.rotation<<','<<e.padding<<','<<raw[i].status<<','<<d.state<<','<<d.reason<<','<<a.upper<<','<<a.lower<<',';
        if(a.geometryAvailable)audit<<a.height<<','<<a.width<<','<<a.angle;else audit<<",,";audit<<",production_boundary_unconfirmed\n";
        if(e.selected){++counts[e.area].first;if(e.method!="original")++counts[e.area].second;}}
    areas<<"area,sections,recovered_sections,status,scope\n";for(auto& c:counts)areas<<c.first<<','<<c.second.first<<','<<c.second.second<<','<<(c.second.first>=3?"continuous_geometry":"isolated_geometry")<<",sampled_support_not_complete_boundary\n";
    overall::save(result.rows,prefix+"continuous_");
}
inline void write(const std::vector<FeatureLine>& source,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin,
 const grouped::Result& graph,const Result& result,const std::string& prefix){
    // Reuse the inspected canvas renderer; explicit separate file and state.
    auto display=result.resolution;for(size_t i=0;i<display.decisions.size();++i){auto& d=display.decisions[i];if(result.evidence[i].selected){d.state="identified_geometry";d.zone=result.evidence[i].area;d.reason=result.evidence[i].method=="original"?"observed_local_chain_extent_unconfirmed":"recovered_existing_line_chain_extent_unconfirmed";}}
    resolved_review::write(source,grid,cell,origin,graph,result.rows,display,prefix+"continuous_");
}
inline void process(const grouped::Result& graph,const std::vector<overall::Preview>& raw,const std::vector<platforms::Measurement>& widths,
 const std::vector<FeatureLine>& source,const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin,const std::string& prefix){
    auto result=run(graph,raw,widths,source,grid,cell,origin);save(result,raw,prefix);write(source,grid,cell,origin,graph,result,prefix);
    std::vector<int> selected,areas;for(size_t i=0;i<result.rows.size();++i){areas.push_back(result.evidence[i].area);if(result.evidence[i].selected)selected.push_back(int(i));}
    auto owned=ownership::build(graph,result.rows,selected);ownership::save(owned,result.rows,areas,prefix);
    through::process(graph,owned,result.rows,widths,source,grid,cell,origin,prefix);
    auto display=result.resolution;for(int i:selected){auto& d=display.decisions[i];d.zone=owned.owner[i];d.state="owned_geometry";d.reason=owned.role[i];}
    resolved_review::write(source,grid,cell,origin,graph,result.rows,display,prefix+"owned_");
}
}
