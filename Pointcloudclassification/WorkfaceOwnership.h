#pragma once
#include "WorkfaceResolution.h"
namespace ownership {
using namespace sections;
struct Wall {std::vector<int> members;std::set<int> groups,heads,tails;bool ordered=false;};
struct Result {std::vector<int> owner;std::vector<std::string> role;std::vector<Wall> walls;};
inline bool sharesOrderedBenches(const std::vector<int>& a,const std::vector<int>& b){
    std::vector<int> left,right;for(int g:a)if(std::find(b.begin(),b.end(),g)!=b.end())left.push_back(g);
    for(int g:b)if(std::find(a.begin(),a.end(),g)!=a.end())right.push_back(g);if(left!=right||left.empty())return false;if(left.size()>=2)return true;
    // Consecutive upper/lower subchains may share only the joining bench.
    // A mere common head or common tail is not enough to merge sibling branches.
    int g=left.front();auto ia=std::find(a.begin(),a.end(),g),ib=std::find(b.begin(),b.end(),g);
    return (ia!=a.begin()&&ib+1!=b.end())||(ib!=b.begin()&&ia+1!=a.end());
}
inline bool aligned(const overall::Preview& a,const overall::Preview& b){
    XY u=xy(a.bottom)-xy(a.top),v=xy(b.bottom)-xy(b.top);return u.norm()>1e-6&&v.norm()>1e-6&&u.normalized().dot(v.normalized())>=.866025404;
}
inline Result build(const grouped::Result& graph,const std::vector<overall::Preview>& rows,const std::vector<int>& selected){
    Result out;out.owner.assign(rows.size(),-1);out.role.assign(rows.size(),"not_selected");
    std::map<std::string,std::vector<int>> chains;for(auto& c:graph.candidates)chains[c.name]=c.chain;
    std::vector<int> ids=selected;std::sort(ids.begin(),ids.end(),[&](int a,int b){return rows[a].id<rows[b].id;});
    std::vector<int> parent(rows.size());std::iota(parent.begin(),parent.end(),0);std::map<int,std::vector<int>> members;for(int i:ids)members[i]={i};
    std::function<int(int)> root=[&](int i){return parent[i]==i?i:parent[i]=root(parent[i]);};
    // Physical bench groups already contain spatial, height and surface evidence.
    // Here a missing measurement does not erase that group's identity.
    for(size_t x=0;x<ids.size();++x)for(size_t y=x+1;y<ids.size();++y){int i=ids[x],j=ids[y],a=root(i),b=root(j);if(a==b)continue;const auto& ci=chains.at(rows[i].name);const auto& cj=chains.at(rows[j].name);
        bool sameCrestEvidence=false;if(rows[i].upper>=0&&rows[i].upper==rows[j].upper){int shared=0;for(int g:ci)if(std::find(cj.begin(),cj.end(),g)!=cj.end())++shared;sameCrestEvidence=shared==1;}
        // One original crest can be split into several sampled bench groups.
        // Its identity plus a common physical bench anchors these fragments.
        if(!sharesOrderedBenches(ci,cj)&&!sameCrestEvidence)continue;
        bool compatible=true;for(int p:members[a])for(int q:members[b])if(!aligned(rows[p],rows[q]))compatible=false;
        if(!compatible)continue;parent[b]=a;members[a].insert(members[a].end(),members[b].begin(),members[b].end());members.erase(b);
    }
    std::map<int,int> wallIds;for(int i:ids){int r=root(i);if(!wallIds.count(r)){wallIds[r]=int(out.walls.size());out.walls.push_back({});}int w=wallIds[r];out.owner[i]=w;out.walls[w].members.push_back(i);}
    for(auto& wall:out.walls){std::map<int,std::set<int>> next,previous;
        for(int i:wall.members){auto& chain=chains.at(rows[i].name);for(int g:chain)wall.groups.insert(g);for(size_t k=1;k<chain.size();++k){next[chain[k-1]].insert(chain[k]);previous[chain[k]].insert(chain[k-1]);}}
        // Selected short sections are incomplete witnesses, not the topology.
        // Retain supported links between owned benches even when no selected
        // section traversed them. Their geometry is rechecked per ray downstream.
        for(auto& link:graph.links)if(wall.groups.count(link.upper)&&wall.groups.count(link.lower)){next[link.upper].insert(link.lower);previous[link.lower].insert(link.upper);}
        for(int g:wall.groups){if(previous[g].empty())wall.heads.insert(g);if(next[g].empty())wall.tails.insert(g);}
        wall.ordered=wall.heads.size()==1&&wall.tails.size()==1;for(int g:wall.groups)if(next[g].size()>1||previous[g].size()>1)wall.ordered=false;
        for(int i:wall.members){auto& chain=chains.at(rows[i].name);out.role[i]=!wall.ordered?"boundary_branch_unresolved":chain.front()==*wall.heads.begin()&&chain.back()==*wall.tails.begin()?"outer_sample_unconfirmed":"internal_partial_section";}
    }return out;
}
inline void save(const Result& result,const std::vector<overall::Preview>& rows,const std::vector<int>& localAreas,const std::string& prefix){
    std::ofstream members(prefix+"workface_membership.csv"),walls(prefix+"workface_ownership.csv");if(!members||!walls)throw std::runtime_error("Cannot save ownership");
    members<<"section_id,workface,local_area,role,scope\n";for(size_t i=0;i<rows.size();++i)members<<rows[i].id<<','<<result.owner[i]<<','<<localAreas[i]<<','<<result.role[i]<<",candidate_production_extent_unconfirmed\n";
    auto writeSet=[](std::ostream& s,const std::set<int>& v){bool first=true;for(int x:v){if(!first)s<<';';first=false;s<<x;}};
    walls<<"workface,sections,local_areas,bench_groups,upper_groups,lower_groups,boundary_status\n";
    for(size_t i=0;i<result.walls.size();++i){auto& w=result.walls[i];std::set<int> areas;for(int k:w.members)areas.insert(localAreas[k]);walls<<i<<','<<w.members.size()<<','<<areas.size()<<',';writeSet(walls,w.groups);walls<<',';writeSet(walls,w.heads);walls<<',';writeSet(walls,w.tails);walls<<','<<(w.ordered?"ordered_candidate_extent_unconfirmed":"branch_extent_unresolved")<<'\n';}
}
}
