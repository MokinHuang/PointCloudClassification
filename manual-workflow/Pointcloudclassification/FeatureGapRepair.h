#pragma once
#include "FeatureLine.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

// Review-only inferred geometry. Never replace measured lines implicitly.
namespace gaprepair {
using V=Eigen::Vector3d;
using XY=Eigen::Vector2d;
struct Options {
    bool localFit=false;
    double maxGap=5.0, maxZResidual=0.75, tangentSpan=5.0;
    double maxAngleDegrees=30.0, boundaryRadius=1.5, junctionRadius=0.3;
};
struct End {int line,side; V p,d; bool valid=false,open=true;};
struct Candidate {int a,b; double gap=0,zResidual=0,angle=0; std::string status; std::string originalStatus,method="legacy";};
struct Result {std::vector<End> ends; std::vector<Candidate> candidates;};
inline double distance(const XY& p,const XY& a,const XY& b) {
    XY d=b-a; double t=d.squaredNorm()>0?std::clamp((p-a).dot(d)/d.squaredNorm(),0.0,1.0):0;
    return (p-a-t*d).norm();
}
inline double cross(const XY&a,const XY&b){return a.x()*b.y()-a.y()*b.x();}
inline bool intersects(const XY&a,const XY&b,const XY&c,const XY&d){
    if(std::max(a.x(),b.x())+1e-8<std::min(c.x(),d.x())||std::max(c.x(),d.x())+1e-8<std::min(a.x(),b.x())||
       std::max(a.y(),b.y())+1e-8<std::min(c.y(),d.y())||std::max(c.y(),d.y())+1e-8<std::min(a.y(),b.y()))return false;
    return cross(b-a,c-a)*cross(b-a,d-a)<=1e-10 && cross(d-c,a-c)*cross(d-c,b-c)<=1e-10;
}
inline V local(const Eigen::Vector3f&p,const Eigen::Vector4f&o){return p.cast<double>()-V(o.x(),o.y(),0);}
// Candidate-independent endpoint regression, in physical arclength; two scales must agree.
inline bool fittedDirection(const FeatureLine& l,int side,const Eigen::Vector4f& origin,double cell,V& direction){
    auto fit=[&](double limit,V& out){
        std::vector<V> ps;std::vector<double> ss;int start=side?int(l.points.size())-1:0,step=side?-1:1;
        double span=0;V prev=local(l.points[start],origin);ps.push_back(prev);ss.push_back(0);
        for(int j=start+step;j>=0&&j<int(l.points.size());j+=step){V q=local(l.points[j],origin);double len=(q-prev).head<2>().norm();if(len<1e-8)continue;
            if(span+len>=limit){ps.push_back(prev+(q-prev)*(limit-span)/len);ss.push_back(limit);span=limit;break;}
            span+=len;ps.push_back(q);ss.push_back(span);prev=q;}
        if(ps.size()<3||span<2*cell)return false;
        double mean=std::accumulate(ss.begin(),ss.end(),0.0)/ss.size(),den=0;V center=V::Zero(),slope=V::Zero();for(auto& p:ps)center+=p;center/=double(ps.size());
        for(size_t k=0;k<ps.size();k++){double t=ss[k]-mean;den+=t*t;slope+=t*(ps[k]-center);}if(den<1e-8)return false;slope/=den;
        double xy=slope.head<2>().norm();if(xy<0.85)return false;
        for(size_t k=0;k<ps.size();k++){V e=ps[k]-(center+slope*(ss[k]-mean));if(e.head<2>().norm()>0.35*cell||std::abs(e.z())>0.35)return false;}
        out=-slope/xy;return true;
    };
    V near,far;if(!fit(2*cell,near)||!fit(3*cell,far))return false;
    if(near.head<2>().dot(far.head<2>())<std::cos(15.0*3.141592653589793/180))return false;
    direction=near;return true;
}
inline Result repair(const std::vector<FeatureLine>& lines,const std::vector<std::vector<GridCell>>& grid,
                     double cell,const Eigen::Vector4f& origin,const Options& opt={}) {
    if(!(cell>0&&std::isfinite(cell)&&opt.maxGap>0&&opt.maxZResidual>=0&&opt.tangentSpan>0&&
         opt.maxAngleDegrees>0&&opt.maxAngleDegrees<=180&&opt.boundaryRadius>=0&&opt.junctionRadius>=0))
        throw std::invalid_argument("Invalid gap repair options");
    std::set<int> ids;
    for(const auto& l:lines){if(!ids.insert(l.id).second||l.type<0||l.type>1)throw std::invalid_argument("Invalid line ID/type");
        for(const auto& p:l.points)if(!p.allFinite())throw std::invalid_argument("Non-finite feature point");}
    Result out;
    for(int i=0;i<int(lines.size());++i)for(int side=0;side<2;++side){
        End e;e.line=i;e.side=side;e.p=V::Zero();e.d=V::Zero();const auto& p=lines[i].points;
        if(p.size()>1){int start=side?int(p.size())-1:0,step=side?-1:1;e.p=local(p[start],origin);
            double span=0;V previous=e.p;
            for(int j=start+step;j>=0&&j<int(p.size());j+=step){V q=local(p[j],origin);span+=(q-previous).head<2>().norm();previous=q;
                if(span>=opt.tangentSpan){V delta=e.p-q;double xy=delta.head<2>().norm();
                    if(xy>=0.8*opt.tangentSpan){e.d=delta/xy;e.valid=true;}break;}}
        }out.ends.push_back(e);
    }
    // An endpoint already touching another line is a junction, not a gap.
    for(auto& e:out.ends)for(int j=0;j<int(lines.size())&&e.open;++j)if(j!=e.line)
        for(size_t k=1;k<lines[j].points.size();++k)
            if(distance(e.p.head<2>(),local(lines[j].points[k-1],origin).head<2>(),local(lines[j].points[k],origin).head<2>())<=opt.junctionRadius){e.open=false;break;}
    auto observed=[&](const V&p)->const GridCell*{
        int r=int(std::floor(p.y()/cell)),c=int(std::floor(p.x()/cell));
        if(r<0||r>=int(grid.size())||c<0||c>=int(grid[r].size()))return nullptr;
        auto& g=grid[r][c];return !g.is_empty&&g.count>0&&std::isfinite(g.avg_z)?&g:nullptr;
    };
    auto boundary=[&](const V&p,int type){
        int r=int(std::floor(p.y()/cell)),c=int(std::floor(p.x()/cell)),n=int(std::ceil(opt.boundaryRadius/cell));
        for(int y=r-n;y<=r+n;++y)for(int x=c-n;x<=c+n;++x){
            V q((x+.5)*cell,(y+.5)*cell,p.z());if((q-p).head<2>().norm()>opt.boundaryRadius)continue;
            auto g=observed(q);if(g&&(type==0?g->is_crest:g->is_toe)&&std::abs(g->avg_z-p.z())<=opt.maxZResidual)return true;
        }return false;
    };
    for(int a=0;a<int(out.ends.size());++a)for(int b=a+1;b<int(out.ends.size());++b){
        auto ea=out.ends[a];auto eb=out.ends[b];if(ea.line==eb.line||lines[ea.line].type!=lines[eb.line].type)continue;
        V delta=eb.p-ea.p;double gap=delta.head<2>().norm();if(gap>opt.maxGap||gap<1e-6)continue;
        Candidate c;c.a=a;c.b=b;c.gap=gap;c.status="eligible";
        if(!ea.valid||!eb.valid)c.status="short_or_curved_tangent";
        else if(!ea.open||!eb.open)c.status="existing_junction";
        else {
            XY dir=delta.head<2>()/gap;
            c.angle=std::acos(std::clamp(std::min(ea.d.head<2>().dot(dir),eb.d.head<2>().dot(-dir)),-1.0,1.0))*180/3.141592653589793;
            c.zResidual=std::max(std::abs(ea.p.z()+ea.d.z()*gap-eb.p.z()),std::abs(eb.p.z()+eb.d.z()*gap-ea.p.z()));
            if(c.angle>opt.maxAngleDegrees)c.status="direction";
            else if(c.zResidual>opt.maxZResidual)c.status="elevation_trend";
        }
        c.originalStatus=c.status;
        if(opt.localFit&&(c.status=="direction"||c.status=="short_or_curved_tangent")){
            V da,db;
            if(fittedDirection(lines[ea.line],ea.side,origin,cell,da)&&fittedDirection(lines[eb.line],eb.side,origin,cell,db)){
                c.method="local_regression_2_3_cells";
                XY dir=delta.head<2>()/gap;
                c.angle=std::acos(std::clamp(std::min(da.head<2>().dot(dir),db.head<2>().dot(-dir)),-1.0,1.0))*180/3.141592653589793;
                c.zResidual=std::max(std::abs(ea.p.z()+da.z()*gap-eb.p.z()),std::abs(eb.p.z()+db.z()*gap-ea.p.z()));
                if(!ea.open||!eb.open)c.status="existing_junction";
                else if(c.angle>opt.maxAngleDegrees)c.status="direction";
                else if(c.zResidual>opt.maxZResidual)c.status="elevation_trend";
                else c.status="eligible";
            }
        }
        if(c.status=="eligible"){
            int steps=std::max(2,int(std::ceil(gap/(cell*.5))));
            for(int k=0;k<=steps;++k){V p=ea.p+delta*(double(k)/steps);
                if(!observed(p)){c.status="unobserved_gap";break;}
                if(!boundary(p,lines[ea.line].type)){c.status="no_boundary_support";break;}}
        }
        if(c.status=="eligible")for(int j=0;j<int(lines.size())&&c.status=="eligible";++j)
            for(size_t k=1;k<lines[j].points.size();++k){
                // Allow only the single incident segment at each source endpoint.
                if((j==ea.line&&((ea.side==0&&k==1)||(ea.side==1&&k==lines[j].points.size()-1)))||
                   (j==eb.line&&((eb.side==0&&k==1)||(eb.side==1&&k==lines[j].points.size()-1))))continue;
                if(intersects(ea.p.head<2>(),eb.p.head<2>(),local(lines[j].points[k-1],origin).head<2>(),local(lines[j].points[k],origin).head<2>())){c.status="crosses_existing_line";break;}}
        out.candidates.push_back(c);
    }
    std::vector<int> degree(out.ends.size());
    for(auto& c:out.candidates)if(c.status=="eligible"){++degree[c.a];++degree[c.b];}
    for(auto& c:out.candidates)if(c.status=="eligible"&&(degree[c.a]!=1||degree[c.b]!=1))c.status="ambiguous_endpoint";
    // Reject BOTH competing bridges, independent of iteration order.
    std::set<size_t> conflicts;
    for(size_t i=0;i<out.candidates.size();++i)if(out.candidates[i].status=="eligible")
        for(size_t j=i+1;j<out.candidates.size();++j)if(out.candidates[j].status=="eligible"){
            auto a=out.candidates[i],b=out.candidates[j];if(intersects(out.ends[a.a].p.head<2>(),out.ends[a.b].p.head<2>(),out.ends[b.a].p.head<2>(),out.ends[b.b].p.head<2>())){conflicts.insert(i);conflicts.insert(j);}}
    for(auto i:conflicts)out.candidates[i].status="crosses_candidate";
    std::vector<int> parent(lines.size());std::iota(parent.begin(),parent.end(),0);
    auto root=[&](int a){while(parent[a]!=a)a=parent[a];return a;};
    for(auto& c:out.candidates)if(c.status=="eligible"){
        int a=root(out.ends[c.a].line),b=root(out.ends[c.b].line);
        if(a==b)c.status="cycle";else{parent[a]=b;c.status="accepted";}}
    return out;
}
inline void save(const Result& r,const std::vector<FeatureLine>& lines,const Eigen::Vector4f& origin,const std::string& prefix){
    std::ofstream log(prefix+"feature_gap_candidates.csv"), merged(prefix+"feature_lines_repaired.csv"),meta(prefix+"feature_gap_provenance.csv");
    if(!log||!merged||!meta)throw std::runtime_error("Cannot write gap repair outputs");
    log<<std::setprecision(12)<<"candidate_id,line_a,end_a,line_b,end_b,type,gap_xy,z_residual,angle,status,ax,ay,az,bx,by,bz,original_status,direction_method\n";
    std::map<int,int> links,bridge;
    for(size_t i=0;i<r.candidates.size();++i){auto& c=r.candidates[i];auto a=r.ends[c.a],b=r.ends[c.b];
        log<<i<<','<<lines[a.line].id<<','<<a.side<<','<<lines[b.line].id<<','<<b.side<<','<<lines[a.line].type<<','<<c.gap<<','<<c.zResidual<<','<<c.angle<<','<<c.status;
        for(auto p:{a.p,b.p})log<<','<<p.x()+origin.x()<<','<<p.y()+origin.y()<<','<<p.z();log<<','<<c.originalStatus<<','<<c.method<<'\n';
        if(c.status=="accepted"){links[c.a]=c.b;links[c.b]=c.a;bridge[c.a]=bridge[c.b]=int(i);}}
    merged<<std::setprecision(12)<<"line_id,type,type_name,point_order,x,y,z\n";
    meta<<"repaired_line_id,point_order,source_line_id,source_point_order,incoming_inferred_bridge_id\n";
    std::set<int> visited;
    auto emit=[&](int start){int end=start,id=lines[start/2].id,order=0,incoming=-1;
        while(true){int li=end/2;bool reverse=end%2!=0;if(visited.count(li))break;visited.insert(li);auto& l=lines[li];
            for(int k=0;k<int(l.points.size());++k){int index=reverse?int(l.points.size())-1-k:k;auto p=l.points[index];
                merged<<id<<','<<l.type<<','<<(l.type==0?"Crest":"Toe")<<','<<order<<','<<p.x()<<','<<p.y()<<','<<p.z()<<'\n';
                meta<<id<<','<<order++<<','<<l.id<<','<<index<<','<<(k==0?incoming:-1)<<'\n';}
            int exit=li*2+(reverse?0:1);auto next=links.find(exit);if(next==links.end())break;incoming=bridge[exit];end=next->second;
        }};
    for(int i=0;i<int(lines.size());++i)if(!visited.count(i)){
        if(!links.count(i*2))emit(i*2);else if(!links.count(i*2+1))emit(i*2+1);}
    if(visited.size()!=lines.size())throw std::runtime_error("Unexpected cyclic repair topology");
}
inline void run(const std::vector<FeatureLine>& lines,const std::vector<std::vector<GridCell>>& grid,double cell,
                const Eigen::Vector4f& origin,const std::string& prefix="",const Options& opt={}){
    save(repair(lines,grid,cell,origin,opt),lines,origin,prefix);
}
}
