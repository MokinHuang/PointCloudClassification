#pragma once
#include "FeatureLine.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

// Independent measurement stage. Never edits the input lines or classification.
namespace sections {
using V = Eigen::Vector3d;
using XY = Eigen::Vector2d;
struct Options {
    double spacing=5.0, tangentWindow=3.0, maxDistance=40.0;
    double minDrop=0.8, minSlopeSupport=0.75, minObservedSupport=0.9;
    double maxBarrierRun=1.5, uniqueDistance=0.5;
};
struct Sample {
    int id, crestLine, toeLine=-1, toeSegment=-1, orderSign=1;
    double crestS=0, toeS=0, toeT=0, width=0, height=0, angle=0, length=0;
    double support=0, observed=0, score=0;
    V crest=V::Zero(), toe=V::Zero();
    std::string status="no_normal_intersection";
};
struct Line {
    int id,type; std::vector<V> p; std::vector<double> s;
};
inline double cross(const XY& a,const XY& b){return a.x()*b.y()-a.y()*b.x();}
inline XY xy(const V& p){return p.head<2>();}
inline V at(const Line& l,double s){
    auto it=std::upper_bound(l.s.begin(),l.s.end(),s);
    size_t j=std::min(l.s.size()-1,std::max(size_t(1),size_t(it-l.s.begin())));
    double t=(s-l.s[j-1])/(l.s[j]-l.s[j-1]);
    return l.p[j-1]+std::clamp(t,0.0,1.0)*(l.p[j]-l.p[j-1]);
}
// Ray a+t*n with segment b+u*(c-b), all in local double coordinates.
inline bool hit(const XY& a,const XY& n,const XY& b,const XY& c,double& t,double& u){
    XY e=c-b; double d=cross(n,e); if(std::abs(d)<1e-10)return false;
    t=cross(b-a,e)/d;u=cross(b-a,n)/d;
    return t>1e-6&&u>=-1e-8&&u<=1+1e-8;
}
inline bool crossing(const Sample& a,const Sample& b){
    XY p=xy(a.crest),r=xy(a.toe)-p,q=xy(b.crest),s=xy(b.toe)-q;
    double d=cross(r,s);
    if(std::abs(d)<1e-9){
        if(std::abs(cross(q-p,r))>1e-7)return false;
        double n=r.squaredNorm(); if(n<1e-12)return true;
        double t0=(q-p).dot(r)/n,t1=(q+s-p).dot(r)/n;
        return std::max(0.0,std::min(t0,t1))<=std::min(1.0,std::max(t0,t1))+1e-8;
    }
    double t=cross(q-p,s)/d,u=cross(q-p,r)/d;
    return t>=-1e-8&&t<=1+1e-8&&u>=-1e-8&&u<=1+1e-8;
}
inline std::vector<Sample> measure(const std::vector<FeatureLine>& source,
    const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin,
    const Options& opt=Options{}){
    if(cell<=0||opt.spacing<=0||opt.maxDistance<=cell)throw std::invalid_argument("Invalid measurement spacing/range");
    std::vector<Sample> result;if(grid.empty()||grid[0].empty())return result;
    V offset(origin.x(),origin.y(),0); std::vector<Line> lines;
    for(const auto& src:source){
        Line l; l.id=src.id;l.type=src.type;
        for(const auto& point:src.points){V p=point.cast<double>()-offset;
            if(!p.allFinite())continue;
            if(!l.p.empty()&&(xy(p)-xy(l.p.back())).norm()<1e-6)continue;
            double s=l.p.empty()?0:l.s.back()+(xy(p)-xy(l.p.back())).norm();
            l.p.push_back(p);l.s.push_back(s);
        } if(l.p.size()>1)lines.push_back(l);
    }
    auto lookup=[&](const V& p)->const GridCell* {
        int c=int(std::floor(p.x()/cell)),r=int(std::floor(p.y()/cell));
        if(r<0||r>=int(grid.size())||c<0||c>=int(grid[r].size()))return nullptr;
        return &grid[r][c];
    };
    // Use observed, unsmoothed cell elevations. No invented height in missing cells.
    auto elevation=[&](V& p){const auto* g=lookup(p);if(!g||g->is_empty||g->count<=0||!std::isfinite(g->avg_z))return false;p.z()=g->avg_z;return true;};
    std::vector<Sample> candidates;
    for(const auto& l:lines)if(l.type==0){
        for(double s=0;s<=l.s.back()+1e-8;s+=opt.spacing){
            Sample a;a.id=int(result.size());a.crestLine=l.id;a.crestS=s;a.crest=at(l,s);
            XY tangent=xy(at(l,std::min(l.s.back(),s+opt.tangentWindow)))-xy(at(l,std::max(0.0,s-opt.tangentWindow)));
            if(tangent.norm()<cell*.25){a.status="unstable_tangent";result.push_back(a);continue;}
            if(!elevation(a.crest)){a.status="unobserved_crest";result.push_back(a);continue;}
            tangent.normalize();XY normal(-tangent.y(),tangent.x());
            int validSides=0;Sample chosen;
            for(int sign:{-1,1}){
                XY direction=normal*sign;double nearest=opt.maxDistance,uBest=0;const Line* target=nullptr;size_t seg=0;
                // Stop at the FIRST other boundary, including intervening crest lines.
                for(const auto& b:lines)for(size_t j=1;j<b.p.size();++j){
                    double t,u;if(hit(xy(a.crest),direction,xy(b.p[j-1]),xy(b.p[j]),t,u)&&t>cell*.5&&t<nearest){
                        nearest=t;uBest=std::clamp(u,0.0,1.0);target=&b;seg=j;
                    }
                }
                if(!target)continue;
                if(target->type!=1){a.status="intervening_crest";continue;}
                Sample b=a;b.toe=target->p[seg-1]+uBest*(target->p[seg]-target->p[seg-1]);
                if(!elevation(b.toe)){a.status="unobserved_toe";continue;}
                b.height=b.crest.z()-b.toe.z();b.width=nearest;
                if(b.height<opt.minDrop){a.status="insufficient_drop";continue;}
                XY tt=xy(target->p[seg])-xy(target->p[seg-1]);tt.normalize();
                double alignment=std::abs(tt.dot(tangent));
                b.orderSign=tt.dot(tangent)>=0?1:-1;
                if(alignment<0.5){a.status="incompatible_boundary_direction";continue;}
                int total=0,observed=0,slope=0;double badRun=0,maxRun=0;
                // Exclude one boundary cell at either end; tolerate short classification holes.
                double step=cell*.5;
                for(double d=cell;d<nearest-cell*.5;d+=step){
                    V p=b.crest;p.head<2>()+=direction*d;auto g=lookup(p);++total;
                    bool seen=g&&!g->is_empty&&g->count>0;if(seen)++observed;
                    bool isSlope=seen&&g->type==2;if(isSlope)++slope;
                    badRun=isSlope?0:badRun+step;maxRun=std::max(maxRun,badRun);
                }
                if(!total){a.status="section_too_narrow";continue;}
                b.support=double(slope)/total;b.observed=double(observed)/total;
                if(b.observed<opt.minObservedSupport){a.status="missing_surface_data";continue;}
                if(b.support<opt.minSlopeSupport||maxRun>opt.maxBarrierRun){a.status="platform_berm_or_surface_gap";continue;}
                b.toeLine=target->id;b.toeSegment=int(seg-1);b.toeT=uBest;
                b.toeS=target->s[seg-1]+uBest*(target->s[seg]-target->s[seg-1]);
                b.angle=std::atan2(b.height,b.width)*180/3.14159265358979323846;
                b.length=std::hypot(b.width,b.height);
                b.score=b.support+b.observed+alignment;
                b.status="matched";chosen=b;++validSides;
            }
            if(validSides==1){candidates.push_back(chosen);a.status="uniqueness_or_crossing_conflict";}
            else if(validSides>1)a.status="ambiguous_downhill_sides";
            result.push_back(a);
        }
    }
    // Conservative quality-first conflict resolution, NOT a maximum-cardinality assignment.
    // No forced fallback to a farther toe after rejecting the first boundary.
    std::stable_sort(candidates.begin(),candidates.end(),[](const Sample& a,const Sample& b){return a.score>b.score;});
    std::vector<Sample> accepted;
    for(const auto& a:candidates){bool conflict=false;
        for(const auto& b:accepted){
            if(a.crestLine==b.crestLine&&a.toeLine==b.toeLine&&
               (a.orderSign!=b.orderSign||(a.crestS-b.crestS)*(a.toeS-b.toeS)*a.orderSign<=0)){
                conflict=true;break;
            }
            if((xy(a.crest)-xy(b.crest)).norm()<opt.uniqueDistance||
               (xy(a.toe)-xy(b.toe)).norm()<opt.uniqueDistance||crossing(a,b)){conflict=true;break;}
        }
        if(!conflict){result[a.id]=a;accepted.push_back(a);}
    }
    for(auto& a:result){a.crest+=offset;if(a.status=="matched")a.toe+=offset;}
    return result;
}
inline void save(const std::vector<Sample>& samples,const std::string& prefix=""){
    std::ofstream csv(prefix+"matched_sections.csv"),unmatched(prefix+"unmatched_points.csv"),ply(prefix+"matched_sections.ply");
    if(!csv||!unmatched||!ply)throw std::runtime_error("Cannot open measurement outputs");
    csv<<std::fixed<<std::setprecision(8);unmatched<<std::fixed<<std::setprecision(8);ply<<std::fixed<<std::setprecision(8);
    csv<<"pair_id,crest_line_id,crest_station,toe_line_id,toe_segment,toe_fraction,toe_station,crest_x,crest_y,crest_z,toe_x,toe_y,toe_z,h_dist,v_drop,angle,straight_length,slope_percent,horizontal_per_vertical,slope_support,observed_support\n";
    unmatched<<"sample_id,crest_line_id,crest_station,x,y,z,reason\n";
    size_t n=0;for(auto& a:samples)if(a.status=="matched")++n;
    ply<<"ply\nformat ascii 1.0\nelement vertex "<<2*n<<"\nproperty double x\nproperty double y\nproperty double z\nelement edge "<<n<<"\nproperty int vertex1\nproperty int vertex2\nend_header\n";
    for(auto& a:samples){if(a.status!="matched"){
        unmatched<<a.id<<','<<a.crestLine<<','<<a.crestS<<','<<a.crest.x()<<','<<a.crest.y()<<','<<a.crest.z()<<','<<a.status<<'\n';continue;}
        csv<<a.id<<','<<a.crestLine<<','<<a.crestS<<','<<a.toeLine<<','<<a.toeSegment<<','<<a.toeT<<','<<a.toeS<<','
           <<a.crest.x()<<','<<a.crest.y()<<','<<a.crest.z()<<','<<a.toe.x()<<','<<a.toe.y()<<','<<a.toe.z()<<','
           <<a.width<<','<<a.height<<','<<a.angle<<','<<a.length<<','<<100*a.height/a.width<<','<<a.width/a.height<<','<<a.support<<','<<a.observed<<'\n';
        for(const auto& p:{a.crest,a.toe})ply<<p.x()<<' '<<p.y()<<' '<<p.z()<<'\n';
    }
    for(size_t i=0;i<n;++i)ply<<2*i<<' '<<2*i+1<<'\n';
}
}
