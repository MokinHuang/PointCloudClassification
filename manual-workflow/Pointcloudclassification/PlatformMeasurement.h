#pragma once
#include "SectionMeasurement.h"

namespace platforms {
using namespace sections;
struct Options {
    double maxDistance=100, minPlatformSupport=.60, minSurfaceSupport=.90;
    double minObservedSupport=.90, maxBadRun=1.5, uniqueDistance=.5;
    double heightTolerance=1.5, maxPlatformGradeDegrees=8;
};
struct Measurement {
    Sample geometry;
    double platformSupport=0, bermFraction=0;
};
inline std::vector<Measurement> measure(const std::vector<Sample>& slopes,
    const std::vector<FeatureLine>& source,const std::vector<std::vector<GridCell>>& grid,
    double cell,const Eigen::Vector4f& origin,const Options& opt=Options{}) {
    if(cell<=0||opt.maxDistance<=cell)throw std::invalid_argument("Invalid platform options");
    std::vector<Measurement> results;std::vector<size_t> candidates;
    V offset(origin.x(),origin.y(),0);std::vector<Line> lines;
    for(const auto& src:source){Line l;l.id=src.id;l.type=src.type;
        for(const auto& p:src.points){V v=p.cast<double>()-offset;
            if(!v.allFinite()||(!l.p.empty()&&(xy(v)-xy(l.p.back())).norm()<1e-6))continue;
            l.s.push_back(l.p.empty()?0:l.s.back()+(xy(v)-xy(l.p.back())).norm());l.p.push_back(v);
        }if(l.p.size()>1)lines.push_back(l);
    }
    auto lookup=[&](const V& p)->const GridCell*{
        int r=int(std::floor(p.y()/cell)),c=int(std::floor(p.x()/cell));
        if(r<0||r>=int(grid.size())||c<0||c>=int(grid[r].size()))return nullptr;return &grid[r][c];
    };
    auto observed=[](const GridCell* g){return g&&!g->is_empty&&g->count>0&&std::isfinite(g->avg_z);};
    for(const auto& slope:slopes){
        Measurement m;auto& a=m.geometry;a.id=slope.id;a.crestLine=slope.crestLine;
        a.crestS=slope.crestS;a.crest=slope.crest-offset;
        auto reject=[&](const char* reason){a.status=reason;results.push_back(m);};
        if(slope.status!="matched"){reject("no_reliable_slope_direction");continue;}
        XY direction=xy(slope.crest)-xy(slope.toe);
        if(direction.norm()<cell*.5){reject("unstable_direction");continue;}direction.normalize();
        XY tangent(-direction.y(),direction.x());
        // Obtain orientation of the crest station sequence, not the ray handedness.
        for(const auto& l:lines)if(l.id==a.crestLine&&l.type==0){
            XY t=xy(at(l,std::min(l.s.back(),a.crestS+3)))-xy(at(l,std::max(0.0,a.crestS-3)));
            if(t.norm()>1e-8)tangent=t.normalized();break;
        }
        double nearest=opt.maxDistance,uBest=0;size_t seg=0;const Line* target=nullptr;
        for(const auto& l:lines)for(size_t j=1;j<l.p.size();++j){double t,u;
            if(hit(xy(a.crest),direction,xy(l.p[j-1]),xy(l.p[j]),t,u)&&t>cell*.5&&t<nearest){
                nearest=t;uBest=std::clamp(u,0.0,1.0);target=&l;seg=j;
            }
        }
        if(!target){reject("no_boundary_within_range");continue;}
        if(target->type!=1){reject("intervening_crest");continue;}
        a.toe=target->p[seg-1]+uBest*(target->p[seg]-target->p[seg-1]);
        const auto* gc=lookup(a.crest);const auto* gt=lookup(a.toe);
        if(!observed(gc)||!observed(gt)){reject("unobserved_endpoint");continue;}
        a.crest.z()=gc->avg_z;a.toe.z()=gt->avg_z;a.width=nearest;a.height=a.toe.z()-a.crest.z();
        if(std::abs(a.height)>opt.heightTolerance+nearest*std::tan(opt.maxPlatformGradeDegrees*3.141592653589793/180)){
            reject("excessive_endpoint_elevation_change");continue;
        }
        XY tt=xy(target->p[seg])-xy(target->p[seg-1]);tt.normalize();
        double alignment=std::abs(tt.dot(tangent));a.orderSign=tt.dot(tangent)>=0?1:-1;
        if(alignment<.5){reject("incompatible_boundary_direction");continue;}
        int total=0,seen=0,flat=0,berm=0;double run=0,maxRun=0;
        for(double d=cell;d<nearest-cell*.5;d+=cell*.5){V p=a.crest;p.head<2>()+=direction*d;
            const auto* g=lookup(p);++total;bool valid=observed(g);if(valid)++seen;
            if(valid&&g->type==1)++flat;if(valid&&g->type==4)++berm;
            bool support=valid&&(g->type==1||g->type==4);run=support?0:run+cell*.5;maxRun=std::max(maxRun,run);
        }
        if(!total){reject("platform_too_narrow");continue;}
        m.platformSupport=double(flat)/total;m.bermFraction=double(berm)/total;
        a.observed=double(seen)/total;a.support=double(flat+berm)/total;
        if(a.observed<opt.minObservedSupport){reject("missing_surface_data");continue;}
        if(m.platformSupport<opt.minPlatformSupport||a.support<opt.minSurfaceSupport||maxRun>opt.maxBadRun){
            reject("not_a_supported_platform");continue;
        }
        a.toeLine=target->id;a.toeSegment=int(seg-1);a.toeT=uBest;
        a.toeS=target->s[seg-1]+uBest*(target->s[seg]-target->s[seg-1]);
        a.score=m.platformSupport+a.observed+alignment;a.status="uniqueness_order_or_crossing_conflict";
        candidates.push_back(results.size());results.push_back(m);
    }
    std::stable_sort(candidates.begin(),candidates.end(),[&](size_t a,size_t b){return results[a].geometry.score>results[b].geometry.score;});
    std::vector<size_t> accepted;
    for(auto i:candidates){auto& a=results[i].geometry;bool conflict=false;
        for(auto j:accepted){const auto& b=results[j].geometry;
            if((a.crestLine==b.crestLine&&a.toeLine==b.toeLine&&
                (a.orderSign!=b.orderSign||(a.crestS-b.crestS)*(a.toeS-b.toeS)*a.orderSign<=0))||
                (xy(a.crest)-xy(b.crest)).norm()<opt.uniqueDistance||
                (xy(a.toe)-xy(b.toe)).norm()<opt.uniqueDistance||crossing(a,b)){conflict=true;break;}
        }
        if(!conflict){a.status="matched";accepted.push_back(i);}
    }
    for(auto& m:results){m.geometry.crest+=offset;if(m.geometry.status=="matched")m.geometry.toe+=offset;}
    return results;
}
inline void save(const std::vector<Measurement>& data,const std::string& prefix=""){
    std::ofstream csv(prefix+"platform_widths.csv"),unmatched(prefix+"platform_unmatched.csv"),ply(prefix+"platform_sections.ply");
    if(!csv||!unmatched||!ply)throw std::runtime_error("Cannot open platform outputs");
    csv<<std::fixed<<std::setprecision(8);unmatched<<std::fixed<<std::setprecision(8);ply<<std::fixed<<std::setprecision(8);
    csv<<"sample_id,crest_line_id,crest_station,toe_line_id,toe_segment,toe_fraction,toe_station,crest_x,crest_y,crest_z,toe_x,toe_y,toe_z,platform_width,elevation_change,platform_support,berm_fraction,surface_support,observed_support\n";
    unmatched<<"sample_id,crest_line_id,crest_station,x,y,z,reason\n";
    size_t n=0;for(auto& m:data)if(m.geometry.status=="matched")++n;
    ply<<"ply\nformat ascii 1.0\nelement vertex "<<n*2<<"\nproperty double x\nproperty double y\nproperty double z\nelement edge "<<n<<"\nproperty int vertex1\nproperty int vertex2\nend_header\n";
    for(auto& m:data){const auto& a=m.geometry;
        if(a.status!="matched"){unmatched<<a.id<<','<<a.crestLine<<','<<a.crestS<<','<<a.crest.x()<<','<<a.crest.y()<<','<<a.crest.z()<<','<<a.status<<'\n';continue;}
        csv<<a.id<<','<<a.crestLine<<','<<a.crestS<<','<<a.toeLine<<','<<a.toeSegment<<','<<a.toeT<<','<<a.toeS<<','
           <<a.crest.x()<<','<<a.crest.y()<<','<<a.crest.z()<<','<<a.toe.x()<<','<<a.toe.y()<<','<<a.toe.z()<<','
           <<a.width<<','<<a.height<<','<<m.platformSupport<<','<<m.bermFraction<<','<<a.support<<','<<a.observed<<'\n';
        for(auto& p:{a.crest,a.toe})ply<<p.x()<<' '<<p.y()<<' '<<p.z()<<'\n';
    }for(size_t i=0;i<n;++i)ply<<2*i<<' '<<2*i+1<<'\n';
}
}
