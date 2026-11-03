#pragma once
#include "SectionMeasurement.h"
#include <map>
#include <set>
#include <sstream>
#include <filesystem>

namespace overall {
using namespace sections;
struct Selection { std::string name;int upper,lower;double start=0,end=-1,spacing=5,maxDistance=1500; };
struct Profile { double distance,z;int type,count;bool observed; };
struct Preview { int id=0,upper=-1,lower=-1,upperType=-1,lowerType=-1;std::string name,status;double station=0,width=0,coverage=0,height=0,angle=0;bool geometryAvailable=false;
    V top=V::Zero(),bottom=V::Zero();int slopeRuns=0;bool hasTarget=false;std::vector<Profile> profile; };
inline std::vector<Line> localLines(const std::vector<FeatureLine>& source,const Eigen::Vector4f& origin){
    V offset(origin.x(),origin.y(),0);std::vector<Line> out;std::set<int> ids;
    for(const auto& src:source){if(!ids.insert(src.id).second)throw std::runtime_error("Duplicate feature line ID");
        Line l;l.id=src.id;l.type=src.type;
        for(const auto& p:src.points){V v=p.cast<double>()-offset;if(!v.allFinite())throw std::runtime_error("Nonfinite feature coordinate");
            if(!l.p.empty()&&(xy(v)-xy(l.p.back())).norm()<1e-6)continue;
            l.s.push_back(l.p.empty()?0:l.s.back()+(xy(v)-xy(l.p.back())).norm());l.p.push_back(v);
        }if(l.p.size()>1)out.push_back(l);
    }return out;
}
inline std::vector<Selection> readSelections(const std::string& path){
    std::ifstream f(path);if(!f)throw std::runtime_error("Cannot read selected workface configuration");
    std::string row;if(!std::getline(f,row))throw std::runtime_error("Empty workface configuration");
    if(!row.empty()&&row.back()=='\r')row.pop_back();
    if(row!="workface_id,upper_line_id,lower_line_id,start_m,end_m,spacing_m,max_distance_m")throw std::runtime_error("Unexpected workface CSV header");
    std::vector<Selection> out;std::set<std::string> names;
    while(std::getline(f,row)){if(row.empty()||row=="\r")continue;std::replace(row.begin(),row.end(),',',' ');std::istringstream ss(row);Selection a;std::string extra;
        if(!(ss>>a.name>>a.upper>>a.lower>>a.start>>a.end>>a.spacing>>a.maxDistance)||(ss>>extra))throw std::runtime_error("Invalid workface CSV row");
        if(a.name.size()>64||a.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos||!names.insert(a.name).second)
            throw std::runtime_error("Workface names must be unique ASCII letters/digits/underscore/hyphen");
        if(!std::isfinite(a.start)||!std::isfinite(a.end)||!std::isfinite(a.spacing)||!std::isfinite(a.maxDistance)||a.start<0||(a.end<0&&a.end!=-1)||a.spacing<1||a.maxDistance<=0||a.maxDistance>10000)
            throw std::runtime_error("Invalid workface sampling/range");
        out.push_back(a);
    }return out;
}
inline std::vector<Preview> measure(const std::vector<FeatureLine>& source,const std::vector<std::vector<GridCell>>& grid,
    double cell,const Eigen::Vector4f& origin,const std::vector<Selection>& choices){
    if(cell<=0||grid.empty()||grid[0].empty())throw std::runtime_error("Missing overall preview grid");
    auto lines=localLines(source,origin);V offset(origin.x(),origin.y(),0);std::vector<Preview> out;
    auto lookup=[&](const V& p)->const GridCell*{int r=int(std::floor(p.y()/cell)),c=int(std::floor(p.x()/cell));
        if(r<0||r>=int(grid.size())||c<0||c>=int(grid[r].size()))return nullptr;return &grid[r][c];};
    auto valid=[](const GridCell* g){return g&&!g->is_empty&&g->count>0&&std::isfinite(g->avg_z);};
    for(const auto& choice:choices){const Line* upper=nullptr;const Line* lower=nullptr;
        for(const auto& l:lines){if(l.id==choice.upper)upper=&l;if(l.id==choice.lower)lower=&l;}
        if(!upper||!lower||upper==lower||upper->type<0||upper->type>1||lower->type<0||lower->type>1)throw std::runtime_error("Select two distinct existing crest/toe boundaries");
        double end=choice.end<0?upper->s.back():choice.end;
        if(choice.spacing<1||choice.maxDistance<=0||!std::isfinite(choice.maxDistance)||!std::isfinite(choice.start)||!std::isfinite(choice.spacing)||!std::isfinite(end)||(choice.end<0&&choice.end!=-1)||choice.start<0||choice.start>end||end>upper->s.back()+1e-6)
            throw std::runtime_error("Selected station range is outside the upper boundary");
        for(double station=choice.start;station<=end+1e-8;station+=choice.spacing){Preview a;a.id=int(out.size());a.name=choice.name;a.upper=choice.upper;a.lower=choice.lower;a.upperType=upper->type;a.lowerType=lower->type;a.station=station;a.top=at(*upper,station);
            auto finish=[&](){a.top+=offset;if(a.hasTarget)a.bottom+=offset;out.push_back(a);};
            XY tangent=xy(at(*upper,std::min(upper->s.back(),station+15)))-xy(at(*upper,std::max(0.0,station-15)));
            if(tangent.norm()<cell*.25){a.status="unstable_upper_direction";finish();continue;}
            tangent.normalize();XY normal(-tangent.y(),tangent.x());int hits=0;V chosen;double chosenDist=0;
            for(int sign:{-1,1}){double nearest=choice.maxDistance;bool found=false;V target;
                for(size_t j=1;j<lower->p.size();++j){double t,u;
                    if(hit(xy(a.top),normal*sign,xy(lower->p[j-1]),xy(lower->p[j]),t,u)&&t>cell&&t<=nearest){nearest=t;target=lower->p[j-1]+std::clamp(u,0.0,1.0)*(lower->p[j]-lower->p[j-1]);found=true;}
                }if(found){++hits;chosen=target;chosenDist=nearest;}
            }
            if(hits!=1){a.status=hits?"ambiguous_target_both_sides":"no_selected_lower_intersection";finish();continue;}
            a.hasTarget=true;a.bottom=chosen;a.width=chosenDist;
            auto top=lookup(a.top),bottom=lookup(a.bottom);
            if(!valid(top)||!valid(bottom)){a.status="unobserved_endpoint";finish();continue;}
            a.top.z()=top->avg_z;a.bottom.z()=bottom->avg_z;
            if(a.top.z()-a.bottom.z()<1){a.status="selected_lower_not_below_upper";finish();continue;}
            int observed=0;double gap=0,maxGap=0,runStart=0;bool onSlope=false,reverse=false;
            XY direction=(xy(a.bottom)-xy(a.top)).normalized();double minSoFar=a.top.z();
            int steps=int(std::ceil(a.width/cell));
            for(int k=0;k<=steps;++k){double d=std::min(a.width,k*cell);V p=a.top;p.head<2>()+=direction*d;auto g=lookup(p);bool seen=valid(g);
                a.profile.push_back({d,seen?double(g->avg_z):0,g?g->type:0,g?g->count:0,seen});
                if(seen){++observed;gap=0;}else{gap+=cell;maxGap=std::max(maxGap,gap);}
                bool isSlope=seen&&g->type==2;
                if(isSlope&&!onSlope)runStart=d;
                if(!isSlope&&onSlope&&d-runStart>=2*cell)++a.slopeRuns;
                onSlope=isSlope;
                // Ignore berm elevations; reject marked ascent after a lower valley.
                if(seen&&g->type!=4){if(g->avg_z>minSoFar+3)reverse=true;minSoFar=std::min(minSoFar,double(g->avg_z));}
            }
            if(onSlope&&a.width-runStart>=2*cell)++a.slopeRuns;
            a.coverage=double(observed)/a.profile.size();
            if(a.coverage<.95||maxGap>3*cell)a.status="incomplete_observed_profile";
            else if(reverse)a.status="possible_opposite_wall_or_uphill_crossing";
            else if(a.slopeRuns<2)a.status="insufficient_multiple_bench_evidence";
            else a.status="boundary_confirmation_required";
            finish();
        }
    }
    // Flag geometric conflicts; previews are not a validated boundary set.
    for(size_t i=0;i<out.size();++i)for(size_t j=i+1;j<out.size();++j){
        if(out[i].name!=out[j].name||!out[i].hasTarget||!out[j].hasTarget)continue;
        Sample a,b;a.crest=out[i].top;a.toe=out[i].bottom;b.crest=out[j].top;b.toe=out[j].bottom;
        if(crossing(a,b)){
            if(out[i].status=="boundary_confirmation_required")out[i].status="crossing_preview_sections";
            if(out[j].status=="boundary_confirmation_required")out[j].status="crossing_preview_sections";
        }
    }
    // Only non-conflicting, observed, multi-bench profiles receive review geometry.
    // This is an endpoint inclination, not a confirmed working-slope angle.
    for(auto& a:out)if(a.status=="boundary_confirmation_required"){
        a.height=a.top.z()-a.bottom.z();a.angle=std::atan2(a.height,a.width)*180/std::acos(-1.0);a.geometryAvailable=true;
    }
    return out;
}
inline void save(const std::vector<Preview>& data,const std::string& prefix){
    std::ofstream csv(prefix+"overall_preview.csv"),profile(prefix+"overall_profiles.csv"),ply(prefix+"overall_preview.ply");
    if(!csv||!profile||!ply)throw std::runtime_error("Cannot write overall preview files");
    csv<<std::fixed<<std::setprecision(8);profile<<std::fixed<<std::setprecision(8);ply<<std::fixed<<std::setprecision(8);
    csv<<"section_id,workface_id,upper_line_id,lower_line_id,station,top_x,top_y,top_z,bottom_x,bottom_y,bottom_z,horizontal_span,observed_coverage,slope_runs,status,upper_boundary_type,lower_boundary_type,height_difference_m,endpoint_inclination_deg,geometry_available,measurement_scope\n";
    profile<<"section_id,distance,z,type,count,observed\n";
    size_t n=0;for(auto& a:data)if(a.hasTarget)++n;
    ply<<"ply\nformat ascii 1.0\ncomment PREVIEW_ONLY_NOT_VALIDATED\nelement vertex "<<2*n<<"\nproperty double x\nproperty double y\nproperty double z\nelement edge "<<n<<"\nproperty int vertex1\nproperty int vertex2\nend_header\n";
    for(auto& a:data){csv<<a.id<<','<<a.name<<','<<a.upper<<','<<a.lower<<','<<a.station<<','<<a.top.x()<<','<<a.top.y()<<','<<a.top.z()<<',';
        if(a.hasTarget)csv<<a.bottom.x()<<','<<a.bottom.y()<<','<<a.bottom.z()<<','<<a.width;else csv<<",,,";
        csv<<','<<a.coverage<<','<<a.slopeRuns<<','<<a.status<<','<<(a.upperType==0?"crest":"toe")<<','<<(a.lowerType==0?"crest":"toe")<<',';
        if(a.geometryAvailable)csv<<a.height;csv<<',';if(a.geometryAvailable)csv<<a.angle;
        csv<<','<<a.geometryAvailable<<",selected_boundary_geometry_unconfirmed\n";
        for(auto& p:a.profile){profile<<a.id<<','<<p.distance<<',';if(p.observed)profile<<p.z;profile<<','<<p.type<<','<<p.count<<','<<p.observed<<'\n';}
        if(a.hasTarget)for(auto& p:{a.top,a.bottom})ply<<p.x()<<' '<<p.y()<<' '<<p.z()<<'\n';
    }for(size_t i=0;i<n;++i)ply<<2*i<<' '<<2*i+1<<'\n';
}
}
