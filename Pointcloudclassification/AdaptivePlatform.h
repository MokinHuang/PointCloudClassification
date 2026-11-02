#pragma once
#include "PlatformMeasurement.h"
#include <queue>
#include <set>

namespace platforms {
struct AdaptiveResult {
    std::vector<Measurement> original, review;
    int rows=0,cols=0;
    std::vector<int> platformIds;
    struct ProfilePoint { int sample;double distance,z,localFit; };
    std::vector<ProfilePoint> profiles;
};
inline bool neighbours(const Sample& a,const Sample& b){
    double separation=(xy(a.crest)-xy(b.crest)).norm();
    if(a.id==b.id||separation<.5||separation>16)return false;
    XY va=xy(a.toe)-xy(a.crest),vb=xy(b.toe)-xy(b.crest);
    return va.norm()>0&&vb.norm()>0&&va.normalized().dot(vb.normalized())>=.94&&
        std::abs(a.width-b.width)<=5+.4*separation&&(xy(a.toe)-xy(b.toe)).norm()<=25;
}
inline bool conflict(const Sample& a,const Sample& b){
    return (xy(a.crest)-xy(b.crest)).norm()<.5||(xy(a.toe)-xy(b.toe)).norm()<.5||crossing(a,b)||
        (a.crestLine==b.crestLine&&a.toeLine==b.toeLine&&
         (a.orderSign!=b.orderSign||(a.crestS-b.crestS)*(a.toeS-b.toeS)*a.orderSign<=0));
}
inline void groupCandidates(AdaptiveResult& result){
    // Resolve geometry conflicts FIRST; rejected candidates cannot support a group.
    std::vector<Sample> occupied;for(const auto& m:result.original)if(m.geometry.status=="matched")occupied.push_back(m.geometry);
    std::vector<size_t> pending;
    for(size_t i=0;i<result.review.size();++i)if(result.review[i].geometry.status=="review_pending")pending.push_back(i);
    std::sort(pending.begin(),pending.end(),[&](size_t i,size_t j){const auto& a=result.review[i].geometry;const auto& b=result.review[j].geometry;
        return a.score!=b.score?a.score>b.score:a.id<b.id;});
    std::vector<size_t> kept;
    for(auto i:pending){auto& a=result.review[i].geometry;bool bad=false;
        for(const auto& b:occupied)if(conflict(a,b)){bad=true;break;}
        if(bad)a.status="conflict_with_existing_or_candidate";
        else{occupied.push_back(a);kept.push_back(i);}
    }
    std::vector<bool> seen(kept.size(),false);
    for(size_t seed=0;seed<kept.size();++seed)if(!seen[seed]){
        std::queue<size_t> q;q.push(seed);seen[seed]=true;std::vector<size_t> component;std::set<int> anchors;
        while(!q.empty()){size_t i=q.front();q.pop();component.push_back(i);const auto& a=result.review[kept[i]].geometry;
            for(const auto& m:result.original)if(m.geometry.status=="matched"&&neighbours(a,m.geometry))anchors.insert(m.geometry.id);
            for(size_t j=0;j<kept.size();++j)if(!seen[j]&&neighbours(a,result.review[kept[j]].geometry)){seen[j]=true;q.push(j);}
        }
        // Connected chains keep their endpoints. All surviving nodes are displayed.
        bool supported=component.size()+anchors.size()>=3;
        for(auto i:component)result.review[kept[i]].geometry.status=supported?"review_candidate":"low_confidence_candidate";
    }
}
// Sliding local regression: allow gradual curvature, retain abrupt-step rejection.
inline std::string checkLocalProfile(const std::vector<std::pair<double,double>>& p,
    std::vector<double>& fitted){
    fitted.clear();if(p.size()<5)return "insufficient_platform_profile";
    const double grade=std::tan(8*3.141592653589793/180);
    bool steep=false,rough=false,step=false;size_t unsupported=0;double steepStart=-1;
    for(size_t k=0;k<p.size();++k){
        double sx=0,sy=0,sxx=0,sxy=0,n=0;
        for(auto q:p)if(std::abs(q.first-p[k].first)<=5.0){
            double x=q.first-p[k].first;sx+=x;sy+=q.second;sxx+=x*x;sxy+=x*q.second;++n;
        }
        // Excluding a berm may leave a few isolated platform samples. Keep their
        // heights for step checks, but do not fit a line from fewer than 3 samples.
        if(k&&std::abs(p[k].second-p[k-1].second)>1+grade*(p[k].first-p[k-1].first))step=true;
        if(n<3){fitted.push_back(p[k].second);++unsupported;steepStart=-1;continue;}
        double den=n*sxx-sx*sx;
        if(den<1e-9){fitted.push_back(p[k].second);return "insufficient_local_profile";}
        double slope=(n*sxy-sx*sy)/den,z=(sy-slope*sx)/n;fitted.push_back(z);
        if(std::abs(slope)>grade){if(steepStart<0)steepStart=p[k].first;
            if(p[k].first-steepStart>=5)steep=true;
        }else steepStart=-1;
        rough=rough||std::abs(p[k].second-z)>1.5;
        if(k&&std::abs(p[k].second-p[k-1].second)>1+grade*(p[k].first-p[k-1].first))step=true;
    }
    if(step)return "profile_abrupt_step";
    if(unsupported>p.size()*.1)return "insufficient_local_profile";
    if(steep)return "profile_local_steep_segment";
    if(rough)return "profile_local_roughness";
    return "review_pending";
}
// Additional long sections remain REVIEW ONLY. Never promote into original measurements.
inline AdaptiveResult adaptive(const std::vector<Sample>& slopes,const std::vector<FeatureLine>& lines,
    const std::vector<std::vector<GridCell>>& grid,double cell,const Eigen::Vector4f& origin){
    AdaptiveResult result;result.original=measure(slopes,lines,grid,cell,origin);
    Options expanded;expanded.maxDistance=200;auto extended=measure(slopes,lines,grid,cell,origin,expanded);
    if(grid.empty()||grid[0].empty())return result;
    const int rows=int(grid.size()),cols=int(grid[0].size());
    std::vector<int> labels(rows*cols,0);int label=0;
    auto valid=[&](int r,int c){const auto& g=grid[r][c];return !g.is_empty&&g.count>0&&std::isfinite(g.avg_z)&&(g.type==1||g.type==4);};
    // Observed platform/berm components with a local elevation-step constraint.
    // This is auxiliary evidence, not proof of geological platform identity.
    for(int r=0;r<rows;++r)for(int c=0;c<cols;++c)if(!labels[r*cols+c]&&valid(r,c)){
        ++label;std::queue<int> q;q.push(r*cols+c);labels[r*cols+c]=label;
        while(!q.empty()){int k=q.front();q.pop();int rr=k/cols,cc=k%cols;
            for(auto delta:{std::pair<int,int>{-1,0},{1,0},{0,-1},{0,1}}){int nr=rr+delta.first,nc=cc+delta.second;
                if(nr<0||nc<0||nr>=rows||nc>=cols||labels[nr*cols+nc]||!valid(nr,nc))continue;
                if(std::abs(grid[nr][nc].avg_z-grid[rr][cc].avg_z)>.75+cell*.14)continue;
                labels[nr*cols+nc]=label;q.push(nr*cols+nc);
            }
        }
    }
    result.rows=rows;result.cols=cols;result.platformIds=labels;
    for(auto m:extended){auto& a=m.geometry;if(a.status!="matched"||a.width<100-1e-6)continue;
        a.status="review_pending";std::vector<std::pair<double,double>> profile;int region=0;bool regionOK=true;
        XY direction=(xy(a.toe)-xy(a.crest)).normalized();
        for(double d=2*cell;d<a.width-2*cell;d+=cell){XY p=xy(a.crest)+direction*d;
            int c=int(std::floor((p.x()-origin.x())/cell)),r=int(std::floor((p.y()-origin.y())/cell));
            if(r<0||c<0||r>=rows||c>=cols){regionOK=false;break;}
            int id=labels[r*cols+c];
            if(!id){regionOK=false;break;}if(!region)region=id;else if(region!=id){regionOK=false;break;}
            // Berm elevations are not the platform surface; retain berm in width, exclude from fit.
            if(grid[r][c].type==1)profile.emplace_back(d,grid[r][c].avg_z);
        }
        if(!regionOK){a.status="different_region_or_profile_gap";result.review.push_back(m);continue;}
        if(profile.size()<5){a.status="insufficient_platform_profile";result.review.push_back(m);continue;}
        std::vector<double> fitted;a.status=checkLocalProfile(profile,fitted);
        for(size_t k=0;k<fitted.size();++k)result.profiles.push_back({a.id,profile[k].first,profile[k].second,fitted[k]});
        if(a.status=="profile_abrupt_step"||a.status=="profile_local_steep_segment"||a.status=="profile_local_roughness"){
            // Diagnostic only: failures confined to the boundary band stay rejected.
            double band=std::min(20.0,std::max(5.0,a.width*.2));
            std::vector<std::pair<double,double>> interior;
            for(auto p:profile)if(p.first>=band&&p.first<=a.width-band)interior.push_back(p);
            std::vector<double> fit;if(interior.size()>=5&&checkLocalProfile(interior,fit)=="review_pending")a.status="boundary_"+a.status;
        }
        result.review.push_back(m);
    }
    groupCandidates(result);
    return result;
}
inline void saveAdaptive(const AdaptiveResult& result,const std::string& prefix=""){
    save(result.original,prefix);
    std::ofstream regions(prefix+"platform_ids.csv"),profiles(prefix+"platform_profiles.csv");
    if(!regions||!profiles)throw std::runtime_error("Cannot write platform diagnostics");
    regions<<"r,c,platform_id\n";
    for(int r=0;r<result.rows;++r)for(int c=0;c<result.cols;++c)
        regions<<r<<','<<c<<','<<result.platformIds[r*result.cols+c]<<'\n';
    profiles<<std::fixed<<std::setprecision(8)<<"sample_id,distance,z,local_fit\n";
    for(auto p:result.profiles)profiles<<p.sample<<','<<p.distance<<','<<p.z<<','<<p.localFit<<'\n';
    std::ofstream audit(prefix+"extension_audit.csv");if(!audit)throw std::runtime_error("Cannot write extension audit");
    audit<<std::fixed<<std::setprecision(8)<<"sample_id,crest_x,crest_y,toe_x,toe_y,width,status\n";
    std::vector<Measurement> candidates,lowConfidence;
    for(auto m:result.review){const auto& a=m.geometry;
        audit<<a.id<<','<<a.crest.x()<<','<<a.crest.y()<<','<<a.toe.x()<<','<<a.toe.y()<<','<<a.width<<','<<a.status<<'\n';
        if(a.status=="review_candidate"){m.geometry.status="matched";candidates.push_back(m);}
        else if(a.status=="low_confidence_candidate"){m.geometry.status="matched";lowConfidence.push_back(m);}
    }
    save(candidates,prefix+"review_"); // Separate filenames, never merged into platform_widths.csv.
    save(lowConfidence,prefix+"low_confidence_");
}
}
