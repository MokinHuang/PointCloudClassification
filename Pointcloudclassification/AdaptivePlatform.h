#pragma once
#include "PlatformMeasurement.h"
#include <queue>

namespace platforms {
struct AdaptiveResult {
    std::vector<Measurement> original, review;
};
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
        double sx=0,sy=0,sxx=0,sxy=0;for(auto v:profile){sx+=v.first;sy+=v.second;sxx+=v.first*v.first;sxy+=v.first*v.second;}
        double n=double(profile.size()),den=n*sxx-sx*sx;
        double slope=den>1e-9?(n*sxy-sx*sy)/den:0,intercept=(sy-slope*sx)/n;
        double residual=0;bool step=false;
        for(size_t k=0;k<profile.size();++k){auto p=profile[k];residual=std::max(residual,std::abs(p.second-intercept-slope*p.first));
            if(k){auto last=profile[k-1];if(std::abs(p.second-last.second)>1+std::tan(8*3.141592653589793/180)*(p.first-last.first))step=true;}
        }
        if(std::abs(slope)>std::tan(8*3.141592653589793/180)||residual>1.5||step)a.status="profile_step_or_excessive_relief";
        result.review.push_back(m);
    }
    // Require two nearby, consistent sections, allowing fragmented crest/toe line IDs.
    // A neighbour is evidence only; even passing candidates stay outside formal CSV.
    const auto profileSupported=result.review; // Snapshot avoids traversal-order-dependent neighbour evidence.
    for(auto& m:result.review){auto& a=m.geometry;if(a.status!="review_pending")continue;int neighbours=0;
        auto support=[&](const Measurement& other){const auto& b=other.geometry;
            if(a.id==b.id)return;
            double separation=(xy(a.crest)-xy(b.crest)).norm();if(separation<.5||separation>16)return;
            XY va=xy(a.toe)-xy(a.crest),vb=xy(b.toe)-xy(b.crest);
            if(va.normalized().dot(vb.normalized())<.94||std::abs(a.width-b.width)>5+.4*separation)return;
            if((xy(a.toe)-xy(b.toe)).norm()>25)return;
            ++neighbours;
        };
        for(const auto& b:result.original)if(b.geometry.status=="matched")support(b);
        for(const auto& b:profileSupported)if(b.geometry.status=="review_pending")support(b);
        a.status=neighbours>=2?"review_supported":"isolated_or_discontinuous_extension";
    }
    std::vector<Sample> occupied;for(auto& m:result.original)if(m.geometry.status=="matched")occupied.push_back(m.geometry);
    for(auto& m:result.review){auto& a=m.geometry;if(a.status!="review_supported")continue;bool conflict=false;
        for(auto& b:occupied)if((xy(a.crest)-xy(b.crest)).norm()<.5||(xy(a.toe)-xy(b.toe)).norm()<.5||crossing(a,b)||
            (a.crestLine==b.crestLine&&a.toeLine==b.toeLine&&
             (a.orderSign!=b.orderSign||(a.crestS-b.crestS)*(a.toeS-b.toeS)*a.orderSign<=0))){conflict=true;break;}
        a.status=conflict?"conflict_with_existing_or_candidate":"review_candidate";
        if(!conflict)occupied.push_back(a);
    }
    return result;
}
inline void saveAdaptive(const AdaptiveResult& result,const std::string& prefix=""){
    save(result.original,prefix);
    std::ofstream audit(prefix+"extension_audit.csv");if(!audit)throw std::runtime_error("Cannot write extension audit");
    audit<<std::fixed<<std::setprecision(8)<<"sample_id,crest_x,crest_y,toe_x,toe_y,width,status\n";
    std::vector<Measurement> candidates;
    for(auto m:result.review){const auto& a=m.geometry;
        audit<<a.id<<','<<a.crest.x()<<','<<a.crest.y()<<','<<a.toe.x()<<','<<a.toe.y()<<','<<a.width<<','<<a.status<<'\n';
        if(a.status=="review_candidate"){m.geometry.status="matched";candidates.push_back(m);}
    }
    save(candidates,prefix+"review_"); // Separate filenames, never merged into platform_widths.csv.
}
}
