#include "Pointcloudclassification/SectionMeasurement.h"
#include "Pointcloudclassification/PlatformMeasurement.h"
#include "Pointcloudclassification/AdaptivePlatform.h"
#include "Pointcloudclassification/FeatureGapRepair.h"
#include <filesystem>
#include <sstream>
#include <iostream>
#include <map>
// Replay the saved measured grid/lines without regenerating or changing classification.
int main(int argc,char** argv){try{
    if(argc!=3&&argc!=4)throw std::runtime_error("Usage: gap_replay INPUT_DIRECTORY OUTPUT_DIRECTORY");
    namespace fs=std::filesystem;fs::path root=argv[1],out=argv[2];fs::create_directories(out);
    Eigen::Vector4f origin=Eigen::Vector4f::Zero();double cell;
    std::ifstream o(root/"grid_origin.txt");if(!(o>>origin.x()>>origin.y()>>cell))throw std::runtime_error("Invalid grid origin");
    std::ifstream f(root/"classification.csv");if(!f)throw std::runtime_error("Missing grid");
    std::string text;std::getline(f,text);std::vector<std::vector<GridCell>> grid;
    while(std::getline(f,text)){if(text.empty())continue;std::replace(text.begin(),text.end(),',',' ');std::istringstream s(text);
        int r,c,crest,toe;GridCell g;if(!(s>>r>>c>>g.avg_z>>g.type>>g.count>>crest>>toe))throw std::runtime_error("Invalid grid row");
        g.is_empty=false;g.is_crest=crest!=0;g.is_toe=toe!=0;
        if(r<0||c<0)throw std::runtime_error("Negative grid index");
        if(grid.size()<=size_t(r))grid.resize(r+1);if(grid[r].size()<=size_t(c))grid[r].resize(c+1);grid[r][c]=g;
    }
    std::ifstream fl(root/"feature_lines.csv");if(!fl)throw std::runtime_error("Missing feature lines");
    std::getline(fl,text);std::map<int,FeatureLine> byId;
    while(std::getline(fl,text)){if(text.empty()||text=="\r")continue;std::replace(text.begin(),text.end(),',',' ');std::istringstream s(text);
        int id,type,order;std::string name;Eigen::Vector3f p;if(!(s>>id>>type>>name>>order>>p.x()>>p.y()>>p.z()))throw std::runtime_error("Invalid feature row");
        auto& l=byId[id];l.id=id;l.type=type;l.points.push_back(p);
    }
    std::vector<FeatureLine> lines;for(auto& entry:byId)lines.push_back(entry.second);
    gaprepair::Options options;
    if(argc==4){std::string mode=argv[3];if(mode!="--local-fit"&&mode!="--wide-270")throw std::runtime_error("Unknown repair mode");options.localFit=true;if(mode=="--wide-270")options.maxAngleDegrees=135.0;}
    auto result=gaprepair::repair(lines,grid,cell,origin,options);gaprepair::save(result,lines,origin,out.string()+"/");
    std::map<std::string,int> counts;for(auto& c:result.candidates)++counts[c.status];
    for(auto& e:counts)std::cout<<e.first<<": "<<e.second<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
