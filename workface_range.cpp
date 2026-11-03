#include "Pointcloudclassification/WorkfaceRangeReview.h"
#include "Pointcloudclassification/GroupedWorkfaceReview.h"
#include "Pointcloudclassification/ResolvedWorkfaceReview.h"
#include "Pointcloudclassification/ContinuousWorkfaces.h"
#include <filesystem>
#include <sstream>
#include <iostream>
#include <map>
// Replay the saved measured grid/lines without regenerating or changing classification.
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("Usage: workface_range INPUT_DIRECTORY OUTPUT_DIRECTORY");
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
    const auto slopes=sections::measure(lines,grid,cell,origin);
    const auto widths=platforms::measure(slopes,lines,grid,cell,origin);
    const auto graph=autowalls::discover(slopes,widths);
    const auto previews=autowalls::measure(graph,lines,grid,cell,origin);
    range_review::write(lines,grid,cell,origin,graph,previews,out.string()+"/");
    const auto groupedGraph=grouped::discover(slopes,widths,lines,grid,cell,origin);
    const auto groupedRows=grouped::measure(groupedGraph,lines,grid,cell,origin);
    grouped::save(groupedGraph,groupedRows,out.string()+"/");
    grouped_review::write(lines,grid,cell,origin,groupedGraph,groupedRows,out.string()+"/");
    const auto resolution=resolved::run(groupedGraph,groupedRows,widths,lines,grid,cell,origin);
    resolved::save(resolution,groupedRows,out.string()+"/");
    continuous::process(groupedGraph,groupedRows,widths,lines,grid,cell,origin,out.string()+"/");
    resolved_review::write(lines,grid,cell,origin,groupedGraph,groupedRows,resolution,out.string()+"/");
    std::cout<<"Grouped benches "<<groupedGraph.groups.size()<<", candidates "<<groupedGraph.candidates.size()<<", sections "<<groupedRows.size()<<"\n";
    std::cout<<"Saved workface_range_review.html. Range review only; original measurements unchanged.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
