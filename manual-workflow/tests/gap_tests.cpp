#include "../Pointcloudclassification/FeatureGapRepair.h"
#include <iostream>
#include <functional>
using namespace gaprepair;
using Map=std::vector<std::vector<GridCell>>;
FeatureLine line(int id,float x,float y,float end,int type=0,float z=10){FeatureLine l;l.id=id;l.type=type;
    for(float a=x;a<=end;a+=1)l.points.push_back(Eigen::Vector3f(a,y,z));return l;}
Map grid(){Map g(30,std::vector<GridCell>(50));for(auto& row:g)for(auto& c:row){c.count=1;c.is_empty=false;c.avg_z=10;c.is_crest=true;c.is_toe=true;}return g;}
int accepted(const Result&r){int n=0;for(auto& c:r.candidates)n+=c.status=="accepted";return n;}
void check(bool ok){if(!ok)throw std::runtime_error("assertion failed");}
int main(){int n=0;auto test=[&](const char* name,std::function<void()> f){try{f();++n;std::cout<<"PASS "<<name<<'\n';}catch(const std::exception&e){std::cerr<<"FAIL "<<name<<": "<<e.what();std::exit(1);}};
    auto o=Eigen::Vector4f::Zero().eval();
    test("aligned short gap",[&]{check(accepted(repair({line(0,2,10,10),line(1,13,10,21)},grid(),1,o))==1);});
    test("wrong type",[&]{check(!accepted(repair({line(0,2,10,10),line(1,13,10,21,1)},grid(),1,o)));});
    test("wrong elevation",[&]{check(!accepted(repair({line(0,2,10,10),line(1,13,10,21,0,12)},grid(),1,o)));});
    test("parallel lateral lines",[&]{check(!accepted(repair({line(0,2,10,10),line(1,2,12,10)},grid(),1,o)));});
    test("missing observations",[&]{auto g=grid();g[10][11].count=0;check(!accepted(repair({line(0,2,10,10),line(1,13,10,21)},g,1,o)));});
    test("missing boundary evidence",[&]{auto g=grid();for(auto&row:g)for(auto&c:row)c.is_crest=false;check(!accepted(repair({line(0,2,10,10),line(1,13,10,21)},g,1,o)));});
    test("two plausible neighbors",[&]{check(!accepted(repair({line(0,2,10,10),line(1,13,9,21),line(2,13,11,21)},grid(),1,o)));});
    test("crossing another type",[&]{auto l=line(2,7,12,15,1);for(auto&p:l.points){float a=p.x();p.x()=12;p.y()=a;}
        check(!accepted(repair({line(0,2,10,10),line(1,13,10,21),l},grid(),1,o)));});
    test("existing junction",[&]{auto l=line(2,4,10,12,1);for(auto&p:l.points){float a=p.x();p.x()=10;p.y()=a;}
        check(!accepted(repair({line(0,2,10,10),line(1,13,10,21),l},grid(),1,o)));});
    test("insufficient tangent",[&]{check(!accepted(repair({line(0,8,10,10),line(1,13,10,21)},grid(),1,o)));});
    test("long gap",[&]{check(!accepted(repair({line(0,2,10,10),line(1,16,10,24)},grid(),1,o)));});
    test("reversed source order",[&]{auto l=line(1,13,10,21);std::reverse(l.points.begin(),l.points.end());check(accepted(repair({l,line(0,2,10,10)},grid(),1,o))==1);});
    test("large UTM coordinates",[&]{auto a=line(0,2,10,10),b=line(1,13,10,21);Eigen::Vector4f origin(508349.84375f,4945767,0,0);
        for(auto* l:{&a,&b})for(auto&p:l->points){p.x()+=origin.x();p.y()+=origin.y();}check(accepted(repair({a,b},grid(),1,origin))==1);});
    test("chain and provenance export",[&]{std::vector<FeatureLine> ls{line(0,2,10,10),line(1,13,10,21),line(2,24,10,32)};auto r=repair(ls,grid(),1,o);check(accepted(r)==2);save(r,ls,o,"build/test_");});
    test("invalid parameters",[&]{bool threw=false;try{repair({},grid(),0,o);}catch(const std::invalid_argument&){threw=true;}check(threw);});
    Options adaptive;adaptive.localFit=true;
    test("local fit recovers straight short segment",[&]{check(accepted(repair({line(0,8,10,10),line(1,13,10,21)},grid(),1,o,adaptive))==1);});
    test("local fit still rejects missing observations",[&]{auto g=grid();g[10][11].count=0;check(!accepted(repair({line(0,8,10,10),line(1,13,10,21)},g,1,o,adaptive)));});
    test("local fit still rejects competing neighbors",[&]{check(!accepted(repair({line(0,8,10,10),line(1,13,9,21),line(2,13,11,21)},grid(),1,o,adaptive)));});
    test("local fit still rejects elevation",[&]{check(!accepted(repair({line(0,8,10,10),line(1,13,10,21,0,12)},grid(),1,o,adaptive)));});
    Options wide;wide.localFit=true;wide.maxAngleDegrees=135;
    test("270 degree cone accepts lateral candidate for later checks",[&]{auto rr=repair({line(0,2,10,10),line(1,2,12,10)},grid(),1,o,wide);bool found=false;for(auto& c:rr.candidates)if(c.angle>89&&c.angle<91){found=true;check(c.status!="direction");}check(found);});
    test("270 cone still rejects missing observations",[&]{auto g=grid();g[10][11].count=0;check(!accepted(repair({line(0,2,10,10),line(1,13,10,21)},g,1,o,wide)));});
    test("270 cone keeps uniqueness guard",[&]{auto rr=repair({line(0,2,10,10),line(1,13,9,21),line(2,13,11,21)},grid(),1,o,wide);bool rejected=false;for(auto& c:rr.candidates)if(rr.ends[c.a].line==0||rr.ends[c.b].line==0){check(c.status!="accepted");rejected|=c.status=="ambiguous_endpoint";}check(rejected);});
    test("invalid direction threshold above 180",[&]{bool threw=false;auto bad=wide;bad.maxAngleDegrees=181;try{repair({},grid(),1,o,bad);}catch(const std::invalid_argument&){threw=true;}check(threw);});
    std::cout<<n<<" gap tests passed\n";
}
