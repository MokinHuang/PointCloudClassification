#pragma once
#include "AutoWorkfaces.h"
namespace paper_rules {
// Paper reproduction only, not a declaration of compliance with current regulations.
inline void save(const std::vector<sections::Sample>& slopes,const std::vector<platforms::Measurement>& widths,
 const std::vector<overall::Preview>& candidates,const std::string& prefix){
    std::ofstream rules(prefix+"paper_risk_rules.csv"),limits(prefix+"design_limits_template.csv"),risk(prefix+"parameter_risk_pending.csv"),state(prefix+"safety_evaluation_status.csv");
    if(!rules||!limits||!risk||!state)throw std::runtime_error("Cannot write paper evaluation metadata");
    rules<<"parameter,level,comparison,design_multiplier,offset_deg,score,source,notes\n"
      <<"bench_height,major,>=,2,0,4,Table2_p1157,original_wording_greater_than_2_times_and_above\n"
      <<"bench_height,higher,>,1.6,0,2,Table2_p1157,apply_highest_matching_level\n"
      <<"bench_height,general,>,1.3,0,1,Table2_p1157,apply_highest_matching_level\n"
      <<"bench_height,low,>,1.1,0,0.5,Table2_p1157,apply_highest_matching_level\n"
      <<"platform_width,major,<,0.5,0,4,Table2_p1157,exclude_temporary_and_final_limit_platforms\n"
      <<"platform_width,higher,<,0.625,0,2,Table2_p1157,exclude_temporary_and_final_limit_platforms\n"
      <<"platform_width,general,<,0.75,0,1,Table2_p1157,exclude_temporary_and_final_limit_platforms\n"
      <<"platform_width,low,range_0.75_to_1,,,0.5,Table2_p1157,endpoint_inclusion_not_explicit\n"
      <<"bench_angle,major,>,1,15,4,Table2_p1157,design_max_angle_plus_offset\n"
      <<"bench_angle,higher,>,1,12,2,Table2_p1157,design_max_angle_plus_offset\n"
      <<"bench_angle,general,>,1,9,1,Table2_p1157,design_max_angle_plus_offset\n"
      <<"bench_angle,low,>,1,5,0.5,Table2_p1157,design_max_angle_plus_offset\n"
      <<"working_slope_angle,major,>,1,0,4,Table2_p1157,requires_confirmed_working_slope\n";
    limits<<"zone_id,parameter,design_value,unit,source\n"
      <<"UNASSIGNED,bench_height,,m,required_mine_design\nUNASSIGNED,platform_width,,m,required_mine_design\n"
      <<"UNASSIGNED,bench_angle,,deg,required_mine_design\nUNASSIGNED,working_slope_angle,,deg,required_mine_design\n";
    risk<<"source,measurement_id,parameter,measured_value,unit,design_value,risk_level,status\n"<<std::fixed<<std::setprecision(6);
    for(auto& s:slopes)if(s.status=="matched"){
        risk<<"single_bench,"<<s.id<<",bench_height,"<<s.height<<",m,,,missing_design_and_zone\n";
        risk<<"single_bench,"<<s.id<<",bench_angle,"<<s.angle<<",deg,,,missing_design_and_zone\n";
    }
    for(auto& w:widths)if(w.geometry.status=="matched")risk<<"platform,"<<w.geometry.id<<",platform_width,"<<w.geometry.width<<",m,,,missing_design_zone_and_platform_usage\n";
    for(auto& c:candidates)if(c.geometryAvailable)risk<<c.name<<','<<c.id<<",candidate_overall_angle,"<<c.angle<<",deg,,,unconfirmed_workface_and_missing_design\n";
    state<<"item,value,status\nSMSI,,not_computed\ndesign_limits,,missing\nproduction_and_platform_usage,,unknown\n"
      <<"platform_weight_text,0.15,conflicts_with_table3\nplatform_weight_table3,0.05,conflicts_with_text\n"
      <<"bench_angle_weight_text,0.05,conflicts_with_table3\nbench_angle_weight_table3,0.15,conflicts_with_text\n"
      <<"height_weight,0.20,paper\nworking_slope_weight,0.60,paper\n"
      <<"risk_count_unit,,must_define_before_aggregation\n";
}
}
