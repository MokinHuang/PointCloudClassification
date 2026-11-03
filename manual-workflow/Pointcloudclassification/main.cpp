#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <windows.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include "PointCloudClassification.h"
#include "FeatureLine.h"
#include "SectionMeasurement.h"
#include "PlatformMeasurement.h"
#include "AdaptivePlatform.h"

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);
    // --- 1. 初始化参数 ---
    std::string filename = argc > 1 ? argv[1] : "test2.pcd";
    float gridSize = 1.0f;  // 网格分辨率 1.0m
    PointCloudClassification processor(gridSize);

    std::cout << "正在加载点云文件: " << filename << "..." << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        std::cerr << "无法读取文件，请检查路径" << std::endl;
        return -1;
    }

    // --- 2. 构建网格地图 ---
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);

    int cols = static_cast<int>((max_pt[0] - min_pt[0]) / gridSize) + 1;
    int rows = static_cast<int>((max_pt[1] - min_pt[1]) / gridSize) + 1;

    std::vector<std::vector<GridCell>> gridMap(rows, std::vector<GridCell>(cols));

    for (const auto& pt : cloud->points) {
        int c = static_cast<int>((pt.x - min_pt[0]) / gridSize);
        int r = static_cast<int>((pt.y - min_pt[1]) / gridSize);
        if (r >= 0 && r < rows && c >= 0 && c < cols) {
            gridMap[r][c].sum_z += pt.z;
            gridMap[r][c].count++;
        }
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (gridMap[r][c].count > 0) {
                gridMap[r][c].avg_z = gridMap[r][c].sum_z / gridMap[r][c].count;
                gridMap[r][c].is_empty = false;
            }
        }
    }

    // --- 3. 分类处理流水线 ---
    std::cout << "正在执行分类处理流水线..." << std::endl;

    processor.fillEmptyGrids(gridMap, rows, cols);
    processor.classifyGrids(gridMap, gridSize);

    for (int i = 0; i < 2; ++i) {
        processor.refineClassification(gridMap);
    }

    for (int i = 0; i < 3; ++i) {
        processor.cleanPlatformNoise(gridMap);
    }
    processor.closeSlopeHoles(gridMap);
    processor.filterFalseSlopes(gridMap);

    processor.identifyBerms(gridMap);
    processor.fillBermGaps(gridMap);
    processor.removeIsolatedBerms(gridMap);

    // --- 4. 提取坡顶/坡底线 ---
    std::cout << "正在提取坡顶(Crest)和坡底(Toe)特征线..." << std::endl;
    std::vector<SlopeParams> measuredParams;
    processor.extractSlopeFeatures(gridMap, measuredParams);
    processor.thinFeatureLines(gridMap);

    std::vector<FeatureLine> featureLines =
        extractFeatureLines(gridMap, gridSize, min_pt, 2, 2, 1.2f);

    saveFeatureLinesToTxt(featureLines, "feature_lines_raw.csv");
    smoothFeatureLines(featureLines, 5);

    std::vector<FeatureLine> finalLines;
    for (const auto& line : featureLines) {
        if (line.points.size() >= 4 || (line.is_branch && line.points.size() >= 2))
            finalLines.push_back(line);
    }

    saveFeatureLinesToTxt(finalLines, "feature_lines.csv");

    const auto sections = sections::measure(finalLines, gridMap, gridSize, min_pt);
    sections::save(sections);
    const auto adaptiveWidths = platforms::adaptive(sections, finalLines, gridMap, gridSize, min_pt);
    platforms::saveAdaptive(adaptiveWidths);
    const auto& platformWidths = adaptiveWidths.original;
    size_t platformCount = 0;
    for (const auto& p : platformWidths) if (p.geometry.status == "matched") ++platformCount;
    std::cout << "[Platforms] " << platformCount << " widths measured.\n";
    size_t matchedCount = 0;
    for (const auto& section : sections) if (section.status == "matched") ++matchedCount;
    std::cout << "[Sections] " << matchedCount << " matched / " << sections.size()
              << " samples; unmatched reasons saved separately.\n";

    // --- 5. 输出分类 PCD ---
    std::string outputName = "result_final.pcd";
    processor.saveVisualizationCloud(gridMap, gridSize, min_pt, max_pt, outputName);

    std::ofstream diagnostic("classification.csv");
    diagnostic << std::fixed << std::setprecision(6);
    diagnostic << "r,c,z,type,count,crest,toe\n";
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const auto& cell = gridMap[r][c];
            diagnostic << r << ',' << c << ',' << cell.avg_z << ',' << cell.type
                << ',' << cell.count << ',' << cell.is_crest << ',' << cell.is_toe << '\n';
        }
    std::ofstream origin("grid_origin.txt");
    origin << std::setprecision(12) << min_pt[0] << ' ' << min_pt[1] << ' ' << gridSize << '\n';

    std::cout << "\n处理完成。特征线已保存至 feature_lines.csv" << std::endl;
    return 0;
}
