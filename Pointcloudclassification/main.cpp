#include <iostream>
#include <vector>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include "PointCloudClassification.h"

int main() {
    // --- 1. 初始化与配置 ---
    std::string filename = "test2.pcd";
    float gridSize = 1.0f; // 格网分辨率 1.0米
    PointCloudClassification processor(gridSize);

    std::cout << "正在加载点云文件: " << filename << "..." << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        std::cerr << "无法读取文件，请检查路径！" << std::endl;
        return -1;
    }

    // --- 2. 建立格网地图 ---
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);

    int cols = static_cast<int>((max_pt[0] - min_pt[0]) / gridSize) + 1;
    int rows = static_cast<int>((max_pt[1] - min_pt[1]) / gridSize) + 1;

    // 初始化格网容器
    std::vector<std::vector<GridCell>> gridMap(rows, std::vector<GridCell>(cols));

    // 填充数据点并计算平均高度
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

    // --- 3. 语义分类流水线 (核心算法) ---
    std::cout << "正在执行语义分类算法流水线..." << std::endl;

    processor.fillEmptyGrids(gridMap, rows, cols);        // 空洞填充
    processor.classifyGrids(gridMap, gridSize);           // 基础红绿分类

    for (int i = 0; i < 2; ++i) {
        processor.refineClassification(gridMap);         // 连通性优化
    }

    processor.cleanPlatformNoise(gridMap);                // 清理平台噪点
    processor.closeSlopeHoles(gridMap);                   // 闭合坡面孔洞

    processor.identifyBerms(gridMap);                     // 识别黄色围挡
    processor.fillBermGaps(gridMap);                      // 3x3/5x5 混合填补围挡缝隙

    // --- 4. 全局特征提取 (坡顶与坡底) ---
    std::cout << "正在提取坡顶(Crest)与坡底(Toe)特征线..." << std::endl;
    std::vector<SlopeParams> measuredParams;
    processor.extractSlopeFeatures(gridMap, measuredParams);

    // --- 5. 最终评估与可视化 ---
    // 设定设计参数：高度10m, 坡角35度, 平台宽35m, 综合角38度
    SlopeParams designParams = { 10.0f, 35.0f, 35.0f, 38.0f };

    // 计算综合安全指数 SMSI
    float totalSmsiSum = processor.calculateSMSI(measuredParams, designParams);

    // 保存结果到 PCD 文件 (此时含红、绿、黄、白、蓝五色)
    std::string outputName = "result_final.pcd";
    processor.saveVisualizationCloud(gridMap, gridSize, min_pt, max_pt, outputName);

    // --- 6. 生成分析报告 ---
    std::cout << "\n==========================================" << std::endl;
    std::cout << "   露天矿山边坡形态自动化评估报告" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "1. 扫描数据概况:" << std::endl;
    std::cout << "   - 格网尺寸: " << gridSize << " m" << std::endl;
    std::cout << "   - 地图规模: " << rows << " x " << cols << std::endl;
    std::cout << "   - 提取有效断面数: " << measuredParams.size() << std::endl;

    if (!measuredParams.empty()) {
        float finalScore = totalSmsiSum / measuredParams.size();
        std::cout << "2. 关键指标:" << std::endl;
        std::cout << "   - 综合安全指数 (ASMSI): " << finalScore << std::endl;

        std::cout << "3. 安全等级评估: ";
        if (finalScore <= 0.5f) {
            std::cout << "[ 优 ] - 状态极佳" << std::endl;
            std::cout << "   建议: 维持现状，按常规周期巡检。" << std::endl;
        }
        else if (finalScore <= 1.0f) {
            std::cout << "[ 良 ] - 轻微超限" << std::endl;
            std::cout << "   建议: 关注黄色围挡完整性，加强局部监测。" << std::endl;
        }
        else {
            std::cout << "[ 警告 ] - 形态严重违规" << std::endl;
            std::cout << "   建议: 立即进行边坡稳定性校核，存在滑坡风险！" << std::endl;
        }
    }
    else {
        std::cout << "错误: 未能识别出任何有效边坡断面，请检查数据质量或分类阈值。" << std::endl;
    }
    std::cout << "==========================================\n" << std::endl;

    return 0;
}