#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <Eigen/Dense>
#include "FeatureLine.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 网格单元：点云降采样后每个格子的状态
struct GridCell {
    float sum_z = 0.0f;         // 落入该格点的z值累加
    int   count = 0;            // 落入该格的原始点数
    float avg_z = 0.0f;         // 平均高度
    bool  is_empty = true;      // 是否无数据
    float slope = 0.0f;         // 坡度角(度)
    int   type = 0;             // 0=未分类, 1=平台, 2=边坡, 4=围堰
    bool  is_crest = false;     // 坡顶标记
    bool  is_toe = false;       // 坡底标记
};

// 边坡截面参数
struct SlopeParams {
    float benchHeight;          // 台阶高度 H
    float platformWidth;        // 平台宽度 Lh
    float benchAngle;           // 台阶坡面角 alpha
    float overallAngle;         // 综合坡面角 phi
};

class PointCloudClassification {
public:
    PointCloudClassification(float size = 1.0f) : gridSize(size) {}

    // ── 预处理 ──

    // 空格子插值：逐级扩大窗口(3x3→5x5→7x7)，用非空邻居的均值填充
    void fillEmptyGrids(std::vector<std::vector<GridCell>>& gridMap, int rows, int cols);

    // 坡度分类：中心差分求梯度 → 梯度模长转角度 → <6°平台, >12°边坡, 其余未分类
    void classifyGrids(std::vector<std::vector<GridCell>>& gridMap, float gridSize);

    // ── 形态学优化 ──

    // 邻域投票：3x3窗口中边坡多→转边坡，平台多→转平台，用于消除过渡带
    void refineClassification(std::vector<std::vector<GridCell>>& gridMap);

    // 平台去噪：边坡格8邻域中平台数量≥5时转为平台，消除平台上孤立噪点
    void cleanPlatformNoise(std::vector<std::vector<GridCell>>& gridMap);

    // 边坡闭洞：先膨胀(邻域有边坡即转边坡)再腐蚀(邻域≥6平台即转回)，填补边坡内部空洞
    void closeSlopeHoles(std::vector<std::vector<GridCell>>& gridMap);

    // ── 围堰处理 ──

    // 围堰识别：5x5窗口检测局部凸起(0.35~2.2m高差+高于均值+局域高差<3m)，连通滤波去噪
    void identifyBerms(std::vector<std::vector<GridCell>>& gridMap);

    // 围堰间隙填补：5次迭代，三规则(夹层/桥接/孤立红)将边坡格转为围堰
    void fillBermGaps(std::vector<std::vector<GridCell>>& gridMap);

    // ── 特征提取 ──

    // 双向扫描提取坡顶/坡底：逐行+逐列检测平台→边坡/围堰→平台的剖面序列
    // 标记 is_crest/is_toe，输出 SlopeParams（台阶高度、坡面角）
    void extractSlopeFeatures(
        std::vector<std::vector<GridCell>>& gridMap,
        std::vector<SlopeParams>& results
    );

    // 特征线提取：收集crest/toe点 → BFS 8邻域连通分量 → 端点追踪排序 → 3D多段线
    // featureType: 0=仅坡顶, 1=仅坡底, 2=全部
    // minComponentSize: 最小连通点数过滤碎线
    // maxGapDist: 端点间距小于此值的同类型线段将被合并
    std::vector<FeatureLine> extractFeatureLines(
        const std::vector<std::vector<GridCell>>& gridMap,
        float gridSize,
        const Eigen::Vector4f& min_pt,
        int featureType = 2,
        int minComponentSize = 3,
        float maxGapDist = 3.0f);

    // Union-Find合并端点间距<maxDist的同类型断线（被extractFeatureLines内部调用）
    std::vector<FeatureLine> mergeNearbyLines(
        const std::vector<FeatureLine>& lines, float maxDist);

    // ── 输出 ──

    // 特征线→CSV文本（line_id,type,type_name,point_order,x,y,z），GIS可导入
    void saveFeatureLinesToTxt(
        const std::vector<FeatureLine>& lines, const std::string& filename);

    // 分类网格→彩色PCD文件（绿=平台, 红=边坡, 黄=围堰, 黑=坡顶标记, 蓝=坡底标记）
    void saveVisualizationCloud(
        const std::vector<std::vector<GridCell>>& gridMap,
        float gridSize,
        const Eigen::Vector4f& min_pt,
        const Eigen::Vector4f& max_pt,
        const std::string& fileName);

private:
    float gridSize;
};
