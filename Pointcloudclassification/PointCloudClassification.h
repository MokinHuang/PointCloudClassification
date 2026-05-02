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

// 1. ���ݽṹ����
struct GridCell {
    float sum_z = 0.0f;
    int count = 0;
    float avg_z = 0.0f;
    bool is_empty = true;
    float slope = 0.0f;
    int type = 0; // 0: ������, 1: ƽ̨, 2: ����
    bool is_crest = false; // �¶���־
    bool is_toe = false;   // �µױ�־
};

struct FeaturePoint {
    float x, y, z;
    int grid_r, grid_c;
};

struct SlopeParams {
    float benchHeight;    // H2
    float platformWidth;  // Lh
    float benchAngle;     // alpha
    float overallAngle;   // phi
};

struct BoundaryPoint {
    int r, c;
    float z;
};

struct SlopeSection {
    BoundaryPoint crest; // �¶�
    BoundaryPoint toe;   // �µ�
    float height;        // ̨�׸߶�
    float angle;         // �����
};

// 2. �ඨ��
class PointCloudClassification {
public:
    // ���캯��
    PointCloudClassification(float size = 1.0f) : gridSize(size) {}

    // �����㷨�������ϸ�ƥ�� .cpp ǩ����
    void fillEmptyGrids(std::vector<std::vector<GridCell>>& gridMap, int rows, int cols);

    void classifyGrids(std::vector<std::vector<GridCell>>& gridMap, float gridSize);

    void saveVisualizationCloud(const std::vector<std::vector<GridCell>>& gridMap,
        float gridSize,
        const Eigen::Vector4f& min_pt,
        const Eigen::Vector4f& max_pt,
        const std::string& fileName);

    void extractTopBottomPoints(std::vector<std::vector<GridCell>>& gridMap,
        float gridSize,
        std::vector<FeaturePoint>& topPoints,
        std::vector<FeaturePoint>& bottomPoints);

    float getScore(float actual, float design, std::string type);

    float calculateSMSI(const std::vector<SlopeParams>& allSegments, const SlopeParams& design);

    bool searchLocalTopBottom(const std::vector<std::vector<GridCell>>& gridMap,
        int r, int c, float gridSize, const Eigen::Vector4f& min_pt,
        FeaturePoint& top, FeaturePoint& bottom);

    void refineClassification(std::vector<std::vector<GridCell>>& gridMap);

    void cleanPlatformNoise(std::vector<std::vector<GridCell>>& gridMap);

    void closeSlopeHoles(std::vector<std::vector<GridCell>>& gridMap);

    void identifyBerms(std::vector<std::vector<GridCell>>& gridMap);

    void fillBermGaps(std::vector<std::vector<GridCell>>& gridMap);

    void extractSlopeFeatures(
        std::vector<std::vector<GridCell>>& gridMap,
        std::vector<SlopeParams>& results
    );

    // ── 特征线提取（需先调用 extractSlopeFeatures 标记 crest/toe） ──
    // featureType: 0=仅坡顶线, 1=仅坡底线, 2=全部
    // minComponentSize: 最小连通点数，过滤碎线
    // maxGapDist: 端点间距小于此值的同类型线段将被合并
    std::vector<FeatureLine> extractFeatureLines(
        const std::vector<std::vector<GridCell>>& gridMap,
        float gridSize,
        const Eigen::Vector4f& min_pt,
        int featureType = 2,
        int minComponentSize = 3,
        float maxGapDist = 3.0f);

    // 合并端点间距过小的同类型线段
    std::vector<FeatureLine> mergeNearbyLines(
        const std::vector<FeatureLine>& lines, float maxDist);

    // 保存为 CSV 文本文件（GIS 可导入）
    void saveFeatureLinesToTxt(
        const std::vector<FeatureLine>& lines, const std::string& filename);

    // 保存为 DXF 格式（AutoCAD / CASS 可打开）
    void saveFeatureLinesToDXF(
        const std::vector<FeatureLine>& lines, const std::string& filename);

private:
    float gridSize;
    static constexpr float W_HEIGHT = 0.20f;
    static constexpr float W_WIDTH = 0.15f;
    static constexpr float W_BENCH_ANGLE = 0.05f;
    static constexpr float W_OVERALL_ANGLE = 0.60f;
};