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

class PointCloudClassification {
public:
    PointCloudClassification(float size = 1.0f) : gridSize(size) {}

    // ── 预处理 ──

    void fillEmptyGrids(std::vector<std::vector<GridCell>>& gridMap, int rows, int cols);

    void classifyGrids(std::vector<std::vector<GridCell>>& gridMap, float gridSize);

    // ── 形态学优化 ──

    void refineClassification(std::vector<std::vector<GridCell>>& gridMap);

    void cleanPlatformNoise(std::vector<std::vector<GridCell>>& gridMap);

    void closeSlopeHoles(std::vector<std::vector<GridCell>>& gridMap);

    // ── 围堰处理 ──

    void identifyBerms(std::vector<std::vector<GridCell>>& gridMap);

    void fillBermGaps(std::vector<std::vector<GridCell>>& gridMap);

    // ── 特征提取 ──

    void extractSlopeFeatures(
        std::vector<std::vector<GridCell>>& gridMap,
        std::vector<SlopeParams>& results
    );

    // ── 输出 ──

    void saveVisualizationCloud(
        const std::vector<std::vector<GridCell>>& gridMap,
        float gridSize,
        const Eigen::Vector4f& min_pt,
        const Eigen::Vector4f& max_pt,
        const std::string& fileName);

private:
    float gridSize;
};
