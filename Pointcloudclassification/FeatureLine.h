#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

// ── 网格单元 ──
struct GridCell {
    float sum_z = 0.0f;
    int   count = 0;
    float avg_z = 0.0f;
    bool  is_empty = true;
    float slope = 0.0f;
    int   type = 0;          // 0=未分类, 1=平台, 2=边坡, 4=围堰
    bool  is_crest = false;  // 坡顶标记
    bool  is_toe = false;    // 坡底标记
};

// ── 边坡截面参数 ──
struct SlopeParams {
    float benchHeight;
    float platformWidth;
    float benchAngle;
    float overallAngle;
};

// ── 特征线：有序顶点序列 ──
struct FeatureLine {
    std::vector<Eigen::Vector3f> points;
    int type;       // 0=坡顶(Crest), 1=坡底(Toe)
    float length;
    int id;

    FeatureLine() : type(0), length(0.0f), id(-1) {}
};

// ── BFS连通分量标记用 ──
struct GridIndex {
    int r, c;
    bool operator==(const GridIndex& o) const { return r == o.r && c == o.c; }
};

struct GridIndexHash {
    size_t operator()(const GridIndex& idx) const {
        return std::hash<int>()(idx.r) ^ (std::hash<int>()(idx.c) << 1);
    }
};

// ── 特征线提取 (独立函数) ──
std::vector<FeatureLine> extractFeatureLines(
    const std::vector<std::vector<GridCell>>& gridMap,
    float gridSize,
    const Eigen::Vector4f& min_pt,
    int featureType = 2,
    int minComponentSize = 3,
    float maxGapDist = 3.0f);

std::vector<FeatureLine> mergeNearbyLines(
    const std::vector<FeatureLine>& lines, float maxDist);

void saveFeatureLinesToTxt(
    const std::vector<FeatureLine>& lines, const std::string& filename);
