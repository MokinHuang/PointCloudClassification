#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

// 特征线：有序顶点序列
struct FeatureLine {
    std::vector<Eigen::Vector3f> points;  // 有序顶点 (x, y, z)
    int type;       // 0 = 坡顶线(Crest), 1 = 坡底线(Toe)
    float length;   // 3D 总长度
    int id;         // 编号

    FeatureLine() : type(0), length(0.0f), id(-1) {}
};

// 用于连通分量标记的临时结构
struct GridIndex {
    int r, c;
    bool operator==(const GridIndex& o) const { return r == o.r && c == o.c; }
};

struct GridIndexHash {
    size_t operator()(const GridIndex& idx) const {
        return std::hash<int>()(idx.r) ^ (std::hash<int>()(idx.c) << 1);
    }
};
