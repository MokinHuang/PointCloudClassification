#include "PointCloudClassification.h"
#include "FeatureLine.h"
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <fstream>
#include <iomanip>

// ── 8邻域行/列偏移量（用于连通分量搜索）──
static const int DR[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
static const int DC[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };

// ── 网格坐标→世界坐标（格子中心点），配合 min_pt 偏移还原真实位置 ──
static Eigen::Vector3f gridToWorld(int r, int c, float z, float gridSize,
                                   const Eigen::Vector4f& min_pt) {
    return Eigen::Vector3f(
        c * gridSize + min_pt[0] + gridSize * 0.5f,
        r * gridSize + min_pt[1] + gridSize * 0.5f,
        z
    );
}

// ── BFS 8邻域连通分量标记：将离散点按连通性分组，每组是一条候选线段 ──
static std::vector<std::vector<GridIndex>> connectedComponents(
    const std::unordered_set<GridIndex, GridIndexHash>& points,
    int rows, int cols)
{
    std::unordered_set<GridIndex, GridIndexHash> visited;
    std::vector<std::vector<GridIndex>> components;

    for (const auto& seed : points) {
        if (visited.count(seed)) continue;

        std::vector<GridIndex> comp;
        std::queue<GridIndex> q;
        q.push(seed);
        visited.insert(seed);

        while (!q.empty()) {
            GridIndex cur = q.front(); q.pop();
            comp.push_back(cur);

            for (int k = 0; k < 8; ++k) {
                GridIndex nb = { cur.r + DR[k], cur.c + DC[k] };
                if (nb.r < 0 || nb.r >= rows || nb.c < 0 || nb.c >= cols) continue;
                if (points.count(nb) && !visited.count(nb)) {
                    visited.insert(nb);
                    q.push(nb);
                }
            }
        }
        components.push_back(std::move(comp));
    }
    return components;
}

// ── 端点追踪排序：构建邻接表 → 找端点(邻接数=1) → 从端点出发沿邻接方向追踪，交叉口选最优方向 ──
static std::vector<GridIndex> orderComponentPoints(
    const std::vector<GridIndex>& comp)
{
    if (comp.size() <= 1) return comp;

    // 构建邻接表 (8邻域)
    std::unordered_map<GridIndex, std::vector<GridIndex>, GridIndexHash> adj;
    std::unordered_set<GridIndex, GridIndexHash> inComp(comp.begin(), comp.end());

    for (const auto& p : comp) {
        for (int k = 0; k < 8; ++k) {
            GridIndex nb = { p.r + DR[k], p.c + DC[k] };
            if (inComp.count(nb)) {
                adj[p].push_back(nb);
            }
        }
    }

    // 找端点：邻接数 == 1 的点
    std::vector<GridIndex> endpoints;
    for (const auto& p : comp) {
        if (adj[p].size() == 1) endpoints.push_back(p);
    }

    std::vector<GridIndex> ordered;
    std::unordered_set<GridIndex, GridIndexHash> used;

    // 确定起点
    GridIndex current;
    if (!endpoints.empty()) {
        current = endpoints[0];  // 从端点出发
    } else {
        current = comp[0];       // 闭合环，从任意点出发
    }

    while (true) {
        ordered.push_back(current);
        used.insert(current);

        // 找下一个未访问的邻接点
        GridIndex next = { -1, -1 };
        const auto& neighbors = adj[current];
        int unvisitedCount = 0;

        for (const auto& nb : neighbors) {
            if (!used.count(nb)) {
                next = nb;
                unvisitedCount++;
            }
        }

        if (unvisitedCount == 0) break;

        // 多个未访问邻接点（交叉口）：选与当前行进方向最一致的方向
        if (unvisitedCount > 1) {
            float bestDot = -2.0f;
            GridIndex bestNb = next;
            // 用最近两个点的向量作为当前方向
            float dirR = 0.0f, dirC = 0.0f;
            if (ordered.size() >= 2) {
                const auto& prev = ordered[ordered.size() - 2];
                dirR = static_cast<float>(current.r - prev.r);
                dirC = static_cast<float>(current.c - prev.c);
            }

            for (const auto& nb : neighbors) {
                if (used.count(nb)) continue;
                float dR = static_cast<float>(nb.r - current.r);
                float dC = static_cast<float>(nb.c - current.c);
                float len = std::sqrt(dR * dR + dC * dC);
                if (len < 1e-6f) continue;
                float dot = (dirR * dR + dirC * dC) / len;
                if (ordered.size() < 2) dot = 1.0f; // 第一步无所谓方向
                if (dot > bestDot) {
                    bestDot = dot;
                    bestNb = nb;
                }
            }
            next = bestNb;
        }

        current = next;
    }

    return ordered;
}

// ── 特征线提取主函数：收集crest/toe点 → 连通分量标记 → 排序 → 坐标转换 → 间隙桥接 ──
std::vector<FeatureLine> PointCloudClassification::extractFeatureLines(
    const std::vector<std::vector<GridCell>>& gridMap,
    float gridSize,
    const Eigen::Vector4f& min_pt,
    int featureType,
    int minComponentSize,
    float maxGapDist)
{
    std::vector<FeatureLine> results;
    int rows = static_cast<int>(gridMap.size());
    if (rows == 0) return results;
    int cols = static_cast<int>(gridMap[0].size());

    // -------- 1. 收集特征点 --------
    // 直接用 is_crest / is_toe 标记（依赖 extractSlopeFeatures 先调用）
    std::unordered_set<GridIndex, GridIndexHash> crestSet, toeSet;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (gridMap[r][c].is_empty) continue;
            if (gridMap[r][c].is_crest) crestSet.insert({ r, c });
            if (gridMap[r][c].is_toe)   toeSet.insert({ r, c });
        }
    }

    // -------- 2. 连通分量标记 --------
    auto processSet = [&](const std::unordered_set<GridIndex, GridIndexHash>& pointSet,
                          int lineType) {
        if (pointSet.empty()) return;

        auto components = connectedComponents(pointSet, rows, cols);

        for (auto& comp : components) {
            if (static_cast<int>(comp.size()) < minComponentSize) continue;

            // 排序
            auto ordered = orderComponentPoints(comp);
            if (ordered.size() < 2) continue;

            // 转世界坐标
            FeatureLine line;
            line.type = lineType;
            line.id   = static_cast<int>(results.size());

            for (const auto& idx : ordered) {
                float z = gridMap[idx.r][idx.c].avg_z;
                line.points.push_back(gridToWorld(idx.r, idx.c, z, gridSize, min_pt));
            }

            // 计算 3D 长度
            line.length = 0.0f;
            for (size_t i = 1; i < line.points.size(); ++i) {
                line.length += (line.points[i] - line.points[i - 1]).norm();
            }

            results.push_back(std::move(line));
        }
    };

    if (featureType == 0 || featureType == 2) processSet(crestSet, 0);
    if (featureType == 1 || featureType == 2) processSet(toeSet,   1);

    // -------- 3. 间隙桥接：合并间距 < maxGapDist 的同类型线段 --------
    if (maxGapDist > 0.0f) {
        results = mergeNearbyLines(results, maxGapDist);
    }

    std::cout << "[特征线] 提取完成: 坡顶线 "
              << std::count_if(results.begin(), results.end(),
                               [](const FeatureLine& l) { return l.type == 0; })
              << " 条, 坡底线 "
              << std::count_if(results.begin(), results.end(),
                               [](const FeatureLine& l) { return l.type == 1; })
              << " 条" << std::endl;

    return results;
}

// ── Union-Find合并断线：检查4种端点组合(头头/头尾/尾头/尾尾)，间距<maxDist则合并 ──
std::vector<FeatureLine> PointCloudClassification::mergeNearbyLines(
    const std::vector<FeatureLine>& lines, float maxDist)
{
    int n = static_cast<int>(lines.size());
    if (n <= 1) return lines;

    // Union-Find
    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;

    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra != rb) parent[ra] = rb;
    };

    // 检查所有同类型线段对的端点距离
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (lines[i].type != lines[j].type) continue;
            if (lines[i].points.empty() || lines[j].points.empty()) continue;

            // 检查 4 种端点组合
            const auto& head_i = lines[i].points.front();
            const auto& tail_i = lines[i].points.back();
            const auto& head_j = lines[j].points.front();
            const auto& tail_j = lines[j].points.back();

            float d1 = (head_i - head_j).norm();
            float d2 = (head_i - tail_j).norm();
            float d3 = (tail_i - head_j).norm();
            float d4 = (tail_i - tail_j).norm();

            if (d1 <= maxDist || d2 <= maxDist || d3 <= maxDist || d4 <= maxDist) {
                unite(i, j);
            }
        }
    }

    // 按根分组
    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < n; ++i) groups[find(i)].push_back(i);

    std::vector<FeatureLine> merged;
    for (auto& [root, indices] : groups) {
        if (indices.size() == 1) {
            merged.push_back(lines[indices[0]]);
            continue;
        }

        // 简易合并：按线段 ID 顺序拼接（实际拼接需要端点匹配逻辑，这里提供基础版）
        FeatureLine combined;
        combined.type = lines[indices[0]].type;
        combined.id   = static_cast<int>(merged.size());

        for (int idx : indices) {
            for (const auto& pt : lines[idx].points) {
                combined.points.push_back(pt);
            }
        }

        combined.length = 0.0f;
        for (size_t i = 1; i < combined.points.size(); ++i) {
            combined.length += (combined.points[i] - combined.points[i - 1]).norm();
        }

        merged.push_back(std::move(combined));
    }

    return merged;
}

// ── 特征线→CSV：line_id,type,type_name,point_order,x,y,z，线段间空行分隔，GIS可导入 ──
void PointCloudClassification::saveFeatureLinesToTxt(
    const std::vector<FeatureLine>& lines, const std::string& filename)
{
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "[错误] 无法创建文件: " << filename << std::endl;
        return;
    }

    ofs << std::fixed << std::setprecision(4);
    ofs << "line_id,type,type_name,point_order,x,y,z\n";

    for (const auto& line : lines) {
        const char* typeName = (line.type == 0) ? "crest" : "toe";
        for (size_t i = 0; i < line.points.size(); ++i) {
            const auto& p = line.points[i];
            ofs << line.id << ","
                << line.type << ","
                << typeName << ","
                << i << ","
                << p.x() << ","
                << p.y() << ","
                << p.z() << "\n";
        }
        // 线段间空行分隔（GIS 常见格式）
        ofs << "\n";
    }

    ofs.close();
    std::cout << "[特征线] 已保存至: " << filename
              << "  (共 " << lines.size() << " 条)" << std::endl;
}

