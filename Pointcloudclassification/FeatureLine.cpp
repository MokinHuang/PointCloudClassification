#include "FeatureLine.h"
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <deque>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── 8邻域偏移量 ──
static const int DR[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
static const int DC[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };

// ── 网格坐标→世界坐标 (格子中心) ──
static Eigen::Vector3f gridToWorld(int r, int c, float z, float gridSize,
                                   const Eigen::Vector4f& min_pt) {
    return Eigen::Vector3f(
        c * gridSize + min_pt[0] + gridSize * 0.5f,
        r * gridSize + min_pt[1] + gridSize * 0.5f,
        z
    );
}

// ── BFS 8邻域连通分量 ──
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

// ── 端点追踪排序 ──
static std::vector<GridIndex> orderComponentPoints(
    const std::vector<GridIndex>& comp)
{
    if (comp.size() <= 1) return comp;

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

    // 找端点 (邻接数=1)
    std::vector<GridIndex> endpoints;
    for (const auto& p : comp) {
        if (adj[p].size() == 1) endpoints.push_back(p);
    }

    std::vector<GridIndex> ordered;
    std::unordered_set<GridIndex, GridIndexHash> used;

    GridIndex current;
    if (!endpoints.empty()) current = endpoints[0];
    else                    current = comp[0];

    while (true) {
        ordered.push_back(current);
        used.insert(current);

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

        if (unvisitedCount > 1) {
            float bestDot = -2.0f;
            GridIndex bestNb = next;
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
                if (ordered.size() < 2) dot = 1.0f;
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

// Keep the original primary trace, then recover vertices on unvisited branches.
// New branches begin at an existing anchor; no jump between disconnected points.
static std::vector<std::vector<GridIndex>> orderAllComponentPaths(
    const std::vector<GridIndex>& comp)
{
    std::vector<std::vector<GridIndex>> paths{orderComponentPoints(comp)};
    std::unordered_set<GridIndex, GridIndexHash> inComp(comp.begin(), comp.end());
    std::unordered_set<GridIndex, GridIndexHash> used(paths[0].begin(), paths[0].end());
    while (used.size() < comp.size()) {
        GridIndex anchor{-1,-1}, first{-1,-1};
        for (const auto& p : comp) {
            if (!used.count(p)) continue;
            for (int k=0;k<8;++k) {
                GridIndex nb{p.r+DR[k],p.c+DC[k]};
                if (inComp.count(nb) && !used.count(nb)) { anchor=p;first=nb;break; }
            }
            if (first.r>=0) break;
        }
        if (first.r<0) break; // Components are connected by construction.
        std::vector<GridIndex> branch{anchor,first};
        used.insert(first);
        while (true) {
            const auto cur=branch.back(), prev=branch[branch.size()-2];
            GridIndex next{-1,-1};float best=-2;
            for (int k=0;k<8;++k) {
                GridIndex nb{cur.r+DR[k],cur.c+DC[k]};
                if (!inComp.count(nb) || used.count(nb)) continue;
                const float len=std::hypot(float(DR[k]),float(DC[k]));
                const float dot=((cur.r-prev.r)*DR[k]+(cur.c-prev.c)*DC[k])/len;
                if (dot>best) {best=dot;next=nb;}
            }
            if (next.r<0) break;
            branch.push_back(next);used.insert(next);
        }
        paths.push_back(std::move(branch));
    }
    return paths;
}

// ── 特征线提取 ──
std::vector<FeatureLine> extractFeatureLines(
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

    // 1. 收集坡顶/坡底点
    std::unordered_set<GridIndex, GridIndexHash> crestSet, toeSet;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (gridMap[r][c].is_empty) continue;
            if (gridMap[r][c].is_crest) crestSet.insert({ r, c });
            if (gridMap[r][c].is_toe)   toeSet.insert({ r, c });
        }
    }

    // 2. 连通分量→排序→世界坐标
    auto processSet = [&](const std::unordered_set<GridIndex, GridIndexHash>& pointSet,
                          int lineType) {
        if (pointSet.empty()) return;

        auto components = connectedComponents(pointSet, rows, cols);

        for (auto& comp : components) {
            if (static_cast<int>(comp.size()) < 3) continue;

            auto paths = orderAllComponentPaths(comp);
            for (size_t pathIndex=0;pathIndex<paths.size();++pathIndex) {
            const auto& ordered = paths[pathIndex];
            if (ordered.size() < 2) continue;

            FeatureLine line;
            line.type = lineType;
            line.id   = static_cast<int>(results.size());
            line.is_branch = pathIndex > 0;

            for (const auto& idx : ordered) {
                float z = gridMap[idx.r][idx.c].avg_z;
                line.points.push_back(gridToWorld(idx.r, idx.c, z, gridSize, min_pt));
            }

            line.length = 0.0f;
            for (size_t i = 1; i < line.points.size(); ++i) {
                line.length += (line.points[i] - line.points[i - 1]).norm();
            }

            results.push_back(std::move(line));
            }
        }
    };

    if (featureType == 0 || featureType == 2) processSet(crestSet, 0);
    if (featureType == 1 || featureType == 2) processSet(toeSet,   1);

    // 3. 间隙桥接（碎线充当"桥墩"）
    if (maxGapDist > 0.0f) {
        results = mergeNearbyLines(results, maxGapDist);
    }

    // 4. 合并后再按最小长度过滤
    if (minComponentSize > 3) {
        results.erase(std::remove_if(results.begin(), results.end(),
            [&](const FeatureLine& l) { return static_cast<int>(l.points.size()) < minComponentSize; }),
            results.end());
    }

    return results;
}

// Preserve every input line; retry pending attachments whenever either end grows.
// Same type and original 3D endpoint distance threshold; no new geometry veto.
std::vector<FeatureLine> mergeNearbyLines(
    const std::vector<FeatureLine>& lines, float maxDist)
{
    std::vector<FeatureLine> merged;
    std::vector<bool> used(lines.size(),false);
    for (size_t seed=0;seed<lines.size();++seed) {
        if (used[seed]) continue;
        used[seed]=true;
        if (lines[seed].points.empty()) {merged.push_back(lines[seed]);continue;}
        FeatureLine combined=lines[seed];
        std::deque<Eigen::Vector3f> pts(combined.points.begin(),combined.points.end());
        bool changed;
        do {
            changed=false;
            for (size_t j=0;j<lines.size();++j) {
                if (used[j] || lines[j].type!=combined.type || lines[j].points.empty()) continue;
                const auto& nl=lines[j].points;
                const float d_th=(pts.back()-nl.front()).norm();
                const float d_tt=(pts.back()-nl.back()).norm();
                const float d_hh=(pts.front()-nl.front()).norm();
                const float d_ht=(pts.front()-nl.back()).norm();
                if (d_th<=maxDist) {
                    for (const auto& p:nl) pts.push_back(p);
                } else if (d_tt<=maxDist) {
                    for (auto it=nl.rbegin();it!=nl.rend();++it) pts.push_back(*it);
                } else if (d_hh<=maxDist) {
                    for (const auto& p:nl) pts.push_front(p);
                } else if (d_ht<=maxDist) {
                    for (auto it=nl.rbegin();it!=nl.rend();++it) pts.push_front(*it);
                } else continue;
                combined.is_branch=combined.is_branch || lines[j].is_branch;
                used[j]=true;changed=true;
            }
        } while(changed);
        combined.points.assign(pts.begin(),pts.end());
        combined.length=0;
        for(size_t i=1;i<combined.points.size();++i)
            combined.length+=(combined.points[i]-combined.points[i-1]).norm();
        combined.id=(int)merged.size();
        merged.push_back(std::move(combined));
    }
    return merged;
}

// ── 矢量折线平滑：对内部点做 Laplacian 平滑 P_new = 0.5*P + 0.25*P_prev + 0.25*P_next ──
void smoothFeatureLines(std::vector<FeatureLine>& lines, int iterations) {
    for (auto& line : lines) {
        if (line.points.size() < 3) continue;
        for (int iter = 0; iter < iterations; ++iter) {
            std::vector<Eigen::Vector3f> smoothed = line.points;
            for (size_t i = 1; i + 1 < line.points.size(); ++i) {
                smoothed[i] = 0.5f * line.points[i]
                            + 0.25f * line.points[i - 1]
                            + 0.25f * line.points[i + 1];
            }
            line.points = std::move(smoothed);
        }
        line.length = 0.0f;
        for (size_t i = 1; i < line.points.size(); ++i)
            line.length += (line.points[i] - line.points[i - 1]).norm();
    }
}

// ── 特征线→CSV ──
void saveFeatureLinesToTxt(
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
        ofs << "\n";
    }

    ofs.close();
    std::cout << "[特征线] 已保存至: " << filename
              << "  (共 " << lines.size() << " 条)" << std::endl;
}

// ── 坡顶坡底空间匹配：每个crest点找水平距离最近且满足高差条件的toe点 ──
std::vector<MatchedPair> matchCrestAndToe(
    const std::vector<FeatureLine>& lines,
    float maxSearchDistXY, float minDropZ)
{
    std::vector<Eigen::Vector3f> crestPts, toePts;
    for (const auto& line : lines) {
        if (line.type == 0) {
            for (const auto& p : line.points) crestPts.push_back(p);
        } else if (line.type == 1) {
            for (const auto& p : line.points) toePts.push_back(p);
        }
    }

    std::vector<MatchedPair> results;
    for (const auto& crest : crestPts) {
        float bestDist = maxSearchDistXY;
        Eigen::Vector3f bestToe;
        bool found = false;

        for (const auto& toe : toePts) {
            float dx = crest.x() - toe.x();
            float dy = crest.y() - toe.y();
            float d_xy = std::sqrt(dx * dx + dy * dy);
            float d_z = crest.z() - toe.z();

            if (d_xy < maxSearchDistXY && d_z > minDropZ && d_xy < bestDist) {
                bestDist = d_xy;
                bestToe = toe;
                found = true;
            }
        }

        if (found) {
            MatchedPair pair;
            pair.crest_pt = crest;
            pair.toe_pt = bestToe;
            pair.horizontal_dist = bestDist;
            pair.vertical_drop = crest.z() - bestToe.z();
            pair.slope_angle = std::atan2(pair.vertical_drop, bestDist) * 180.0 / M_PI;
            results.push_back(pair);
        }
    }
    return results;
}

// ── 匹配结果→CSV ──
void saveMatchedPairsToCSV(
    const std::vector<MatchedPair>& pairs, const std::string& filename)
{
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "[错误] 无法创建文件: " << filename << std::endl;
        return;
    }

    ofs << std::fixed << std::setprecision(4);
    ofs << "crest_x,crest_y,crest_z,toe_x,toe_y,toe_z,h_dist,v_drop,angle\n";

    for (const auto& p : pairs) {
        ofs << p.crest_pt.x() << "," << p.crest_pt.y() << "," << p.crest_pt.z() << ","
            << p.toe_pt.x()   << "," << p.toe_pt.y()   << "," << p.toe_pt.z()   << ","
            << p.horizontal_dist << "," << p.vertical_drop << "," << p.slope_angle << "\n";
    }

    ofs.close();
    std::cout << "[匹配] 已保存至: " << filename
              << "  (共 " << pairs.size() << " 对)" << std::endl;
}
