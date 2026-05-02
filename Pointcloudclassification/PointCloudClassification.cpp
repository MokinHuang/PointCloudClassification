#include "PointCloudClassification.h"

void PointCloudClassification::fillEmptyGrids(std::vector<std::vector<GridCell>>& gridMap, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (gridMap[r][c].is_empty) {
                int win = 1;
                bool found = false;
                while (!found && win <= 3) {
                    float neighbor_sum = 0.0f;
                    int neighbor_count = 0;
                    for (int i = -win; i <= win; ++i) {
                        for (int j = -win; j <= win; ++j) {
                            int nr = r + i, nc = c + j;
                            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !gridMap[nr][nc].is_empty) {
                                neighbor_sum += gridMap[nr][nc].avg_z;
                                neighbor_count++;
                            }
                        }
                    }
                    if (neighbor_count > 0) {
                        gridMap[r][c].avg_z = neighbor_sum / neighbor_count;
                        gridMap[r][c].is_empty = false;
                        found = true;
                    }
                    else win++;
                }
            }
        }
    }
}

void PointCloudClassification::classifyGrids(std::vector<std::vector<GridCell>>& gridMap, float gridSize) {
    int rows = (int)gridMap.size();
    int cols = (int)gridMap[0].size();
    const float p1 = 6.0f, p2 = 12.0f;  //设定边坡识别阈值
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (gridMap[r][c].is_empty) continue;
            float dz_dx = (gridMap[r][c + 1].avg_z - gridMap[r][c - 1].avg_z) / (2.0f * gridSize);
            float dz_dy = (gridMap[r + 1][c].avg_z - gridMap[r - 1][c].avg_z) / (2.0f * gridSize);
            float gradient = std::sqrt(dz_dx * dz_dx + dz_dy * dz_dy);
            float slope_deg = std::atan(gradient) * 180.0f / (float)M_PI;
            gridMap[r][c].slope = slope_deg;
            if (slope_deg < p1) gridMap[r][c].type = 1;
            else if (slope_deg > p2) gridMap[r][c].type = 2;
            else gridMap[r][c].type = 0;
        }
    }
}

void PointCloudClassification::saveVisualizationCloud(const std::vector<std::vector<GridCell>>& gridMap, float gridSize, const Eigen::Vector4f& min_pt, const Eigen::Vector4f& max_pt, const std::string& fileName) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr visCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    int rows = (int)gridMap.size();
    int cols = (int)gridMap[0].size();

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (gridMap[r][c].is_empty) continue;

            pcl::PointXYZRGB pt;
            pt.x = c * gridSize + min_pt[0];
            pt.y = r * gridSize + min_pt[1];
            pt.z = gridMap[r][c].avg_z;

            // --- 核心修改：着色优先级逻辑 ---

            // 1. 优先判定特征线，让它们“盖”在普通颜色上面
           // ... 在 saveVisualizationCloud 循环内部 ...

// 1. 优先判定特征线
            if (gridMap[r][c].is_crest) {
                // 坡顶 - 黑色
                pt.r = 0; pt.g = 0; pt.b = 0;
            }
            else if (gridMap[r][c].is_toe) {
                // 坡底 - 蓝色
                pt.r = 0; pt.g = 0; pt.b = 255;
            }
            // 2. 语义分类颜色
            else if (gridMap[r][c].type == 1) { // 平台 - 绿色
                pt.r = 0; pt.g = 255; pt.b = 0;
            }
            else if (gridMap[r][c].type == 2) { // 坡面 - 红色
                pt.r = 255; pt.g = 0; pt.b = 0;
            }
            else if (gridMap[r][c].type == 4) { // 挡墙 - 黄色
                pt.r = 255; pt.g = 255; pt.b = 0;
            }
            // ... 后续逻辑 ...
            visCloud->push_back(pt);
        }
    }

    pcl::io::savePCDFileBinary(fileName, *visCloud);
    std::cout << "[可视化] 结果已保存至: " << fileName << " (含特征线标记)" << std::endl;
}

void PointCloudClassification::extractTopBottomPoints(std::vector<std::vector<GridCell>>& gridMap, float gridSize, std::vector<FeaturePoint>& topPoints, std::vector<FeaturePoint>& bottomPoints) {
    int rows = gridMap.size(); int cols = gridMap[0].size();
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (gridMap[r][c].type != 2) continue;
            float dx = (gridMap[r][c + 1].avg_z - gridMap[r][c - 1].avg_z);
            float dy = (gridMap[r + 1][c].avg_z - gridMap[r - 1][c].avg_z);
            float angle = std::atan2(dy, dx);
            FeaturePoint currentTop = { 0, 0, -1e9, r, c };
            FeaturePoint currentBottom = { 0, 0, 1e9, r, c };
            for (float step = -15.0f; step <= 15.0f; step += 0.5f) {
                int nr = r + static_cast<int>(step * std::sin(angle) / gridSize);
                int nc = c + static_cast<int>(step * std::cos(angle) / gridSize);
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && gridMap[nr][nc].type >= 2) {
                    float z = gridMap[nr][nc].avg_z;
                    if (z > currentTop.z) { currentTop.z = z; currentTop.grid_r = nr; currentTop.grid_c = nc; }
                    if (z < currentBottom.z) { currentBottom.z = z; currentBottom.grid_r = nr; currentBottom.grid_c = nc; }
                }
            }
            if (currentTop.z > -1e8) topPoints.push_back(currentTop);
            if (currentBottom.z < 1e8) bottomPoints.push_back(currentBottom);
        }
    }
}

float PointCloudClassification::getScore(float actual, float design, std::string type) {
    float ratio = actual / design;
    if (type == "height") {
        if (ratio >= 2.0f) return 4.0f;
        if (ratio >= 1.6f) return 2.0f;
        if (ratio >= 1.3f) return 1.0f;
        if (ratio >= 1.1f) return 0.5f;
    }
    else if (type == "width") {
        if (ratio <= 0.5f) return 4.0f;
        if (ratio <= 0.75f) return 2.0f;
        if (ratio < 1.0f) return 0.5f;
    }
    else if (type == "benchAngle") {
        float diff = actual - design;
        if (diff >= 15.0f) return 4.0f;
        if (diff >= 10.0f) return 2.0f;
        if (diff >= 5.0f) return 0.5f;
    }
    return 0.0f;
}

float PointCloudClassification::calculateSMSI(const std::vector<SlopeParams>& allSegments, const SlopeParams& design) {
    float totalSMSI = 0.0f;
    for (const auto& s : allSegments) {
        float scoH = getScore(s.benchHeight, design.benchHeight, "height");
        float scoW = getScore(s.platformWidth, design.platformWidth, "width");
        float scoBA = getScore(s.benchAngle, design.benchAngle, "benchAngle");
        float scoOA = getScore(s.overallAngle, design.overallAngle, "overallAngle");
        totalSMSI += (scoH * W_HEIGHT + scoW * W_WIDTH + scoBA * W_BENCH_ANGLE + scoOA * W_OVERALL_ANGLE);
    }
    return totalSMSI;
}

bool PointCloudClassification::searchLocalTopBottom(const std::vector<std::vector<GridCell>>& gridMap, int r, int c, float gridSize, const Eigen::Vector4f& min_pt, FeaturePoint& top, FeaturePoint& bottom) {
    int rows = (int)gridMap.size(); int cols = (int)gridMap[0].size();
    float dz_dx = (gridMap[r][c + 1].avg_z - gridMap[r][c - 1].avg_z) / (2.0f * gridSize);
    float dz_dy = (gridMap[r + 1][c].avg_z - gridMap[r - 1][c].avg_z) / (2.0f * gridSize);
    float angle = std::atan2(dz_dy, dz_dx);
    top.z = -1e9; bottom.z = 1e9;
    bool foundTop = false, foundBottom = false;
    const float maxSearchDist = 20.0f, stepSize = 0.5f;
    for (float dist = -maxSearchDist; dist <= maxSearchDist; dist += stepSize) {
        float offsetX = dist * std::cos(angle);
        float offsetY = dist * std::sin(angle);
        int nr = r + static_cast<int>(std::round(offsetY / gridSize));
        int nc = c + static_cast<int>(std::round(offsetX / gridSize));
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
        if (gridMap[nr][nc].type == 2) {
            float currentZ = gridMap[nr][nc].avg_z;
            if (currentZ > top.z) { top.z = currentZ; top.x = nc * gridSize + min_pt[0]; top.y = nr * gridSize + min_pt[1]; foundTop = true; }
            if (currentZ < bottom.z) { bottom.z = currentZ; bottom.x = nc * gridSize + min_pt[0]; bottom.y = nr * gridSize + min_pt[1]; foundBottom = true; }
        }
    }
    return (foundTop && foundBottom && (top.z - bottom.z > 0.5f));
}

void PointCloudClassification::refineClassification(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = (int)gridMap.size(); int cols = (int)gridMap[0].size();
    auto tempMap = gridMap;
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (gridMap[r][c].type == 0 && !gridMap[r][c].is_empty) {
                int sC = 0, pC = 0;
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        if (i == 0 && j == 0) continue;
                        if (gridMap[r + i][c + j].type == 2) sC++;
                        if (gridMap[r + i][c + j].type == 1) pC++;
                    }
                }
                if (sC > pC) tempMap[r][c].type = 2;
                else if (pC > 0) tempMap[r][c].type = 1;
            }
        }
    }
    gridMap = tempMap;
}

void PointCloudClassification::cleanPlatformNoise(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = (int)gridMap.size(); int cols = (int)gridMap[0].size();
    auto tempMap = gridMap;
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (gridMap[r][c].type == 2) {
                int gN = 0;
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j)
                        if (!(i == 0 && j == 0) && gridMap[r + i][c + j].type == 1) gN++;
                if (gN >= 5) tempMap[r][c].type = 1;
            }
        }
    }
    gridMap = tempMap;
}

void PointCloudClassification::closeSlopeHoles(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = gridMap.size(), cols = gridMap[0].size();
    auto tempMap = gridMap;
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (gridMap[r][c].type == 1) {
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j)
                        if (gridMap[r + i][c + j].type == 2) { tempMap[r][c].type = 2; goto next_dil; }
            }
        next_dil:;
        }
    }
    gridMap = tempMap;
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (tempMap[r][c].type == 2) {
                int gC = 0;
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j)
                        if (tempMap[r + i][c + j].type == 1) gC++;
                if (gC >= 6) gridMap[r][c].type = 1;
            }
        }
    }
}

void PointCloudClassification::identifyBerms(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = (int)gridMap.size();
    int cols = (int)gridMap[0].size();
    auto tempMap = gridMap;

    const float minBermH = 0.35f;   // 调低一点点，防止漏检
    const float maxBermH = 2.2f;
    const int radius = 2;           // 5x5 邻域

    for (int r = radius; r < rows - radius; ++r) {
        for (int c = radius; c < cols - radius; ++c) {
            if (gridMap[r][c].is_empty) continue;

            // 1. 获取局部基准：寻找 5x5 邻域内的最低和最高
            float minZ = 1e9, maxZ = -1e9;
            float sumZ = 0;
            int count = 0;
            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    float z = gridMap[r + i][c + j].avg_z;
                    if (z < minZ) minZ = z;
                    if (z > maxZ) maxZ = z;
                    sumZ += z;
                    count++;
                }
            }
            float avgZ = sumZ / count;
            float diffH = gridMap[r][c].avg_z - minZ;

            // 2. 关键：计算“局部粗糙度”（这里用极差和标准差的简化版）
            // 平台虽然有高差，但它在 5x5 范围内通常是斜面，而围挡是凸起
            bool isBump = (gridMap[r][c].avg_z > avgZ + 0.15f); // 显著高于平均值

            // 3. 语义过滤：围挡必须“孤独”
            // 如果 5x5 范围内最高和最低差了 10 米，那肯定是大边坡
            bool notBigSlope = (maxZ - minZ < 3.0f);

            if (diffH >= minBermH && diffH <= maxBermH && isBump && notBigSlope) {
                tempMap[r][c].type = 4;
            }
            else {
                // 如果不满足，维持原样或打回原形
                if (tempMap[r][c].type == 4) tempMap[r][c].type = gridMap[r][c].type;
            }
        }
    }

    // 4. 【大杀器】连通域过滤：清理平台上的“黄色雀斑”
    // 围挡是条状的，连在一起的；平台上的误报通常是孤立的几个点
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (tempMap[r][c].type == 4) {
                int neighborBermCount = 0;
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        if (tempMap[r + i][c + j].type == 4) neighborBermCount++;
                    }
                }
                // 如果周围没几个同类，说明是孤立噪声，删掉
                if (neighborBermCount < 3) {
                    tempMap[r][c].type = gridMap[r][c].type;
                }
            }
        }
    }
    gridMap = tempMap;
}

void PointCloudClassification::fillBermGaps(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = (int)gridMap.size();
    int cols = (int)gridMap[0].size();

    // 迭代 5 次，像刷漆一样反复覆盖，直到缝隙填满
    for (int iter = 0; iter < 5; ++iter) {
        auto tempMap = gridMap;
        for (int r = 2; r < rows - 2; ++r) {
            for (int c = 2; c < cols - 2; ++c) {
                if (gridMap[r][c].type == 2) {

                    // --- 探测周围环境 ---
                    bool hasYellow5x5 = false;
                    bool hasGreen5x5 = false;

                    // 3x3 邻域的方向状态
                    bool N = (gridMap[r - 1][c].type == 4);
                    bool S = (gridMap[r + 1][c].type == 4);
                    bool W = (gridMap[r][c - 1].type == 4);
                    bool E = (gridMap[r][c + 1].type == 4);
                    bool NW = (gridMap[r - 1][c - 1].type == 4);
                    bool SE = (gridMap[r + 1][c + 1].type == 4);
                    bool NE = (gridMap[r - 1][c + 1].type == 4);
                    bool SW = (gridMap[r + 1][c - 1].type == 4);

                    // 1. 5x5 广度搜索（为了夹击逻辑）
                    for (int i = -2; i <= 2; ++i) {
                        for (int j = -2; j <= 2; ++j) {
                            int type = gridMap[r + i][c + j].type;
                            if (type == 4) hasYellow5x5 = true;
                            if (type == 1) hasGreen5x5 = true;
                        }
                    }

                    // --- 核心判定决策树 ---

                    // 逻辑 A：经典的黄绿夹击（5x5 范围）
                    bool isSandwich = (hasYellow5x5 && hasGreen5x5);

                    // 逻辑 B：【新增】桥梁逻辑（3x3 范围）
                    // 只要在水平、垂直或对角线方向上被黄色“夹击”，就判定为围挡的一部分
                    bool isBridge = (N && S) || (W && E) || (NW && SE) || (NE && SW);

                    // 逻辑 C：极度严苛的孤岛判定（只有当周围全是黄或空时）
                    int notYellowCount = 0;
                    for (int i = -1; i <= 1; ++i) {
                        for (int j = -1; j <= 1; ++j) {
                            if (i == 0 && j == 0) continue;
                            int t = gridMap[r + i][c + j].type;
                            if (t != 4 && !gridMap[r + i][c + j].is_empty) notYellowCount++;
                        }
                    }
                    bool isLoneRed = (notYellowCount == 0);

                    if (isSandwich || isBridge || isLoneRed) {
                        tempMap[r][c].type = 4;
                    }
                }
            }
        }
        // 如果这一次迭代没有任何变化，可以提前退出，节省 5700X3D 的功耗
        gridMap = tempMap;
    }
}

void PointCloudClassification::extractSlopeFeatures(std::vector<std::vector<GridCell>>& gridMap, std::vector<SlopeParams>& results) {
    int rows = (int)gridMap.size();
    int cols = (int)gridMap[0].size();

    // 1. 清理标记
    for (auto& row : gridMap) for (auto& cell : row) { cell.is_crest = cell.is_toe = false; }

    // 定义一个 Lambda 内部函数来处理检测逻辑，减少重复代码
    auto scanLogic = [&](int startIdx, int endIdx, int outerIdx, bool isColumnScan) {
        bool searching = false;
        int firstIdx = -1;

        for (int i = 1; i < endIdx - 1; ++i) {
            int r = isColumnScan ? i : outerIdx;
            int c = isColumnScan ? outerIdx : i;
            int pr = isColumnScan ? i - 1 : outerIdx;
            int pc = isColumnScan ? outerIdx : i - 1;

            if (gridMap[r][c].is_empty) continue;

            // 触发：进入坡体（从平台切换到坡面/围挡）
            if (!searching) {
                // 允许前一个点是平台，或者是在 2 格范围内有平台（增加容错）
                bool fromPlatform = (gridMap[pr][pc].type == 1);
                if (fromPlatform && (gridMap[r][c].type == 2 || gridMap[r][c].type == 4)) {
                    firstIdx = i;
                    searching = true;
                }
            }
            // 触发：离开坡体（从坡面/围挡切换回平台）
            else {
                int nr = isColumnScan ? i + 1 : outerIdx;
                int nc = isColumnScan ? outerIdx : i + 1;

                if ((gridMap[r][c].type == 2 || gridMap[r][c].type == 4) && gridMap[nr][nc].type == 1) {
                    int lastIdx = i;

                    // 获取两端点的 Z 值
                    float z1 = gridMap[isColumnScan ? firstIdx : outerIdx][isColumnScan ? outerIdx : firstIdx].avg_z;
                    float z2 = gridMap[isColumnScan ? lastIdx : outerIdx][isColumnScan ? outerIdx : lastIdx].avg_z;

                    float height = std::abs(z1 - z2);
                    if (height > 1.5f) { // 稍微降低高度阈值，增加召回
                        int crestR = (z1 > z2) ? (isColumnScan ? firstIdx : outerIdx) : (isColumnScan ? lastIdx : outerIdx);
                        int crestC = (z1 > z2) ? (isColumnScan ? outerIdx : firstIdx) : (isColumnScan ? outerIdx : lastIdx);
                        int toeR = (z1 > z2) ? (isColumnScan ? lastIdx : outerIdx) : (isColumnScan ? firstIdx : outerIdx);
                        int toeC = (z1 > z2) ? (isColumnScan ? outerIdx : lastIdx) : (isColumnScan ? outerIdx : firstIdx);

                        gridMap[crestR][crestC].is_crest = true;
                        gridMap[toeR][toeC].is_toe = true;

                        SlopeParams p;
                        p.benchHeight = height;
                        float hDist = std::abs(firstIdx - lastIdx) * this->gridSize;
                        p.benchAngle = std::atan2(height, hDist) * 180.0f / 3.14159f;
                        results.push_back(p);
                    }
                    searching = false;
                }
            }
        }
        };

    // 2. 执行双向扫描：先列后行，全方位捕捉
    std::cout << "正在执行双向特征提取..." << std::endl;
    for (int c = 0; c < cols; ++c) scanLogic(0, rows, c, true);  // 纵向扫
    for (int r = 0; r < rows; ++r) scanLogic(0, cols, r, false); // 横向扫
}