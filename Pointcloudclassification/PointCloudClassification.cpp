#include "PointCloudClassification.h"

// ── 空格子插值：逐级扩大窗口(3x3→5x5→7x7)，用非空邻居的均值填充 ──
void PointCloudClassification::fillEmptyGrids(std::vector<std::vector<GridCell>>& gridMap, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!gridMap[r][c].is_empty) continue;
            for (int win = 1; win <= 3; ++win) {
                float sum = 0;
                int cnt = 0;
                for (int i = -win; i <= win; ++i) {
                    for (int j = -win; j <= win; ++j) {
                        int nr = r + i, nc = c + j;
                        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !gridMap[nr][nc].is_empty) {
                            sum += gridMap[nr][nc].avg_z;
                            cnt++;
                        }
                    }
                }
                if (cnt > 0) {
                    gridMap[r][c].avg_z = sum / cnt;
                    gridMap[r][c].is_empty = false;
                    break;
                }
            }
        }
    }
}

// ── 坡度分类：中心差分求梯度 → 梯度模长转角度 → <6°平台, >12°边坡, 其余未分类 ──
void PointCloudClassification::classifyGrids(std::vector<std::vector<GridCell>>& gridMap, float gridSize) {
    int rows = (int)gridMap.size();
    int cols = (int)gridMap[0].size();
    const float p1 = 8.0f, p2 = 18.0f;  //�趨����ʶ����ֵ
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

// ── 分类网格→彩色PCD：绿=平台, 红=边坡, 黄=围堰, 黑=坡顶标记格, 蓝=坡底标记格 ──
void PointCloudClassification::saveVisualizationCloud(
    const std::vector<std::vector<GridCell>>& gridMap,
    float gridSize,
    const Eigen::Vector4f& min_pt,
    const Eigen::Vector4f& max_pt,
    const std::string& fileName)
{
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

            if (gridMap[r][c].is_crest) {
                pt.r = 0; pt.g = 0; pt.b = 0;
            } else if (gridMap[r][c].is_toe) {
                pt.r = 0; pt.g = 0; pt.b = 255;
            } else if (gridMap[r][c].type == 1) {
                pt.r = 0; pt.g = 255; pt.b = 0;
            } else if (gridMap[r][c].type == 2) {
                pt.r = 255; pt.g = 0; pt.b = 0;
            } else if (gridMap[r][c].type == 4) {
                pt.r = 255; pt.g = 255; pt.b = 0;
            }
            visCloud->push_back(pt);
        }
    }

    pcl::io::savePCDFileBinary(fileName, *visCloud);
    std::cout << "[可视化] 已保存至: " << fileName << std::endl;
}

// ── 邻域投票重分类：对 type==0 的格子，统计3x3邻居中边坡/平台数量，归入多数类 ──
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

// ── 平台去噪：边坡格8邻域中平台数量≥5时转平台，消除孤立噪点 ──
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

// ── 边坡闭洞(形态学闭运算)：先膨胀(邻域有边坡即转) → 再腐蚀(邻域≥6平台转回) ──
void PointCloudClassification::closeSlopeHoles(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = gridMap.size(), cols = gridMap[0].size();
    auto tempMap = gridMap;
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (gridMap[r][c].type != 1) continue;
            bool hasSlope = false;
            for (int i = -1; i <= 1 && !hasSlope; ++i)
                for (int j = -1; j <= 1; ++j)
                    if (gridMap[r + i][c + j].type == 2) { hasSlope = true; break; }
            if (hasSlope) tempMap[r][c].type = 2;
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

// ── 假边坡过滤：边坡格5x5邻域内高差<1.5m则转为平台，消除碎石/车辙等局部凸起误判 ──
void PointCloudClassification::filterFalseSlopes(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = (int)gridMap.size(), cols = (int)gridMap[0].size();
    auto tempMap = gridMap;
    for (int r = 2; r < rows - 2; ++r) {
        for (int c = 2; c < cols - 2; ++c) {
            if (gridMap[r][c].type != 2) continue;
            float minZ = 1e9f, maxZ = -1e9f;
            for (int i = -2; i <= 2; ++i) {
                for (int j = -2; j <= 2; ++j) {
                    if (gridMap[r + i][c + j].is_empty) continue;
                    float z = gridMap[r + i][c + j].avg_z;
                    if (z < minZ) minZ = z;
                    if (z > maxZ) maxZ = z;
                }
            }
            if (maxZ - minZ < 1.5f) tempMap[r][c].type = 1;
        }
    }
    gridMap = tempMap;
}

// ── 围堰识别：5x5窗口局部凸起检测(0.35~2.2m高差+高于均值+局域高差<3m)→连通滤波去噪 ──
void PointCloudClassification::identifyBerms(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = (int)gridMap.size();
    int cols = (int)gridMap[0].size();
    auto tempMap = gridMap;

    const float minBermH = 0.35f;   // ����һ��㣬��ֹ©��
    const float maxBermH = 2.2f;
    const int radius = 2;           // 5x5 ����

    for (int r = radius; r < rows - radius; ++r) {
        for (int c = radius; c < cols - radius; ++c) {
            if (gridMap[r][c].is_empty) continue;

            // 1. ��ȡ�ֲ���׼��Ѱ�� 5x5 �����ڵ���ͺ����
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

            // 2. �ؼ������㡰�ֲ��ֲڶȡ��������ü���ͱ�׼��ļ򻯰棩
            // ƽ̨��Ȼ�и߲������ 5x5 ��Χ��ͨ����б�棬��Χ����͹��
            bool isBump = (gridMap[r][c].avg_z > avgZ + 0.15f); // ��������ƽ��ֵ

            // 3. ������ˣ�Χ�����롰�¶���
            // ��� 5x5 ��Χ����ߺ���Ͳ��� 10 �ף��ǿ϶��Ǵ����
            bool notBigSlope = (maxZ - minZ < 3.0f);

            if (diffH >= minBermH && diffH <= maxBermH && isBump && notBigSlope) {
                tempMap[r][c].type = 4;
            }
        }
    }

    // 4. ����ɱ������ͨ����ˣ�����ƽ̨�ϵġ���ɫȸ�ߡ�
    // Χ������״�ģ�����һ��ģ�ƽ̨�ϵ���ͨ���ǹ����ļ�����
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (tempMap[r][c].type == 4) {
                int neighborBermCount = 0;
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        if (tempMap[r + i][c + j].type == 4) neighborBermCount++;
                    }
                }
                // �����Χû����ͬ�࣬˵���ǹ���������ɾ��
                if (neighborBermCount < 3) {
                    tempMap[r][c].type = gridMap[r][c].type;
                }
            }
        }
    }
    gridMap = tempMap;
}

// ── 围堰间隙填补：5次迭代，三规则(夹层/桥接/孤立红)将边坡格转为围堰 ──
void PointCloudClassification::fillBermGaps(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = (int)gridMap.size();
    int cols = (int)gridMap[0].size();

    // ���� 5 �Σ���ˢ��һ���������ǣ�ֱ����϶����
    for (int iter = 0; iter < 5; ++iter) {
        auto tempMap = gridMap;
        for (int r = 2; r < rows - 2; ++r) {
            for (int c = 2; c < cols - 2; ++c) {
                if (gridMap[r][c].type == 2) {
                    auto isY = [&](int dr, int dc) { return gridMap[r + dr][c + dc].type == 4; };
                    bool hasYellow5x5 = false, hasGreen5x5 = false;
                    int notYellow3x3 = 0;

                    for (int i = -2; i <= 2; ++i) {
                        for (int j = -2; j <= 2; ++j) {
                            int t = gridMap[r + i][c + j].type;
                            if (t == 4) hasYellow5x5 = true;
                            if (t == 1) hasGreen5x5 = true;
                            if (i >= -1 && i <= 1 && j >= -1 && j <= 1 && !(i == 0 && j == 0))
                                if (t != 4 && !gridMap[r + i][c + j].is_empty) notYellow3x3++;
                        }
                    }

                    bool isSandwich = (hasYellow5x5 && hasGreen5x5);
                    bool isBridge = (isY(-1,0) && isY(1,0))
                                 || (isY(0,-1) && isY(0,1))
                                 || (isY(-1,-1) && isY(1,1))
                                 || (isY(-1,1) && isY(1,-1));
                    bool isLoneRed = (notYellow3x3 == 0);

                    if (isSandwich || isBridge || isLoneRed)
                        tempMap[r][c].type = 4;
                }
            }
        }
        // �����һ�ε���û���κα仯��������ǰ�˳�����ʡ 5700X3D �Ĺ���
        gridMap = tempMap;
    }
}

// ── 围堰去噪：围堰格3x3邻域中平台数量≥5时转平台 ──
void PointCloudClassification::removeIsolatedBerms(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = (int)gridMap.size(), cols = (int)gridMap[0].size();
    auto tempMap = gridMap;
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (gridMap[r][c].type != 4) continue;
            int gC = 0;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j)
                    if (gridMap[r + i][c + j].type == 1) gC++;
            if (gC >= 5) tempMap[r][c].type = 1;
        }
    }
    gridMap = tempMap;
}

// ── 双向扫描提取坡顶/坡底：追踪连续type2边坡区块，在区块首尾标记crest/toe ──
void PointCloudClassification::extractSlopeFeatures(std::vector<std::vector<GridCell>>& gridMap, std::vector<SlopeParams>& results) {
    int rows = (int)gridMap.size();
    int cols = (int)gridMap[0].size();

    for (auto& row : gridMap)
        for (auto& cell : row) { cell.is_crest = cell.is_toe = false; }

    auto scan = [&](int endIdx, int outerIdx, bool isCol) {
        auto rc = [&](int idx) { return isCol ? std::make_pair(idx, outerIdx) : std::make_pair(outerIdx, idx); };
        bool searching = false;
        int firstIdx = -1;

        for (int i = 1; i < endIdx - 1; ++i) {
            auto [r,  c]  = rc(i);
            auto [pr, pc] = rc(i - 1);
            if (gridMap[r][c].is_empty) continue;

            // 起点：当前是边坡 且 前一格不是边坡
            if (!searching) {
                if (gridMap[r][c].type == 2 && gridMap[pr][pc].type != 2) {
                    firstIdx = i;
                    searching = true;
                }
            }
            // 终点：当前是边坡 且 下一格不是边坡 → 立即结算
            else {
                auto [nr, nc] = rc(i + 1);
                if (gridMap[r][c].type == 2 && gridMap[nr][nc].type != 2) {
                    int lastIdx = i;

                    auto [r1, c1] = rc(firstIdx);
                    auto [r2, c2] = rc(lastIdx);
                    float z1 = gridMap[r1][c1].avg_z;
                    float z2 = gridMap[r2][c2].avg_z;

                    float height = std::abs(z1 - z2);
                    float hDist = std::abs(firstIdx - lastIdx) * gridSize;

                    if (height > 0.8f && hDist >= 1.0f) {
                        if (z1 > z2) {
                            gridMap[r1][c1].is_crest = true;
                            gridMap[r2][c2].is_toe   = true;
                        } else {
                            gridMap[r2][c2].is_crest = true;
                            gridMap[r1][c1].is_toe   = true;
                        }
                        SlopeParams p;
                        p.benchHeight = height;
                        p.benchAngle = std::atan2(height, hDist) * 180.0f / (float)M_PI;
                        results.push_back(p);
                    }
                    searching = false;
                }
            }
        }
    };

    std::cout << "正在执行双向特征提取..." << std::endl;
    for (int c = 0; c < cols; ++c) scan(rows, c, true);
    for (int r = 0; r < rows; ++r) scan(cols, r, false);
}

// ── 网格骨架细化：原地顺序剔除冗余内角格，保持连通性 ──
void PointCloudClassification::thinFeatureLines(std::vector<std::vector<GridCell>>& gridMap) {
    int rows = (int)gridMap.size();
    int cols = (int)gridMap[0].size();

    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (gridMap[r][c].is_crest) {
                bool N = gridMap[r - 1][c].is_crest;
                bool S = gridMap[r + 1][c].is_crest;
                bool E = gridMap[r][c + 1].is_crest;
                bool W = gridMap[r][c - 1].is_crest;
                if ((N && E) || (N && W) || (S && E) || (S && W))
                    gridMap[r][c].is_crest = false;
            }
            if (gridMap[r][c].is_toe) {
                bool N = gridMap[r - 1][c].is_toe;
                bool S = gridMap[r + 1][c].is_toe;
                bool E = gridMap[r][c + 1].is_toe;
                bool W = gridMap[r][c - 1].is_toe;
                if ((N && E) || (N && W) || (S && E) || (S && W))
                    gridMap[r][c].is_toe = false;
            }
        }
    }
}

// 在 FeatureLine.cpp 中添加
void PointCloudClassification::smoothFeatureLines(std::vector<FeatureLine>& lines, int iterations) {
    for (auto& line : lines) {
        // 点数太少没法平滑
        if (line.points.size() < 3) continue;

        std::vector<Eigen::Vector3f> temp = line.points;
        for (int iter = 0; iter < iterations; ++iter) {
            // 两端点固定，只平滑中间的点
            for (size_t i = 1; i < line.points.size() - 1; ++i) {
                // 当前点取自身权重的 50%，前后相邻点各取 25%
                temp[i] = 0.5f * line.points[i] +
                    0.25f * line.points[i - 1] +
                    0.25f * line.points[i + 1];
            }
            line.points = temp;
        }
    }
}