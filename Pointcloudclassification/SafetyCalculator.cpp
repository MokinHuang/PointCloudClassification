#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iomanip>

// ==========================================
//    设计理想值（待填写）
// ==========================================
const float DESIGN_BENCH_HEIGHT   = 0.0f;   // 设计台阶高度 (m)
const float DESIGN_BENCH_ANGLE    = 0.0f;   // 设计坡面角 (度)
const float DESIGN_PLATFORM_WIDTH = 0.0f;   // 设计平台宽度 (m)
const float DESIGN_OVERALL_ANGLE  = 0.0f;   // 设计综合坡面角 (度)

// ==========================================
//    权重值（待填写，四项之和建议为 1.0）
// ==========================================
const float W_BENCH_HEIGHT   = 0.0f;  // 台阶高度权重
const float W_BENCH_ANGLE    = 0.0f;  // 坡面角权重
const float W_PLATFORM_WIDTH = 0.0f;  // 平台宽度权重
const float W_OVERALL_ANGLE  = 0.0f;  // 综合坡面角权重

// ==========================================
//    数据结构
// ==========================================
struct Vec3 { float x, y, z; };

struct Polyline {
    std::vector<Vec3> pts;
    int type;  // 0=crest, 1=toe
    int id;
};

struct BenchSection {
    float benchHeight;       // 台阶高度 (m)
    float benchAngle;        // 台阶坡面角 (度)
    float crestToeHorizDist; // 坡顶→坡底水平距离 (m)
    float platformWidth;     // 安全平台宽度 = 本坡底→下一坡顶 (m)
    Vec3  crestCenter;
    Vec3  toeCenter;
    int   crestId;
    int   toeId;
};

// ==========================================
//    CSV 读取
// ==========================================
std::vector<Polyline> readCSV(const std::string& filename) {
    std::vector<Polyline> result;
    std::ifstream f(filename);
    if (!f) {
        std::cerr << "[错误] 无法打开: " << filename << std::endl;
        return result;
    }

    std::string line;
    std::getline(f, line);  // 跳过标题

    Polyline cur;
    int lastId = -1;

    while (std::getline(f, line)) {
        if (line.empty()) {
            if (!cur.pts.empty()) result.push_back(cur);
            cur = Polyline();
            continue;
        }

        // line_id, type, type_name, point_order, x, y, z
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);
        if (cols.size() < 7) continue;

        int   id  = std::stoi(cols[0]);
        int   tp  = std::stoi(cols[1]);
        float x   = std::stof(cols[4]);
        float y   = std::stof(cols[5]);
        float z   = std::stof(cols[6]);

        if (id != lastId) {
            if (!cur.pts.empty()) result.push_back(cur);
            cur = Polyline();
            cur.id   = id;
            cur.type = tp;
            lastId    = id;
        }
        cur.pts.push_back({x, y, z});
    }
    if (!cur.pts.empty()) result.push_back(cur);
    return result;
}

// ==========================================
//    几何工具
// ==========================================
static Vec3 centroid(const std::vector<Vec3>& pts) {
    Vec3 c{0,0,0};
    if (pts.empty()) return c;
    for (auto& p : pts) { c.x += p.x; c.y += p.y; c.z += p.z; }
    c.x /= pts.size(); c.y /= pts.size(); c.z /= pts.size();
    return c;
}

static float avgZ(const std::vector<Vec3>& pts) {
    if (pts.empty()) return 0;
    float s = 0;
    for (auto& p : pts) s += p.z;
    return s / pts.size();
}

static float xyDist(const Vec3& a, const Vec3& b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// 点到折线的最近水平距离
static float distToPolylineXY(const Vec3& pt, const std::vector<Vec3>& poly) {
    float best = std::numeric_limits<float>::max();
    for (size_t i = 0; i + 1 < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[i+1];
        float dx = b.x - a.x, dy = b.y - a.y;
        float len2 = dx*dx + dy*dy;
        if (len2 < 1e-9f) { best = std::min(best, xyDist(pt, a)); continue; }
        float t = ((pt.x - a.x)*dx + (pt.y - a.y)*dy) / len2;
        t = std::max(0.0f, std::min(1.0f, t));
        Vec3 proj{a.x + t*dx, a.y + t*dy, 0};
        best = std::min(best, xyDist(pt, proj));
    }
    return best;
}

// ==========================================
//    坡顶-坡底配对
// ==========================================
static std::vector<BenchSection> pairCrestToe(
    const std::vector<Polyline>& crests,
    const std::vector<Polyline>& toes)
{
    std::vector<BenchSection> sections;

    struct Info { Vec3 c; float z; int idx; };
    std::vector<Info> ci, ti;
    for (size_t i = 0; i < crests.size(); ++i)
        ci.push_back({centroid(crests[i].pts), avgZ(crests[i].pts), (int)i});
    for (size_t i = 0; i < toes.size(); ++i)
        ti.push_back({centroid(toes[i].pts), avgZ(toes[i].pts), (int)i});

    // 按高程降序
    std::sort(ci.begin(), ci.end(), [](auto& a, auto& b){return a.z > b.z;});
    std::sort(ti.begin(), ti.end(), [](auto& a, auto& b){return a.z > b.z;});

    std::vector<bool> toeUsed(toes.size(), false);

    for (auto& cr : ci) {
        int    best = -1;
        float  bestDist = std::numeric_limits<float>::max();

        for (size_t j = 0; j < ti.size(); ++j) {
            if (toeUsed[j]) continue;
            if (ti[j].z >= cr.z) continue;  // 坡底须低于坡顶

            float d   = xyDist(cr.c, ti[j].c);
            float dh  = cr.z - ti[j].z;

            // 坡面角合理性检查
            if (d > 0.01f) {
                float ang = std::atan2(dh, d) * 180.0f / 3.14159f;
                if (ang < 10.0f || ang > 80.0f) continue;
            }

            if (d < bestDist) { bestDist = d; best = (int)j; }
        }

        if (best >= 0) {
            toeUsed[best] = true;
            auto& cl = crests[cr.idx];
            auto& tl = toes[ti[best].idx];

            BenchSection s;
            s.crestCenter = cr.c;
            s.toeCenter   = ti[best].c;
            s.crestId     = cl.id;
            s.toeId       = tl.id;
            s.benchHeight = cr.z - ti[best].z;
            s.crestToeHorizDist = distToPolylineXY(s.crestCenter, tl.pts);
            s.benchAngle  = (s.crestToeHorizDist > 0.01f)
                ? std::atan2(s.benchHeight, s.crestToeHorizDist) * 180.0f / 3.14159f
                : 90.0f;
            s.platformWidth = 0;  // 下一步填充

            sections.push_back(s);
        }
    }
    return sections;
}

// ==========================================
//    平台宽度: 上一个坡底 → 下一个坡顶
// ==========================================
static void calcPlatformWidths(std::vector<BenchSection>& secs) {
    std::sort(secs.begin(), secs.end(),
        [](auto& a, auto& b){return a.crestCenter.z > b.crestCenter.z;});

    for (size_t i = 0; i + 1 < secs.size(); ++i) {
        secs[i].platformWidth = xyDist(secs[i].toeCenter, secs[i+1].crestCenter);
    }
    // 最底层: 取已有平台宽度的平均值
    if (secs.size() >= 2) {
        float sum = 0; int cnt = 0;
        for (size_t i = 0; i + 1 < secs.size(); ++i) { sum += secs[i].platformWidth; cnt++; }
        secs.back().platformWidth = sum / cnt;
    }
}

// ==========================================
//    综合坡面角: 最高坡顶 → 最低坡底
// ==========================================
static float calcOverallAngle(const std::vector<Polyline>& crests,
                              const std::vector<Polyline>& toes,
                              float& outH, float& outD) {
    float topZ = -std::numeric_limits<float>::max();
    float botZ =  std::numeric_limits<float>::max();
    Vec3 topPt{}, botPt{};

    for (auto& c : crests) for (auto& p : c.pts)
        if (p.z > topZ) { topZ = p.z; topPt = p; }
    for (auto& t : toes) for (auto& p : t.pts)
        if (p.z < botZ) { botZ = p.z; botPt = p; }

    outH = topZ - botZ;
    outD = xyDist(topPt, botPt);
    return (outD > 0.01f)
        ? std::atan2(outH, outD) * 180.0f / 3.14159f
        : 90.0f;
}

// ==========================================
//    评分函数
// ==========================================
static float score(float actual, float design, const std::string& type) {
    if (design <= 0.0f) return 0.0f;

    if (type == "height") {
        float r = actual / design;
        if (r >= 2.0f) return 4.0f;
        if (r >= 1.6f) return 2.0f;
        if (r >= 1.3f) return 1.0f;
        if (r >= 1.1f) return 0.5f;
        return 0.0f;
    }
    else if (type == "width") {
        float r = actual / design;
        if (r <= 0.5f) return 4.0f;
        if (r <= 0.75f) return 2.0f;
        if (r < 1.0f)  return 0.5f;
        return 0.0f;
    }
    else if (type == "angle") {
        float d = actual - design;
        if (d >= 15.0f) return 4.0f;
        if (d >= 10.0f) return 2.0f;
        if (d >= 5.0f)  return 0.5f;
        return 0.0f;
    }
    return 0.0f;
}

// ==========================================
//    主函数
// ==========================================
int main(int argc, char* argv[]) {
    std::string csvFile = "feature_lines.csv";
    if (argc > 1) csvFile = argv[1];

    std::cout << std::fixed << std::setprecision(2);

    // ---- 读取 ----
    auto all = readCSV(csvFile);
    std::vector<Polyline> crests, toes;
    for (auto& l : all) {
        if (l.type == 0) crests.push_back(l);
        else             toes.push_back(l);
    }
    std::cout << "\n[读取] 坡顶线 " << crests.size()
              << " 条, 坡底线 " << toes.size() << " 条\n";

    if (crests.empty() || toes.empty()) {
        std::cerr << "[错误] 特征线数据不足以计算参数" << std::endl;
        return -1;
    }

    // ---- 配对 ----
    auto sections = pairCrestToe(crests, toes);
    std::cout << "[配对] 成功 " << sections.size() << " 组\n";
    if (sections.empty()) return -1;

    // ---- 平台宽度 ----
    calcPlatformWidths(sections);

    // ---- 综合坡面角 ----
    float totalH = 0, totalD = 0;
    float overallAngle = calcOverallAngle(crests, toes, totalH, totalD);

    // ---- 平台宽度统计 ----
    float maxW = -std::numeric_limits<float>::max();
    float minW =  std::numeric_limits<float>::max();
    float sumW = 0;
    int   cntW = 0;
    for (auto& s : sections) {
        if (s.platformWidth > 0) {
            maxW = std::max(maxW, s.platformWidth);
            minW = std::min(minW, s.platformWidth);
            sumW += s.platformWidth; cntW++;
        }
    }
    float avgW = cntW > 0 ? sumW / cntW : 0;

    // ---- SMSI ----
    float smsiSum = 0;
    int   smsiCnt = 0;
    for (auto& s : sections) {
        float sH  = score(s.benchHeight,  DESIGN_BENCH_HEIGHT,   "height");
        float sA  = score(s.benchAngle,   DESIGN_BENCH_ANGLE,    "angle");
        float sW  = score(s.platformWidth, DESIGN_PLATFORM_WIDTH, "width");
        smsiSum  += sH * W_BENCH_HEIGHT + sA * W_BENCH_ANGLE + sW * W_PLATFORM_WIDTH;
        smsiCnt++;
    }
    float sOA  = score(overallAngle, DESIGN_OVERALL_ANGLE, "angle");
    smsiSum   += sOA * W_OVERALL_ANGLE;
    float asmsi = smsiCnt > 0 ? smsiSum / smsiCnt : 0;

    // ==========================================
    //    输出报告
    // ==========================================
    std::cout << "\n┌─────────────────────────────────────────┐\n";
    std::cout <<   "│     基于特征线的边坡安全参数计算报告    │\n";
    std::cout <<   "└─────────────────────────────────────────┘\n";

    // 一、设计值
    std::cout << "\n一、设计理想值\n";
    std::cout << "  设计台阶高度:    " << std::setw(8) << DESIGN_BENCH_HEIGHT   << " m\n";
    std::cout << "  设计坡面角:      " << std::setw(8) << DESIGN_BENCH_ANGLE    << " °\n";
    std::cout << "  设计平台宽度:    " << std::setw(8) << DESIGN_PLATFORM_WIDTH << " m\n";
    std::cout << "  设计综合坡面角:  " << std::setw(8) << DESIGN_OVERALL_ANGLE  << " °\n";
    std::cout << "  权重: 高度=" << W_BENCH_HEIGHT
              << "  坡面角=" << W_BENCH_ANGLE
              << "  宽度=" << W_PLATFORM_WIDTH
              << "  综合角=" << W_OVERALL_ANGLE << "\n";

    // 二、逐台阶
    std::cout << "\n二、各台阶实测参数\n";
    std::cout << "  " << std::left
              << std::setw(6)  << "台阶"
              << std::setw(12) << "高度(m)"
              << std::setw(14) << "坡面角(°)"
              << std::setw(16) << "坡顶坡底距(m)"
              << std::setw(14) << "平台宽(m)"
              << "\n  " << std::string(62, '-') << "\n";

    for (size_t i = 0; i < sections.size(); ++i) {
        auto& s = sections[i];
        std::cout << "  " << std::setw(6)  << (i + 1)
                  << std::setw(12) << s.benchHeight
                  << std::setw(14) << s.benchAngle
                  << std::setw(16) << s.crestToeHorizDist
                  << std::setw(14) << s.platformWidth << "\n";
    }

    // 三、综合
    std::cout << "\n三、综合参数\n";
    std::cout << "  综合总高度:      " << totalH        << " m\n";
    std::cout << "  综合水平距离:    " << totalD        << " m\n";
    std::cout << "  综合坡面角:      " << overallAngle  << " °\n";
    std::cout << "  最大平台宽度:    " << maxW          << " m\n";
    std::cout << "  最小平台宽度:    " << minW          << " m\n";
    std::cout << "  平均平台宽度:    " << avgW          << " m\n";

    // 四、安全评价
    std::cout << "\n四、安全评价\n";
    std::cout << "  SMSI 总分:       " << smsiSum << "\n";
    std::cout << "  平均 SMSI:       " << asmsi   << "\n";
    std::cout << "  安全等级:        ";

    if (asmsi <= 0.5f) {
        std::cout << "[ 安全 ] - 状态稳定\n";
    } else if (asmsi <= 1.0f) {
        std::cout << "[ 注意 ] - 轻微扰动\n";
    } else {
        std::cout << "[ 危险 ] - 临界失稳风险\n";
    }

    std::cout << "\n==========================================\n\n";
    return 0;
}
