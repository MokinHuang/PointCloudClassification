#include <iostream>
#include <vector>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include "PointCloudClassification.h"
#include "FeatureLine.h"

int main() {
    // --- 1. ��ʼ�������� ---
    std::string filename = "test2.pcd";
    float gridSize = 1.0f; // �����ֱ��� 1.0��
    PointCloudClassification processor(gridSize);

    std::cout << "���ڼ��ص����ļ�: " << filename << "..." << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        std::cerr << "�޷���ȡ�ļ�������·����" << std::endl;
        return -1;
    }

    // --- 2. ����������ͼ ---
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);

    int cols = static_cast<int>((max_pt[0] - min_pt[0]) / gridSize) + 1;
    int rows = static_cast<int>((max_pt[1] - min_pt[1]) / gridSize) + 1;

    // ��ʼ����������
    std::vector<std::vector<GridCell>> gridMap(rows, std::vector<GridCell>(cols));

    // ������ݵ㲢����ƽ���߶�
    for (const auto& pt : cloud->points) {
        int c = static_cast<int>((pt.x - min_pt[0]) / gridSize);
        int r = static_cast<int>((pt.y - min_pt[1]) / gridSize);
        if (r >= 0 && r < rows && c >= 0 && c < cols) {
            gridMap[r][c].sum_z += pt.z;
            gridMap[r][c].count++;
        }
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (gridMap[r][c].count > 0) {
                gridMap[r][c].avg_z = gridMap[r][c].sum_z / gridMap[r][c].count;
                gridMap[r][c].is_empty = false;
            }
        }
    }

    // --- 3. ���������ˮ�� (�����㷨) ---
    std::cout << "����ִ����������㷨��ˮ��..." << std::endl;

    processor.fillEmptyGrids(gridMap, rows, cols);        // �ն����
    processor.classifyGrids(gridMap, gridSize);           // �������̷���

    for (int i = 0; i < 2; ++i) {
        processor.refineClassification(gridMap);         // ��ͨ���Ż�
    }

    processor.cleanPlatformNoise(gridMap);                // ����ƽ̨���
    processor.closeSlopeHoles(gridMap);                   // �պ�����׶�

    processor.identifyBerms(gridMap);                     // ʶ���ɫΧ��
    processor.fillBermGaps(gridMap);                      // 3x3/5x5 ����Χ����϶

    // --- 4. 提取坡顶/坡底线 ---
    std::cout << "正在提取坡顶(Crest)和坡底(Toe)特征线..." << std::endl;
    std::vector<SlopeParams> measuredParams;
    processor.extractSlopeFeatures(gridMap, measuredParams);

    std::vector<FeatureLine> featureLines =
        processor.extractFeatureLines(gridMap, gridSize, min_pt, 2, 3, 3.0f);

    // 输出特征线 CSV
    processor.saveFeatureLinesToTxt(featureLines, "feature_lines.csv");

    // --- 5. 输出分类 PCD ---
    std::string outputName = "result_final.pcd";
    processor.saveVisualizationCloud(gridMap, gridSize, min_pt, max_pt, outputName);

    std::cout << "\n处理完成。特征线已保存至 feature_lines.csv" << std::endl;
    return 0;
}