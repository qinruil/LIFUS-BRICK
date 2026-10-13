# -*- coding: utf-8 -*-
"""Insert figure references into the six report markdown files.

Each figure is placed right after the heading line of its matching results
section, so the rendered Word shows: section heading, centered figure,
centered one-line caption.

Mapping (figure -> anchor heading fragment):
  spectrum (CN)  : **2. 算子层 pre/post 谱差异 -> eigenvalue_spectrum
                   **3. pre/post 模态匹配 -> mode_matching
                   **4. 折间模态拓扑稳定性 -> fold_topography
                   **7. ZI 侧模态能量参与 -> modal_energy
  spectrum (EN)  : ### 2.2 The operator-level pre/post spectral -> eigenvalue
                   ### 2.3 Pre/post mode matching -> mode_matching
                   ### 2.4 Across-fold mode-topography -> fold_topography
                   ### 2.7 ZI-side shift -> modal_energy
  variance (CN)  : **1. 分频段后 K 的方差占比 -> percent_K_by_band
                   **2. 协方差归属规则 -> covariance_split
                   **3. 控制输入 u_t 的时间尺度 -> u_t_autocorrelation
                   **5. K 占比对训练规模不敏感 -> percent_K_vs_training
  variance (EN)  : ### 2.1 The band-resolved K share -> percent_K_by_band
                   ### 2.2 The sensitivity of the K share -> covariance_split
                   ### 2.3 The time scale of the control inputs -> u_t_autocorrelation
                   ### 2.5 The K share is insensitive to training size -> percent_K_vs_training
  topology (CN)  : **1. K 的网络核心 -> K_network_graph, K_betweenness_hub_ranking
                   **2. 阈值扫描显示 -> threshold_scan
                   **4. 滑窗拓扑连通分量 -> sliding_window
  topology (EN)  : ### 2.1 The network core of K -> K_network_graph, K_betweenness_hub_ranking
                   ### 2.2 Threshold scanning shows -> threshold_scan
                   ### 2.4 The sliding-window connected-component count -> sliding_window
"""
import io
import os
import re

OUT = r"D:\Nadege\LIFUS-BRICK-main\20261005_231508-bad40202"
FIG = r"D:/Nadege/LIFUS-BRICK-main/LIFUS-BRICK-main/results/figures"

FIG_LINE = "![{caption}]({path})"

# (file, [(anchor_fragment, [(img_rel, caption)])])
PLAN = [
    ("谱分析补充报告.md", [
        ("**2. 算子层 pre/post 谱差异",
         [("spectrum_stats/eigenvalue_spectrum_complex_plane.png",
           "图：参考 Koopman 算子特征值复平面分布，半径代表持续性、颜色映射 |Λ|，虚线圆为单位圆。")]),
        ("**3. pre/post 模态匹配",
         [("spectrum_stats/mode_matching_similarity_distribution.png",
           "图：pre/post 模态空间匹配的余弦相似度直方图，红色虚线为匹配均值、灰色虚线为随机基线 0.005。")]),
        ("**4. 折间模态拓扑稳定性",
         [("spectrum_stats/fold_topography_stability.png",
           "图：各 LOSO 折模态拓扑对全体折质心的平均相似度，橙色柱为重建发散折（09/15/19）。")]),
        ("**7. ZI 侧模态能量参与",
         [("spectrum_stats/modal_energy_magnitude_by_condition.png",
           "图：四条件下能量加权特征值幅值均值（误差棒为 SEM），浅色连线为同被试前后配对轨迹。")]),
    ]),
    ("Spectrum_Analysis_Supplementary_Report.md", [
        ("### 2.2 The operator-level pre/post spectral",
         [("spectrum_stats/eigenvalue_spectrum_complex_plane.png",
           "Figure: eigenvalue spectrum of the reference Koopman operator in the complex plane; radius is persistence, colour maps |Lambda|, dashed circle is the unit circle.")]),
        ("### 2.3 Pre/post mode matching",
         [("spectrum_stats/mode_matching_similarity_distribution.png",
           "Figure: distribution of cosine similarity for pre/post matched mode pairs; red dashed line is the matched mean, grey dashed line is the random baseline of 0.005.")]),
        ("### 2.4 Across-fold mode-topography stability",
         [("spectrum_stats/fold_topography_stability.png",
           "Figure: per-fold mean mode similarity to the across-fold centroid; orange bars mark the divergent folds (09, 15, 19).")]),
        ("### 2.7 ZI-side shift in modal energy participation",
         [("spectrum_stats/modal_energy_magnitude_by_condition.png",
           "Figure: energy-weighted eigenvalue magnitude by condition (error bars are SEM); light lines connect paired pre/post observations of the same subject.")]),
    ]),
    ("方差分解补充报告.md", [
        ("**1. 分频段后 K 的方差占比",
         [("variance_decomposition/percent_K_by_band_and_condition.png",
           "图：四条件乘四频段的 %K 分组柱状图，水平虚线为宽带基线 1.24%。")]),
        ("**2. 协方差归属规则",
         [("variance_decomposition/covariance_split_sensitivity.png",
           "图：%K 随协方差归属参数 θ 的敏感性曲线，三档取值下结论稳健。")]),
        ("**3. 控制输入 u_t 的时间尺度",
         [("variance_decomposition/u_t_autocorrelation_pre_post.png",
           "图：VIM 与 ZI 各被试控制输入滞后一阶自相关的前后散点，对角线为无变化参考。")]),
        ("**5. K 占比对训练规模不敏感",
         [("variance_decomposition/percent_K_vs_training_size.png",
           "图：各检查点 %K 对训练规模（13/17）的散点，按 LOSO、种子与 task-state 分族，异常发散折已标注。")]),
    ]),
    ("Variance_Decomposition_Supplementary_Report.md", [
        ("### 2.1 The band-resolved K share",
         [("variance_decomposition/percent_K_by_band_and_condition.png",
           "Figure: pooled %K by frequency band and condition; the dashed horizontal line marks the broadband baseline of 1.24%.")]),
        ("### 2.2 The sensitivity of the K share",
         [("variance_decomposition/covariance_split_sensitivity.png",
           "Figure: sensitivity of %K to the covariance-assignment parameter theta; the conclusion is stable across the three values.")]),
        ("### 2.3 The time scale of the control inputs",
         [("variance_decomposition/u_t_autocorrelation_pre_post.png",
           "Figure: per-subject median lag-1 autocorrelation of the control inputs, pre versus post, for VIM and ZI; the diagonal is the no-change reference.")]),
        ("### 2.5 The K share is insensitive to training size",
         [("variance_decomposition/percent_K_vs_training_size.png",
           "Figure: %K versus training-set size across checkpoints, coloured by family (LOSO, seed, task-state); divergent folds are annotated.")]),
    ]),
    ("图论拓扑补充报告.md", [
        ("**1. K 的网络核心",
         [("graph_topology/K_network_graph.png",
           "图：K 耦合网络可视化，节点大小对应介数中心性、颜色对应网络分组，枢纽区已标注。"),
          ("graph_topology/K_betweenness_hub_ranking.png",
           "图：K 网络各区域介数中心性排名条形图，颜色为网络分组。")]),
        ("**2. 阈值扫描显示",
         [("graph_topology/threshold_scan_density_assortativity.png",
           "图：K 与 ΔK 网络平均度及度关联性随保留连接比例的演化，未见结构相变。")]),
        ("**4. 滑窗拓扑连通分量",
         [("graph_topology/sliding_window_components_by_condition.png",
           "图：滑窗相关图平均连通分量数的条件箱线图，ZI 后连通性趋于下降。")]),
    ]),
    ("Graph_and_Topology_Supplementary_Report.md", [
        ("### 2.1 The network core of K",
         [("graph_topology/K_network_graph.png",
           "Figure: visualisation of the K coupling network; node size is betweenness centrality, node colour is network group, hubs are labelled."),
          ("graph_topology/K_betweenness_hub_ranking.png",
           "Figure: betweenness-centrality ranking of regions in the K network; colour indicates network group.")]),
        ("### 2.2 Threshold scanning shows",
         [("graph_topology/threshold_scan_density_assortativity.png",
           "Figure: mean degree and assortativity of the K and delta-K networks versus the retained connection fraction; no structural transition is present.")]),
        ("### 2.4 The sliding-window connected-component count",
         [("graph_topology/sliding_window_components_by_condition.png",
           "Figure: box plots of the mean connected-component count of the windowed correlation graphs by condition; connectivity decreases after ZI.")]),
    ]),
]


def main():
    for fname, anchors in PLAN:
        p = os.path.join(OUT, fname)
        t = io.open(p, encoding="utf-8").read()
        lines = t.split("\n")
        # build anchor -> insertion text
        for anchor, figs in anchors:
            # find the heading line
            idx = None
            for i, l in enumerate(lines):
                if l.strip().startswith(anchor):
                    idx = i
                    break
            if idx is None:
                print(f"MISSING anchor in {fname}: {anchor!r}")
                continue
            # build figure block lines
            block = []
            for rel, cap in figs:
                block.append(FIG_LINE.format(caption=cap, path=f"{FIG}/{rel}"))
            # insert right after the heading line
            lines[idx + 1:idx + 1] = [""] + block + [""]
        io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
        print("updated", fname)


if __name__ == "__main__":
    main()