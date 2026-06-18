# TOPSIS_Final.py (仅保留全局归一化 + 按100nm区间独立TOPSIS + 每区间选前2 → 最终12个波段)
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize

# 设置中文和负号显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 核心配置（完全贴合需求） =================
ORIGINAL_DATA_FILE = "土壤碳含量-LASSO20特征波段_分区间保留.xlsx"
TARGET_COLUMN = "有机碳" 

OUTPUT_DIR = "TOPSIS_Per_Interval_12Bands_Results"
FINAL_12BANDS_EXCEL = os.path.join(OUTPUT_DIR, "各区间前2名_12个波段汇总(仅全局归一化).xlsx")
INTERVAL_TOPSIS_EXCEL = os.path.join(OUTPUT_DIR, "各区间TOPSIS完整排序(仅全局归一化).xlsx")
ALL_BANDS_NORMALIZED_EXCEL = os.path.join(OUTPUT_DIR, "所有波段_原始+全局归一化指标汇总.xlsx")
FINAL_PLOT = os.path.join(OUTPUT_DIR, "12个波段_Topsis贴近度排序图.png")
RADAR_PLOT = os.path.join(OUTPUT_DIR, "各区间最优波段雷达图汇总.png")

# 字体设置
FONT_SIZE = 46
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 核心参数（关键！按100nm区间拆分）
INTERVAL_SIZE = 100              # 100nm一个区间
SELECT_PER_INTERVAL = 4          # 每个区间选前2名
# 定义所有需要分析的100nm区间（根据高光谱常见范围）
ANALYSIS_INTERVALS = {
    "400-500nm": (400, 500),
    "500-600nm": (500, 600),
    "600-700nm": (600, 700),
    "700-800nm": (700, 800),
    "800-900nm": (800, 900),
    "900-1000nm": (900, 1000)
}
# TOPSIS指标权重（可自定义，也可优化）
TOPSIS_WEIGHTS = {
    'SNR': 0.4,
    'MI': 0.2,
    'Entropy': 0.1,
    'RF_Importance': 0.15,
    'Anti_Redundancy': 0.15
}
METRICS_LIST = list(TOPSIS_WEIGHTS.keys())
# 仅保留全局归一化指标列名
GLOBAL_NORMALIZED_METRICS_LIST = [f"{metric}_全局归一化" for metric in METRICS_LIST]
# =======================================

# ================= 辅助函数 =================
def parse_wavelength_from_feature(feature_name):
    """从特征名解析波长（如"500nm"→500，"650"→650）"""
    if pd.isna(feature_name):
        return np.nan
    s = str(feature_name).replace('nm', '').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

def get_band_interval(wavelength):
    """判断波段所属的100nm区间（返回区间名称，如650nm→"600-700nm"）"""
    if np.isnan(wavelength):
        return "未知区间"
    for interval_name, (min_wl, max_wl) in ANALYSIS_INTERVALS.items():
        if min_wl <= wavelength < max_wl:
            return interval_name
    return "未知区间"

def safe_minmax_norm(df):
    """安全的min-max归一化（避免除以0）"""
    out = df.copy().astype(float)
    for col in out.columns:
        mn = out[col].min(skipna=True)
        mx = out[col].max(skipna=True)
        if np.isclose(mx, mn):
            out[col] = 0.0
        else:
            out[col] = (out[col] - mn) / (mx - mn)
    return out

def calculate_topsis_per_interval(metrics_df, weights):
    """
    按100nm区间独立计算TOPSIS（使用全局归一化指标）
    返回：
        1. 各区间TOPSIS完整结果（含原始+全局归一化指标）
        2. 各区间前2名汇总（12个波段）- 保留原始索引（波段名称）
    """
    interval_topsis_results = {}  # 存储每个区间的TOPSIS结果
    interval_top2_results = []    # 存储每个区间前2名
    
    print("\n=== 按100nm区间独立计算TOPSIS ===")
    for interval_name, (min_wl, max_wl) in ANALYSIS_INTERVALS.items():
        # 筛选该区间的波段
        interval_bands = metrics_df[
            (metrics_df['Wavelength'] >= min_wl) & 
            (metrics_df['Wavelength'] < max_wl)
        ].copy()
        
        if len(interval_bands) == 0:
            print(f"⚠️ {interval_name}：无有效波段，跳过")
            continue
        
        print(f"\n📌 {interval_name}：共{len(interval_bands)}个波段")
        
        # 1. 提取该区间的全局归一化指标矩阵（不再计算区间内归一化）
        interval_metrics_norm = interval_bands[GLOBAL_NORMALIZED_METRICS_LIST].copy()
        
        # 2. TOPSIS计算（使用全局归一化指标）
        # 向量归一化
        norm_matrix = interval_metrics_norm / np.sqrt((interval_metrics_norm**2).sum(axis=0))
        # 加权归一化矩阵
        weighted_matrix = norm_matrix.copy()
        for col in weighted_matrix.columns:
            original_col = col.replace("_全局归一化", "")
            weighted_matrix[col] *= weights[original_col]
        # 正负理想解
        ideal_best = weighted_matrix.max()
        ideal_worst = weighted_matrix.min()
        # 距离计算
        dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
        # 贴近度
        closeness = dist_worst / (dist_best + dist_worst + 1e-12)
        
        # 3. 合并原始指标+全局归一化指标+TOPSIS结果
        interval_bands['TOPSIS贴近度'] = closeness
        interval_bands['区间内排名'] = interval_bands['TOPSIS贴近度'].rank(ascending=False, method='min')
        interval_bands['所属区间'] = interval_name
        
        # 保存该区间完整结果
        interval_topsis_results[interval_name] = interval_bands
        
        # 选取该区间前2名（保留原始索引）
        interval_top2 = interval_bands.sort_values('TOPSIS贴近度', ascending=False).head(SELECT_PER_INTERVAL)
        interval_top2_results.append(interval_top2)
        
        # 打印该区间前2名（含全局归一化指标）
        print(f"✅ {interval_name} 前{SELECT_PER_INTERVAL}名：")
        print(interval_top2[['Wavelength', 'TOPSIS贴近度', '区间内排名'] + GLOBAL_NORMALIZED_METRICS_LIST].to_string(index=True))
    
    # 汇总所有区间的前2名（最终12个波段）- 保留原始索引
    final_12bands = pd.concat(interval_top2_results)
    # 重置索引为列，避免后续拼接混乱，同时保留原始波段名称
    final_12bands = final_12bands.reset_index().rename(columns={'index': '波段名称'})
    # 全局排序
    final_12bands['全局排名'] = final_12bands['TOPSIS贴近度'].rank(ascending=False, method='min')
    final_12bands = final_12bands.sort_values('全局排名')
    
    return interval_topsis_results, final_12bands

def create_radar_plot_for_best_per_interval(final_12bands, metrics_norm, save_path):
    """绘制每个区间最优波段（前1名）的雷达图汇总 - 修复索引匹配问题"""
    # 筛选每个区间的第1名
    best_per_interval = final_12bands[final_12bands['区间内排名'] == 1].copy()
    
    # 准备雷达图数据
    categories = METRICS_LIST
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图
    
    # 创建子图（2行3列，对应6个区间）
    fig, axes = plt.subplots(2, 3, figsize=(24, 16), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    interval_index = 0
    
    for interval_name in ANALYSIS_INTERVALS.keys():
        ax = axes[interval_index]
        interval_index += 1
        
        # 获取该区间最优波段
        interval_best = best_per_interval[best_per_interval['所属区间'] == interval_name]
        if len(interval_best) == 0:
            ax.set_title(f"{interval_name}\n无有效波段", fontsize=FONT_SIZE-10)
            ax.axis('off')
            continue
        
        # 获取原始波段名称（关键修复点）
        band_name = interval_best['波段名称'].iloc[0]
        wl = interval_best['Wavelength'].iloc[0]
        
        # 校验波段名称是否存在（容错处理）
        if band_name not in metrics_norm.index:
            print(f"⚠️ 波段{band_name}不在指标矩阵中，跳过雷达图绘制")
            ax.set_title(f"{interval_name}\n波段{wl}nm\n无指标数据", fontsize=FONT_SIZE-10)
            ax.axis('off')
            continue
        
        # 获取全局归一化指标值（修复索引匹配）
        metrics_vals = metrics_norm.loc[band_name, categories].values.tolist()
        metrics_vals += metrics_vals[:1]  # 闭合
        
        # 绘制雷达图
        ax.plot(angles, metrics_vals, linewidth=3, linestyle='solid', 
                color=color_palette[interval_index-1], marker='o', markersize=8)
        ax.fill(angles, metrics_vals, color=color_palette[interval_index-1], alpha=0.3)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=FONT_SIZE-12)
        ax.set_yticklabels([])
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{interval_name}\n最优波段：{wl}nm", fontsize=FONT_SIZE-8, pad=20)
    
    plt.suptitle("各100nm区间最优波段指标雷达图", fontsize=FONT_SIZE, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_12bands(final_12bands, save_path):
    """绘制最终12个波段的TOPSIS贴近度排序图"""
    plt.figure(figsize=(30, 12))
    
    # 按区间着色
    interval_colors = {
        "400-500nm": "#FF6B6B",
        "500-600nm": "#4ECDC4",
        "600-700nm": "#45B7D1",
        "700-800nm": "#96CEB4",
        "800-900nm": "#FECA57",
        "900-1000nm": "#FF9FF3"
    }
    final_12bands['颜色'] = final_12bands['所属区间'].map(interval_colors)
    
    # 绘制柱状图
    bars = plt.bar(
        x=final_12bands['Wavelength'].astype(str) + 'nm',
        height=final_12bands['TOPSIS贴近度'],
        color=final_12bands['颜色'],
        edgecolor='black',
        linewidth=2
    )
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE-10)
    
    # 图表样式
    plt.xlabel('光谱波段 (nm)', fontsize=FONT_SIZE)
    plt.ylabel('TOPSIS 贴近度', fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE-8, rotation=45)
    plt.yticks(fontsize=FONT_SIZE-8)
    plt.title('各100nm区间前2名（共12个）波段TOPSIS贴近度排序', fontsize=FONT_SIZE+2)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 添加区间图例
    handles = [plt.Rectangle((0,0),1,1, color=color, label=interval) 
               for interval, color in interval_colors.items()]
    plt.legend(handles=handles, fontsize=FONT_SIZE-10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ================= 主流程 =================
def run_pipeline():
    # 1. 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出文件夹: {OUTPUT_DIR}")

    # 2. 读取数据
    try:
        df_all = pd.read_excel(ORIGINAL_DATA_FILE, header=0)
        print(f"\n✅ 读取数据成功：{df_all.shape}")
        print(f"目标列：{TARGET_COLUMN}")
    except FileNotFoundError:
        print(f"❌ 错误：文件未找到 {ORIGINAL_DATA_FILE}")
        return
    except Exception as e:
        print(f"❌ 读取数据失败：{e}")
        return

    # 3. 数据预处理
    # 提取特征列（仅保留数字型波段列）
    df_all.columns = [str(col) for col in df_all.columns]
    target_col = TARGET_COLUMN if TARGET_COLUMN in df_all.columns else "土壤碳含量"
    
    # 筛选特征列（数字型，排除目标列/非波段列）
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    feature_cols = [col for col in numeric_cols if col.replace('.', '', 1).isdigit()]
    
    if len(feature_cols) == 0:
        print("❌ 未找到有效波段特征列")
        return
    
    X = df_all[feature_cols].copy().fillna(df_all[feature_cols].mean())
    y = df_all[target_col].copy().fillna(df_all[target_col].mean())
    print(f"✅ 特征预处理完成：{len(feature_cols)}个波段，{len(X)}个样本")

    # 4. 计算各波段的TOPSIS指标（原始值）
    print("\n=== 计算TOPSIS指标（原始值）===")
    # 4.1 信噪比（SNR）：均值/标准差
    snr = X.mean() / (X.std() + 1e-12)
    # 4.2 互信息（MI）：衡量与目标的非线性相关性
    mi_scores = mutual_info_regression(X.values, y, random_state=42)
    mi = pd.Series(mi_scores, index=X.columns)
    # 4.3 熵值：衡量波段信息丰富度
    entropy = {}
    for col in X.columns:
        vals = X[col].dropna()
        if len(vals) < 2:
            entropy[col] = 0.0
            continue
        hist, bins = np.histogram(vals, bins='auto', density=True)
        hist = hist * np.diff(bins)
        hist = hist[hist > 1e-12]
        entropy[col] = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0.0
    entropy = pd.Series(entropy, index=X.columns)
    # 4.4 随机森林重要性
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    # 4.5 反冗余度（1 - 平均相关系数）
    corr_matrix = X.corr(method='pearson').abs()
    np.fill_diagonal(corr_matrix.values, np.nan)
    redundancy = corr_matrix.mean(axis=1)
    anti_redundancy = 1 - redundancy

    # 5. 组装指标表（保留原始波段名称作为索引）
    metrics_df = pd.DataFrame({
        'SNR': snr,
        'MI': mi,
        'Entropy': entropy,
        'RF_Importance': rf_importance,
        'Anti_Redundancy': anti_redundancy
    })
    # 添加波长和区间信息
    metrics_df['Wavelength'] = metrics_df.index.map(parse_wavelength_from_feature)
    metrics_df['所属区间'] = metrics_df['Wavelength'].apply(get_band_interval)
    metrics_df = metrics_df.dropna(subset=['Wavelength'])
    
    # 仅计算全局归一化指标（删除区间内归一化）
    metrics_norm_global = safe_minmax_norm(metrics_df[METRICS_LIST])
    metrics_norm_global.columns = GLOBAL_NORMALIZED_METRICS_LIST
    # 合并原始指标+全局归一化指标
    metrics_df = pd.concat([metrics_df, metrics_norm_global], axis=1)
    
    print(f"✅ 指标计算完成：{len(metrics_df)}个有效波段")
    print(f"📋 指标列：原始指标({METRICS_LIST}) + 全局归一化指标({GLOBAL_NORMALIZED_METRICS_LIST})")

    # 6. 保存所有波段的原始+全局归一化指标
    metrics_df_export = metrics_df.reset_index().rename(columns={'index': '波段名称'})
    metrics_df_export.to_excel(ALL_BANDS_NORMALIZED_EXCEL, index=False)
    print(f"\n✅ 所有波段原始+全局归一化指标已保存：{ALL_BANDS_NORMALIZED_EXCEL}")

    # 7. 按区间计算TOPSIS + 选前2名（使用全局归一化指标）
    interval_topsis, final_12bands = calculate_topsis_per_interval(metrics_df, TOPSIS_WEIGHTS)

    # 8. 保存结果
    # 8.1 保存各区间完整TOPSIS结果（含原始+全局归一化指标）
    with pd.ExcelWriter(INTERVAL_TOPSIS_EXCEL, engine='openpyxl') as writer:
        for interval_name, interval_data in interval_topsis.items():
            # 重置索引为列，方便查看
            interval_data_export = interval_data.reset_index().rename(columns={'index': '波段名称'})
            interval_data_export.to_excel(writer, sheet_name=interval_name, index=False)
    print(f"\n✅ 各区间TOPSIS结果（含全局归一化指标）已保存：{INTERVAL_TOPSIS_EXCEL}")

    # 8.2 保存最终12个波段结果（含原始+全局归一化指标）
    final_12bands_output = final_12bands[['波段名称', '所属区间', 'Wavelength', 'TOPSIS贴近度', 
                                          '区间内排名', '全局排名'] + METRICS_LIST + GLOBAL_NORMALIZED_METRICS_LIST]
    final_12bands_output.to_excel(FINAL_12BANDS_EXCEL, index=False)
    print(f"✅ 12个波段汇总结果（含全局归一化指标）已保存：{FINAL_12BANDS_EXCEL}")

    # 9. 绘制图表
    print("\n=== 绘制可视化图表 ===")
    # 9.1 12个波段排序图
    plot_final_12bands(final_12bands, FINAL_PLOT)
    # 9.2 各区间最优波段雷达图（使用全局归一化指标）
    metrics_norm_for_radar = safe_minmax_norm(metrics_df[METRICS_LIST])
    create_radar_plot_for_best_per_interval(final_12bands, metrics_norm_for_radar, RADAR_PLOT)

    # 10. 输出最终汇总信息（含全局归一化指标）
    print(f"\n=== 最终结果汇总 ===")
    print(f"📊 各100nm区间前{SELECT_PER_INTERVAL}名，共{len(final_12bands)}个波段")
    print("\n最终12个波段列表（按全局排名，含全局归一化指标）：")
    print(final_12bands[['波段名称', '所属区间', 'Wavelength', 'TOPSIS贴近度', '全局排名'] + GLOBAL_NORMALIZED_METRICS_LIST].to_string(index=False))

    # 11. 保存最终数据集（12个波段+目标列）
    final_band_cols = final_12bands['波段名称'].tolist()
    final_data_cols = [target_col] + final_band_cols
    final_data_cols = [col for col in final_data_cols if col in df_all.columns]
    df_final_data = df_all[final_data_cols].copy()
    df_final_data.to_excel(os.path.join(OUTPUT_DIR, "特征波段.xlsx"), index=False)
    print(f"\n✅ 最终12个波段数据集已保存：{os.path.join(OUTPUT_DIR, '最终12个波段数据集.xlsx')}")
    print("\n=== 全部流程完成 ===")

if __name__ == "__main__":
    run_pipeline()