import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

# ==================== 配置 ====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = "土壤碳含量-光谱反射率.xlsx"
TARGET_COL = "有机碳"
NON_FEATURE_COLS = ["序号", "编号"]

N_FEATURES = 30
MIN_WAVE_GAP = 10

# 自动创建输出文件夹
OUTPUT_FOLDER = os.path.join(CURRENT_DIR, "RFE_特征提取结果")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==================== 读取数据 ====================
df = pd.read_excel(DATA_FILE)
y = df[TARGET_COL].values
X = df.drop(columns=NON_FEATURE_COLS + [TARGET_COL])
feature_names = X.columns.tolist()

# ==================== XGBoost-RFE ====================
model = XGBRegressor(n_estimators=100, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=N_FEATURES)
rfe.fit(X.values, y)

# ==================== 排名表 ====================
ranking_df = pd.DataFrame({
    "波长(nm)": feature_names,
    "RFE排名": rfe.ranking_,
    "是否选中": rfe.support_
}).sort_values("RFE排名")

ranking_df.to_excel(os.path.join(OUTPUT_FOLDER, "XGBoost-RFE特征排名.xlsx"), index=False)

# ==================== 选中波段 ====================
selected_waves = [float(w) for w, s in zip(feature_names, rfe.support_) if s]
selected_waves = sorted(selected_waves)

# ==================== 去连续波段 ====================
def remove_consecutive(waves, min_gap=10):
    res = []
    last = -np.inf
    for w in waves:
        if w - last >= min_gap:
            res.append(w)
            last = w
    return res

final_waves = remove_consecutive(selected_waves, min_gap=MIN_WAVE_GAP)
final_waves_str = [str(int(w)) for w in final_waves]

print("✅ 最终无连续最优波段：")
print(final_waves_str)
print(f"最终数量：{len(final_waves_str)}")

# ==================== 保存筛选后的数据 ====================
df_out = df[NON_FEATURE_COLS + [TARGET_COL] + final_waves_str]
df_out.to_excel(os.path.join(OUTPUT_FOLDER, "特征波段子集.xlsx"), index=False)

# ==================== 绘图 1：RFE 排名图 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 把波长转成数字，并且按波长排序
waves_numeric = np.array([float(f) for f in feature_names])
sorted_idx = np.argsort(waves_numeric)
waves_sorted = waves_numeric[sorted_idx]
ranks_sorted = rfe.ranking_[sorted_idx]
selected_sorted = rfe.support_[sorted_idx]

plt.figure(figsize=(16, 6))
plt.bar(waves_sorted, ranks_sorted, color='skyblue', width=3)
plt.bar(waves_sorted[selected_sorted], ranks_sorted[selected_sorted], color='crimson', width=4, label='选中波段')
plt.axhline(1, color='red', linestyle='--', linewidth=2, label='选中波段(排名=1)')

# ✅ 这里设置 X 轴间隔 100nm
plt.xticks(np.arange(400, 1100, 100), fontsize=14)
plt.xlabel("波长(nm)")
plt.ylabel("RFE 排名（越小越重要）")
plt.title("XGBoost-RFE 特征排名")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "1_RFE特征排名图.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==================== 绘图 2：最终无连续波段 ====================
plt.figure(figsize=(12, 4))
plt.bar(final_waves, [1]*len(final_waves), width=4, color='orange')
plt.xlabel("波长(nm)")
plt.title(f"最终最优波段（间隔≥{MIN_WAVE_GAP}nm）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "2_最终无连续波段.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ 全部结果已保存到文件夹：\n{OUTPUT_FOLDER}")