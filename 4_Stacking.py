import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# ===================== 固定随机种子 =====================
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# ===================== 绘图格式 =====================
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1DCNN 模型 =====================
def get_best_1dcnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
    return model

# ===================== XGBoost =====================
def get_best_xgb():
    return xgb.XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=114,
        subsample=0.7,
        reg_alpha=4.168980052024814,
        reg_lambda=2.9245765443862353,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

# ===================== SVR =====================
def get_best_svr():
    return SVR(
        kernel='rbf',
        C=14.03543667913898,
        gamma=0.29285250388668294,
        epsilon=0.18951730321390894
    )

# ===================== PLS  =====================
def get_best_pls():
    return PLSRegression(n_components=3)

# ===================== 数据读取 =====================
df_train = pd.read_excel('Train_Set_SPXY.xlsx')
df_test = pd.read_excel('Test_Set_SPXY.xlsx')

X_train = df_train.iloc[:, 3:].values
y_train = df_train.iloc[:, 2].values
X_test = df_test.iloc[:, 3:].values
y_test = df_test.iloc[:, 2].values

# ===================== 预处理 =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
input_shape = (X_train_cnn.shape[1], 1)

# ===================== 指标计算（最全版本） =====================
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    rpd = np.std(y_true) / rmse
    return r2, rmse, mae, bias, rpd

# ===================== 基模型列表 =====================
base_models = [
    ("1DCNN", get_best_1dcnn(input_shape)),
    ("XGB", get_best_xgb()),
    ("SVR", get_best_svr()),
    ("PLS", get_best_pls()),
]
n_models = len(base_models)

# ===================== 5折 Stacking OOF =====================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

new_train = np.zeros((len(X_train), n_models))
new_test = np.zeros((len(X_test), n_models))

for idx, (name, model) in enumerate(base_models):
    print(f"训练基模型：{name}")

    for train_idx, val_idx in kf.split(X_train):
        if name == "1DCNN":
            model.fit(X_train_cnn[train_idx], y_train[train_idx], epochs=100, batch_size=4, verbose=0)
            new_train[val_idx, idx] = model.predict(X_train_cnn[val_idx], verbose=0).flatten()
        else:
            model.fit(X_train_scaled[train_idx], y_train[train_idx])
            new_train[val_idx, idx] = model.predict(X_train_scaled[val_idx]).flatten()

    if name == "1DCNN":
        model.fit(X_train_cnn, y_train, epochs=100, batch_size=4, verbose=0)
        new_test[:, idx] = model.predict(X_test_cnn, verbose=0).flatten()
    else:
        model.fit(X_train_scaled, y_train)
        new_test[:, idx] = model.predict(X_test_scaled).flatten()

# ===================== 元学习器 =====================
meta_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
meta_model.fit(new_train, y_train)

# ===================== 预测 =====================
y_train_pred = meta_model.predict(new_train)
y_test_pred = meta_model.predict(new_test)

# ===================== 输出指标 =====================
r2_train, rmse_train, mae_train, bias_train, rpd_train = calculate_metrics(y_train, y_train_pred)
r2_test, rmse_test, mae_test, bias_test, rpd_test = calculate_metrics(y_test, y_test_pred)

print("=" * 70)
print("        🔥 Stacking 最终版：1DCNN + XGB + SVR + PLS 🔥")
print("=" * 70)
print(f"训练集 | R²: {r2_train:.4f} | RMSE: {rmse_train:.4f} | MAE: {mae_train:.4f} | Bias: {bias_train:.4f} | RPD: {rpd_train:.4f}")
print(f"测试集 | R²: {r2_test:.4f} | RMSE: {rmse_test:.4f} | MAE: {mae_test:.4f} | Bias: {bias_test:.4f} | RPD: {rpd_test:.4f}")
print("=" * 70)

# ===================== 【核心：导出所有数据到 Excel】=====================
# 1. 训练集详细数据
train_detail = pd.DataFrame({
    "Train_Measured": y_train,
    "Train_Predicted": y_train_pred,
    "Train_Residual": y_train_pred - y_train,          # 残差（偏差）
    "Train_Abs_Residual": np.abs(y_train_pred - y_train) # 绝对残差
})

# 2. 测试集详细数据
test_detail = pd.DataFrame({
    "Test_Measured": y_test,
    "Test_Predicted": y_test_pred,
    "Test_Residual": y_test_pred - y_test,
    "Test_Abs_Residual": np.abs(y_test_pred - y_test)
})

# 3. 模型最终评价指标
metrics = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "R2": [r2_train, r2_test],
    "RMSE": [rmse_train, rmse_test],
    "MAE": [mae_train, mae_test],
    "Bias": [bias_train, bias_test],
    "RPD": [rpd_train, rpd_test]
})

# 4. 基模型输出（元模型输入）→ 绘图超级有用
base_train_df = pd.DataFrame(new_train, columns=["1DCNN_OOF", "XGB_OOF", "SVR_OOF", "PLS_OOF"])
base_test_df = pd.DataFrame(new_test, columns=["1DCNN_Pred", "XGB_Pred", "SVR_Pred", "PLS_Pred"])

# 合并所有数据
all_data = pd.concat([
    train_detail, 
    test_detail, 
    base_train_df, 
    base_test_df
], axis=1)

# 保存 Excel（3个sheet，完美适配论文）
with pd.ExcelWriter("Stacking_Results_For_Paper8.xlsx", engine="openpyxl") as writer:
    all_data.to_excel(writer, sheet_name="Prediction_Details", index=False)
    metrics.to_excel(writer, sheet_name="Model_Metrics", index=False)
    pd.DataFrame({
        "Model": ["1DCNN", "XGBoost", "SVR", "PLS", "Stacking"]
    }).to_excel(writer, sheet_name="Model_List", index=False)

print("✅ Excel 已保存：Stacking_Results_For_Paper.xlsx")
print("✅ 包含：预测值、残差、偏差、基模型输出、指标表")

# ===================== 论文格式散点图 =====================
plt.figure(figsize=(7, 6))
plt.scatter(y_train, y_train_pred, c='royalblue', alpha=0.6, label='Train', s=25)
plt.scatter(y_test, y_test_pred, c='crimson', alpha=0.7, label='Test', s=25)
min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
plt.xlabel('Measured SOC (g/kg)', fontsize=12)
plt.ylabel('Predicted SOC (g/kg)', fontsize=12)
plt.title(f'Stacking Ensemble Model\nTest R²={r2_test:.4f}, RPD={rpd_test:.2f}', fontsize=13)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Stacking_1to1.png", dpi=300, bbox_inches="tight")
plt.show()