import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ===================== 可调参数 =====================
FILE_NAME   = "最终12个波段数据集.xlsx"  # 输入文件
TEST_SIZE   = 0.30                    # 测试集占总样本的比例
RANDOM_STATE= 42
OUT_TRAIN   = "Train_Set_SPXY.xlsx"   # 导出的训练集文件名
OUT_TEST    = "Test_Set_SPXY.xlsx"    # 导出的测试集文件名
# ===================================================

np.random.seed(RANDOM_STATE)

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
file_path = os.path.join(current_dir, FILE_NAME)

def spxy_split(X, y, test_size, random_state=None):
    """
    使用 SPXY 算法划分样本集，返回训练集和测试集的索引。
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(np.ceil(n_samples * test_size))
    n_train = n_samples - n_test
    
    # X 和 Y 归一化
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1 
    X_norm = (X - X_min) / X_range
    
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    y_range = y_range if y_range != 0 else 1
    y_norm = (y - y_min) / y_range
    y_norm = y_norm.reshape(-1, 1)

    # 计算 SPXY 联合距离矩阵
    XY_norm = np.hstack((X_norm, y_norm))
    D = cdist(XY_norm, XY_norm, metric='euclidean')

    # 贪婪选择 (Minimax 策略)
    initial_idx = np.random.randint(0, n_samples)
    train_indices = [initial_idx]
    remaining_indices = list(range(n_samples))
    remaining_indices.remove(initial_idx)
    
    for _ in range(n_train - 1):
        min_dist_to_train = D[train_indices, :][:, remaining_indices].min(axis=0)
        max_min_dist_idx = np.argmax(min_dist_to_train)
        
        new_idx = remaining_indices[max_min_dist_idx]
        train_indices.append(new_idx)
        remaining_indices.pop(max_min_dist_idx)

    test_indices = remaining_indices
    return np.array(train_indices), np.array(test_indices)


def main_export():
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到文件：{file_path}")
    
    print(f"开始使用 SPXY 划分 {data.shape[0]} 个样本...")

    # 数据提取
    y = data.iloc[:, 2].astype(float).values
    X_df = data.iloc[:, 3:].copy()
    X = X_df.values.astype(float)
    
    # 执行 SPXY 划分
    train_indices, test_indices = spxy_split(X, y, TEST_SIZE, RANDOM_STATE)
    
    # 导出文件
    df_train_out = data.iloc[train_indices].reset_index(drop=True)
    df_test_out = data.iloc[test_indices].reset_index(drop=True)
    
    df_train_out.to_excel(os.path.join(current_dir, OUT_TRAIN), index=False)
    df_test_out.to_excel(os.path.join(current_dir, OUT_TEST), index=False)
    
    print("=" * 30)
    print(f"[成功] 训练集导出到: {OUT_TRAIN} ({len(train_indices)} 个样本)")
    print(f"[成功] 测试集导出到: {OUT_TEST} ({len(test_indices)} 个样本)")
    print("=" * 30)

if __name__ == "__main__":
    main_export()