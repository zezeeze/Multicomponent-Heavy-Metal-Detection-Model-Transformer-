import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# ====================== 配置参数 ======================
ABSORBANCE_FILE = r"E:\重金属实验\酶标仪\纯+真=吸光度.xlsx"
CONCENTRATION_FILE = r"E:\重金属实验\酶标仪\纯+真=浓度.xlsx"
ABS_OUTPUT = "aug_absorbance.xlsx"
CONC_OUTPUT = "aug_concentration.xlsx"

# 增强数量控制（绝对值）
LINEAR_AUG_NUM = 400  # 线性插值生成数量
NOISE_AUG_NUM = 100  # 噪声注入生成数量
SMOTE_AUG_NUM = 100  # SMOTE生成数量

NOISE_LEVEL = 0.2
K_NEIGHBORS = 5


# ====================== 数据加载 ======================
def load_data_with_id(abs_file, conc_file):
    """加载带样本编号的数据"""
    df_abs = pd.read_excel(abs_file, index_col=0)
    df_conc = pd.read_excel(conc_file, index_col=0)

    # 验证样本编号一致性
    assert df_abs.index.tolist() == df_conc.index.tolist(), "样本编号不一致"

    print(f"成功加载数据：{len(df_abs)}个样本")
    print(f"吸光度维度：{df_abs.shape[1]}，浓度维度：5")
    return df_abs, df_conc


# ====================== 预处理 ======================
def preprocess_data(df_abs, df_conc):
    """分离编号并标准化数据"""
    sample_ids = df_abs.index.values
    X = df_abs.values
    y = df_conc.values

    # 标准化
    X_scaler = StandardScaler().fit(X)
    y_scaler = MinMaxScaler().fit(y)

    return sample_ids, X_scaler.transform(X), y_scaler.transform(y), X_scaler, y_scaler


# ====================== 增强方法 ======================
# ====================== 数据加载 ======================
def load_data_with_id(abs_file, conc_file):
    """加载带数字编号的数据"""
    # 读取时明确指定索引列为第一列，保持原始数字格式
    df_abs = pd.read_excel(abs_file, index_col=0)
    df_conc = pd.read_excel(conc_file, index_col=0)

    # 验证数据一致性
    assert df_abs.index.tolist() == df_conc.index.tolist(), "样本编号不一致"
    print(f"成功加载 {len(df_abs)} 个样本，最大编号：{df_abs.index.max()}")
    return df_abs, df_conc


# ====================== 预处理 ======================
def preprocess_data(df_abs, df_conc):
    """数据标准化处理"""
    X = df_abs.values.astype(np.float32)
    y = df_conc.values.astype(np.float32)

    X_scaler = StandardScaler().fit(X)
    y_scaler = MinMaxScaler().fit(y)

    return X_scaler.transform(X), y_scaler.transform(y), X_scaler, y_scaler


# ====================== 增强方法 ======================
def linear_augmentation(X, y, num_samples):
    """线性插值增强"""
    augmented = []
    for _ in range(num_samples):
        i, j = np.random.choice(len(X), 2, replace=False)
        alpha = np.random.uniform(0.3, 0.7)
        new_X = alpha * X[i] + (1 - alpha) * X[j]
        new_y = alpha * y[i] + (1 - alpha) * y[j]
        augmented.append((new_X, new_y))
    return augmented


def noise_augmentation(X, y, num_samples):
    """噪声注入增强"""
    augmented = []
    for _ in range(num_samples):
        idx = np.random.randint(len(X))
        noise = np.random.normal(0, NOISE_LEVEL, X.shape[1])
        augmented.append((X[idx] + noise, y[idx]))
    return augmented


def smote_augmentation(X, y, num_samples, k=5):
    """SMOTE增强"""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    augmented = []
    for _ in range(num_samples):
        i = np.random.randint(len(X))
        _, indices = nn.kneighbors([X[i]], n_neighbors=k)
        j = np.random.choice(indices[0][1:])  # 排除自身
        alpha = np.random.rand()
        new_X = alpha * X[i] + (1 - alpha) * X[j]
        new_y = alpha * y[i] + (1 - alpha) * y[j]
        augmented.append((new_X, new_y))
    return augmented


# ====================== 验证与后处理 ======================
def validate_samples(augmented, original_y):
    """验证增强样本合理性"""
    valid = []
    total_conc = np.sum(original_y, axis=1)
    conc_threshold = np.percentile(total_conc, 97) * 1.2

    for x, y in augmented:
        # 浓度非负且总和合理
        if np.any(y < 0) or np.sum(y) > conc_threshold:
            continue
        # 吸光度在合理范围
        if np.max(x) > 3.5 or np.min(x) < -3.5:
            continue
        valid.append((x, y))
    return valid


def generate_new_ids(original_ids, aug_num):
    """生成连续数字编号"""
    max_id = original_ids.max()
    return list(range(max_id + 1, max_id + aug_num + 1))


# ====================== 可视化 ======================
def plot_distribution_comparison(orig_X, aug_X, orig_y, aug_y):
    """分布对比可视化"""
    plt.figure(figsize=(15, 6))

    # 吸光度PCA
    pca = PCA(n_components=2)
    combined_X = np.vstack([orig_X, aug_X])
    pca_result = pca.fit_transform(combined_X)

    plt.subplot(121)
    plt.scatter(pca_result[:len(orig_X), 0], pca_result[:len(orig_X), 1],
                alpha=0.5, label='原始数据')
    plt.scatter(pca_result[len(orig_X):, 0], pca_result[len(orig_X):, 1],
                alpha=0.5, label='增强数据')
    plt.title("吸光度PCA分布")
    plt.legend()

    # 浓度分布
    plt.subplot(122)
    bins = np.linspace(0, np.max(orig_y) * 1.1, 20)
    for i in range(5):
        plt.hist(orig_y[:, i], bins, alpha=0.3, label=f'原始 Metal{i + 1}')
        plt.hist(aug_y[:, i], bins, alpha=0.3, label=f'增强 Metal{i + 1}')
    plt.title("浓度分布对比")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('aug_distribution.png', dpi=300)
    plt.show()


# ====================== 主流程 ======================
if __name__ == "__main__":
    # 加载数据
    df_abs, df_conc = load_data_with_id(ABSORBANCE_FILE, CONCENTRATION_FILE)
    original_ids = df_abs.index.values

    # 预处理
    X, y, X_scaler, y_scaler = preprocess_data(df_abs, df_conc)

    # 执行增强
    linear = linear_augmentation(X, y, LINEAR_AUG_NUM)
    noise = noise_augmentation(X, y, NOISE_AUG_NUM)
    smote = smote_augmentation(X, y, SMOTE_AUG_NUM, K_NEIGHBORS)

    # 验证样本
    valid_samples = validate_samples(linear + noise + smote, y)
    print(f"有效增强样本：{len(valid_samples)}/{LINEAR_AUG_NUM + NOISE_AUG_NUM + SMOTE_AUG_NUM}")

    # 准备输出数据
    X_aug = np.array([s[0] for s in valid_samples])
    y_aug = np.array([s[1] for s in valid_samples])

    # 反标准化
    X_final = X_scaler.inverse_transform(np.vstack([X, X_aug]))
    y_final = y_scaler.inverse_transform(np.vstack([y, y_aug]))

    # 生成新编号
    new_ids = generate_new_ids(original_ids, len(valid_samples))
    all_ids = np.concatenate([original_ids, new_ids])

    # 保存文件
    pd.DataFrame(X_final, index=all_ids, columns=df_abs.columns).to_excel(ABS_OUTPUT)
    pd.DataFrame(y_final, index=all_ids, columns=df_conc.columns).to_excel(CONC_OUTPUT)

    # 可视化
    plot_distribution_comparison(X, X_aug, y, y_aug)