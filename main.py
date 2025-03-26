import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from pathlib import Path
import joblib

# 设置中文显示和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------
# 1. 数据加载与预处理
# ----------------------
def load_and_preprocess():
    # 加载数据
    data = pd.read_excel(r"Your absorbance file.xlsx", index_col="样本编号")
    labels = pd.read_excel(r"Your concentration file.xlsx", index_col="样本编号")


    # 转换为NumPy数组
    X = data.values.astype(np.float32)
    y = labels.values.astype(np.float32)

    # 数据预处理
    y_log = np.log1p(y)  # 对数变换
    min_log = y_log.min(axis=0)  # 记录每个特征的最小log值
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_log)

    # 调整数据维度 (样本数, 序列长度, 特征数)
    X_scaled = np.expand_dims(X_scaled, axis=-1)

    return X_scaled, y_scaled, scaler_X, scaler_y, min_log


# ----------------------
# 2. Transformer模型构建
# ----------------------
class PositionalEncoding(layers.Layer):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def get_angles(self, position, i):
        # 使用TensorFlow操作代替NumPy
        angle_rate = 1 / tf.pow(
            10000.0,
            (2 * (i // 2)) / tf.cast(self.d_model, tf.float32)
        )
        return position * angle_rate

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]

        # 生成位置索引
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]

        # 计算角度
        angle_rads = self.get_angles(position, i)

        # 正弦和余弦编码
        sin_mask = tf.cast(tf.range(self.d_model) % 2, tf.float32)
        sin_part = tf.math.sin(angle_rads) * (1 - sin_mask)
        cos_part = tf.math.cos(angle_rads) * sin_mask

        # 合并位置编码
        pos_encoding = sin_part + cos_part
        pos_encoding = pos_encoding[tf.newaxis, ...]  # 增加batch维度

        # 调整幅度并相加
        return inputs * tf.math.sqrt(tf.cast(self.d_model, tf.float32)) + pos_encoding


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # 多头注意力
    x = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    res = x + inputs  # 残差连接

    # 前馈网络
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_transformer(input_shape, num_outputs):

    inputs = layers.Input(shape=input_shape)

    # 输入嵌入层
    x = layers.Conv1D(128, kernel_size=2, activation="relu")(inputs)
    x = PositionalEncoding(128)(x)

    # 堆叠Transformer层
    for _ in range(3):  # 3个编码器层
        x = transformer_encoder(x, head_size=128, num_heads=8, ff_dim=256)

    # 输出层
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_outputs)(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mae",
        metrics=["mae"]
    )
    return model


# ----------------------
# 3. 模型训练与评估
# ----------------------
def train_and_evaluate():
    # 加载数据
    X, y, scaler_X, scaler_y, min_log = load_and_preprocess()

    # 定义交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metals = ["Sb", "Fe", "Ni", "Cd", "Cu"]
    fold_performance = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"开始第 {fold + 1} 折训练...")

        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 构建模型
        model = build_transformer(X_train.shape[1:], y_train.shape[1])

        # 训练配置
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)
        ]

        # 训练模型
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )

        # 模型评估
        def inverse_transform(y):
            return np.expm1(scaler_y.inverse_transform(y))

        y_pred_val = model.predict(X_val)
        y_val_orig = inverse_transform(y_val)
        y_pred_val_orig = inverse_transform(y_pred_val)

        # 评估指标计算
        fold_metrics = {}
        for i, metal in enumerate(metals):
            rmse_val = np.sqrt(mean_squared_error(y_val_orig[:, i], y_pred_val_orig[:, i]))
            r2_val = r2_score(y_val_orig[:, i], y_pred_val_orig[:, i])
            mae_val = mean_absolute_error(y_val_orig[:, i], y_pred_val_orig[:, i])

            fold_metrics[metal] = {
                "RMSE": rmse_val,
                "R2": r2_val,
                "MAE": mae_val
            }

        fold_performance.append(fold_metrics)

    # 打印交叉验证结果
    print("\n交叉验证结果：")
    for fold, metrics in enumerate(fold_performance):
        print(f"\n第 {fold + 1} 折评估结果：")
        for metal, scores in metrics.items():
            print(f"金属 {metal}: RMSE={scores['RMSE']:.4f}, R2={scores['R2']:.4f}, MAE={scores['MAE']:.4f}")

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="训练损失")
    plt.plot(history.history["val_loss"], label="验证损失")
    plt.title("训练过程损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # 在模型评估部分添加以下代码
    def plot_scatter(y_true, y_pred, metals, save_path=None):
        """
        生成带统计指标的专业散点图
        :param y_true: 真实浓度数组 (n_samples, n_metals)
        :param y_pred: 预测浓度数组 (n_samples, n_metals)
        :param metals: 金属名称列表
        :param save_path: 图片保存路径（可选）
        """
        plt.figure(figsize=(18, 12))

        # 颜色配置
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 科学配色

        for idx, metal in enumerate(metals):
            ax = plt.subplot(2, 3, idx + 1)  # 调整为2行3列布局

            # 计算关键指标
            rmse = np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))
            r2 = r2_score(y_true[:, idx], y_pred[:, idx])
            mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])

            # 绘制散点
            scatter = ax.scatter(
                y_true[:, idx],
                y_pred[:, idx],
                c=colors[idx],
                alpha=0.6,
                edgecolors='w',
                linewidths=0.5,
                s=60
            )

            # 绘制理想线
            max_val = max(y_true[:, idx].max(), y_pred[:, idx].max()) * 1.05
            ax.plot([0, max_val], [0, max_val], '--', color='#2ca02c', linewidth=1.5, alpha=0.8)

            # 设置坐标轴
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_xlabel('Real (mg/L)', fontsize=10, labelpad=8)
            ax.set_ylabel('Prediction (mg/L)', fontsize=10, labelpad=8)
            ax.tick_params(axis='both', which='major', labelsize=9)

            # 添加统计信息
            text_str = f'$R^2 = {r2:.2f}$\n$RMSE = {rmse:.2f}$\n$MAE = {mae:.2f}$'
            ax.text(
                0.05, 0.85, text_str,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3.0)
            )

            # 标题设置
            ax.set_title(f'{metal}', fontsize=12, pad=12, fontweight='bold')

            # 网格线优化
            ax.grid(True, linestyle='--', alpha=0.4, which='both')

        # 调整布局并保存
        plt.tight_layout(pad=3.0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"散点图已保存至: {save_path}")

        plt.show()

    # 在评估代码中调用
    plot_scatter(
        y_val_orig,
        y_pred_val_orig,
        metals=["Sb", "Fe", "Ni", "Cd", "Cu"],
        save_path=r"E:\xxxx\浓度预测散点图.png"
    )

    # 保存结果
    def save_results(y_true, y_pred, folder, filename):
        try:
            save_path = Path(folder)
            save_path.mkdir(parents=True, exist_ok=True)

            # 创建DataFrame时确保样本对齐
            data = {}
            metals = ["Sb", "Fe", "Ni", "Cd", "Cu"]
            for i, metal in enumerate(metals):
                data[f"True_{metal}"] = y_true[:, i]
                data[f"Pred_{metal}"] = y_pred[:, i]

            # 创建DataFrame并保存
            df = pd.DataFrame(data)
            df.to_excel(save_path / filename, index=False)
            print(f"成功保存到：{save_path / filename}")
        except Exception as e:
            print(f"保存失败：{str(e)}")

    output_folder = r"E:\xxx"
    save_results(y_val_orig, y_pred_val_orig, output_folder, "Transformer_测试结果.xlsx")

    # ----------------------
    # 4. 模型保存
    # ----------------------
def save_model_artifacts(model, scaler_X, scaler_y, min_log):
    """保存模型及相关预处理对象"""
    save_dir = Path(r"E:\xxxx")
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
         # 保存完整模型
        model.save(save_dir / "transformer_model.h5")
        # 保存预处理对象
        joblib.dump(scaler_X, save_dir / "scaler_X.pkl")
        joblib.dump(scaler_y, save_dir / "scaler_y.pkl")
        joblib.dump(min_log, save_dir / "min_log.pkl")
        print(f"模型及相关文件已保存至：{save_dir}")
    except Exception as e:
        print(f"保存失败：{str(e)}")

# 在train_and_evaluate()函数末尾调用
    save_model_artifacts(model, scaler_X, scaler_y, min_log)


# ----------------------
# 主程序入口
# ----------------------
if __name__ == "__main__":
    # 环境验证
    # 启动训练流程
    train_and_evaluate()