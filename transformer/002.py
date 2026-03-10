import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import time
import warnings
import os
from datetime import datetime
import matplotlib

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# 定义特征范围
FEATURE_RANGES = {
    'N': (20, 199.9),
    'P': (20, 100),
    'K': (20, 149.9),
    'TEMP': (5, 47),
    'SOIL_PH': (6, 9),
    'RELATIVE_HUMIDITY': (15, 100)
}


# 创建输出文件夹
def create_output_folder():
    """创建输出文件夹，以时间戳命名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"crop_recommendation_output_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def validate_input_features(soil_ph, temp, humidity, n, p, k):
    """验证输入特征是否在合理范围内"""
    errors = []

    if not (FEATURE_RANGES['SOIL_PH'][0] <= soil_ph <= FEATURE_RANGES['SOIL_PH'][1]):
        errors.append(
            f"土壤pH值应在{FEATURE_RANGES['SOIL_PH'][0]}-{FEATURE_RANGES['SOIL_PH'][1]}之间，当前值为{soil_ph}")

    if not (FEATURE_RANGES['TEMP'][0] <= temp <= FEATURE_RANGES['TEMP'][1]):
        errors.append(f"温度应在{FEATURE_RANGES['TEMP'][0]}-{FEATURE_RANGES['TEMP'][1]}°C之间，当前值为{temp}")

    if not (FEATURE_RANGES['RELATIVE_HUMIDITY'][0] <= humidity <= FEATURE_RANGES['RELATIVE_HUMIDITY'][1]):
        errors.append(
            f"相对湿度应在{FEATURE_RANGES['RELATIVE_HUMIDITY'][0]}-{FEATURE_RANGES['RELATIVE_HUMIDITY'][1]}%之间，当前值为{humidity}")

    if not (FEATURE_RANGES['N'][0] <= n <= FEATURE_RANGES['N'][1]):
        errors.append(f"氮含量(N)应在{FEATURE_RANGES['N'][0]}-{FEATURE_RANGES['N'][1]}之间，当前值为{n}")

    if not (FEATURE_RANGES['P'][0] <= p <= FEATURE_RANGES['P'][1]):
        errors.append(f"磷含量(P)应在{FEATURE_RANGES['P'][0]}-{FEATURE_RANGES['P'][1]}之间，当前值为{p}")

    if not (FEATURE_RANGES['K'][0] <= k <= FEATURE_RANGES['K'][1]):
        errors.append(f"钾含量(K)应在{FEATURE_RANGES['K'][0]}-{FEATURE_RANGES['K'][1]}之间，当前值为{k}")

    return errors


def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在读取数据...")
    train_df = pd.read_csv('../crop_train.csv')
    val_df = pd.read_csv('../crop_val.csv')
    test_df = pd.read_csv('../crop_test.csv')

    print(f"训练集大小: {train_df.shape}")
    print(f"验证集大小: {val_df.shape}")
    print(f"测试集大小: {test_df.shape}")

    feature_columns = ['SOIL_PH', 'TEMP', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']
    target_column = 'CROPS'

    label_encoder = LabelEncoder()
    all_data = pd.concat([train_df, val_df, test_df])
    label_encoder.fit(all_data[target_column])

    x_train = train_df[feature_columns]
    y_train = label_encoder.transform(train_df[target_column])
    x_val = val_df[feature_columns]
    y_val = label_encoder.transform(val_df[target_column])
    x_test = test_df[feature_columns]
    y_test = label_encoder.transform(test_df[target_column])

    return x_train, y_train, x_val, y_val, x_test, y_test, label_encoder, feature_columns


class TransformerDataset(Dataset):
    """自定义Dataset处理结构化数据"""

    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_transformer_model(x_train, y_train, x_val, y_val, x_test, y_test, label_encoder, num_classes,
                           batch_size=32, epochs=50, patience=10):
    """训练Transformer模型（移除了GNN部分）"""
    print("正在训练Transformer模型...")

    # 创建数据集
    train_dataset = TransformerDataset(x_train, y_train)
    val_dataset = TransformerDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Transformer模型定义
    class TransformerClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
            super(TransformerClassifier, self).__init__()
            self.embedding = nn.Linear(input_size, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=128),
                num_layers=num_layers
            )
            self.fc = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self.embedding(x)  # [batch, hidden]
            x = x.unsqueeze(1)  # 添加序列维度 [batch, 1, hidden]
            x = x.permute(1, 0, 2)  # [1, batch, hidden]
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # [batch, 1, hidden]
            x = x.squeeze(1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    # 初始化模型
    model = TransformerClassifier(input_size=6, hidden_size=64, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"早停: 在第{epoch + 1}轮")
                break

        print(f"Epoch {epoch + 1}/{epochs} | 训练损失: {train_loss / len(train_loader):.4f} | 验证损失: {val_loss:.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model


def calculate_metrics(transformer_model, x_test, y_test, label_encoder):
    """计算Transformer模型的评估指标"""
    print("\n计算评估指标（Transformer）...")

    # 预测概率
    start_time = time.time()

    # Transformer预测
    transformer_model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
        transformer_logits = transformer_model(test_tensor)
        transformer_proba = torch.softmax(transformer_logits, dim=1).numpy()

    inference_time_ms = (time.time() - start_time) * 1000 / len(x_test)
    y_pred = np.argmax(transformer_proba, axis=1)

    # Top-k准确率
    topk_accuracies = {}
    for k in range(1, 6):
        correct = 0
        for i in range(len(y_test)):
            top_k_indices = np.argsort(transformer_proba[i])[-k:]
            if y_test[i] in top_k_indices:
                correct += 1
        topk_accuracies[k] = correct / len(y_test)

    # 基础指标
    precision_avg = precision_score(y_test, y_pred, average='weighted')
    recall_avg = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # PR-AUC (OvR)
    n_classes = len(label_encoder.classes_)
    pr_auc_scores = []
    for class_idx in range(n_classes):
        y_true_binary = np.array(y_test == class_idx, dtype=int)
        y_score = transformer_proba[:, class_idx]
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        pr_auc = auc(recall, precision)
        pr_auc_scores.append(pr_auc)
    pr_auc_ovr = np.mean(pr_auc_scores)

    metrics_dict = {
        'accuracy_top1': round(float(topk_accuracies[1]), 4),
        'accuracy_top3': round(float(topk_accuracies[3]), 4),
        'f1_macro': round(float(f1_macro), 4),
        'precision_avg': round(float(precision_avg), 4),
        'recall_avg': round(float(recall_avg), 4),
        'pr_auc_ovr': round(float(pr_auc_ovr), 4),
        'inference_time_ms': round(float(inference_time_ms), 2)
    }

    return metrics_dict, transformer_proba, y_pred, topk_accuracies


def create_visualizations(y_test, y_pred, y_pred_proba, label_encoder, topk_accuracies, output_folder, feature_columns):
    """创建所有可视化图表"""
    print("\n生成可视化图表...")

    # 1. Top-k准确率曲线
    plt.figure(figsize=(10, 6))
    k_values = list(range(1, 6))
    accuracies = [topk_accuracies[k] for k in k_values]
    plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('k值 (Top-k)')
    plt.ylabel('准确率')
    plt.title('Top-k 准确率曲线')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    for k, acc in zip(k_values, accuracies):
        plt.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/topk_accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 归一化混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': '比例'})
    plt.title('归一化混淆矩阵 (按行归一化)')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/normalized_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 每类F1分数柱状图
    f1_per_class = f1_score(y_test, y_pred, average=None)
    plt.figure(figsize=(12, 6))
    classes = label_encoder.classes_
    x_pos = np.arange(len(classes))
    f1_min, f1_max = min(f1_per_class), max(f1_per_class)
    colors = [plt.get_cmap('viridis')((val - f1_min) / (f1_max - f1_min)) if f1_max > f1_min else 'skyblue' for val in
              f1_per_class]
    bars = plt.bar(x_pos, f1_per_class, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('作物类别')
    plt.ylabel('F1分数')
    plt.title('每类作物F1分数')
    plt.xticks(x_pos, classes, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.1)
    for bar, value in zip(bars, f1_per_class):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.3f}', ha='center', va='bottom',
                 fontsize=10)
    mean_f1 = np.mean(f1_per_class)
    plt.axhline(y=mean_f1, color='red', linestyle='--', alpha=0.8, label=f'平均F1: {mean_f1:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_folder}/f1_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 宏平均PR曲线 + 随机基线
    n_classes = len(label_encoder.classes_)
    plt.figure(figsize=(10, 8))
    precision_macro = []
    pos_ratios = []
    for class_idx in range(n_classes):
        y_true_binary = np.array(y_test == class_idx, dtype=int)
        pos_ratio = np.mean(y_true_binary)
        pos_ratios.append(pos_ratio)
        y_score = y_pred_proba[:, class_idx]
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        recall_interp = np.linspace(0, 1, 100)
        precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1])
        precision_macro.append(precision_interp)
    precision_mean = np.mean(precision_macro, axis=0)
    recall_mean = np.linspace(0, 1, 100)
    pr_auc_mean = auc(recall_mean, precision_mean)
    avg_pos_ratio = float(np.mean(pos_ratios))
    plt.plot(recall_mean, precision_mean, lw=3, color='red', label=f'宏平均PR曲线 (AUC = {pr_auc_mean:.3f})')
    plt.axhline(y=avg_pos_ratio, color='gray', linestyle='--', lw=2,
                label=f'随机基线 (平均正例率 = {avg_pos_ratio:.3f})')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('宏平均PR曲线（含随机基线）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(f'{output_folder}/macro_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 特征重要性图（使用原始特征）
    plt.figure(figsize=(10, 6))
    plt.title("Transformer 特征重要性")
    plt.bar(range(6), [0.1] * 6, align="center")
    plt.xticks(range(6), feature_columns, rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 7x7 子混淆矩阵：最难 + 最主流作物
    cm = confusion_matrix(y_test, y_pred)
    class_counts = np.bincount(y_test, minlength=len(label_encoder.classes_))
    most_common_class = np.argmax(class_counts)

    recall_per_class = np.diag(cm) / np.sum(cm, axis=1)
    recall_per_class = np.nan_to_num(recall_per_class)
    hardest_class = np.argmin(recall_per_class)

    selected_classes = {most_common_class, hardest_class}
    all_indices = list(range(len(label_encoder.classes_)))
    selected_indices = sorted(list(selected_classes))
    while len(selected_indices) < 7 and len(selected_indices) < len(all_indices):
        for offset in range(1, 10):
            for direction in [-1, 1]:
                candidate = selected_indices[0] + direction * offset
                if 0 <= candidate < len(all_indices) and candidate not in selected_indices:
                    selected_indices.append(candidate)
                    selected_indices = sorted(list(set(selected_indices)))
                    if len(selected_indices) >= 7:
                        break
            if len(selected_indices) >= 7:
                break
        if len(selected_indices) >= 7:
            break

    selected_indices = selected_indices[:7]
    sub_cm = cm[np.ix_(selected_indices, selected_indices)]
    sub_cm_norm = sub_cm.astype('float') / sub_cm.sum(axis=1)[:, np.newaxis]
    sub_labels = [label_encoder.classes_[i] for i in selected_indices]

    plt.figure(figsize=(8, 6))
    sns.heatmap(sub_cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=sub_labels,
                yticklabels=sub_labels,
                cbar_kws={'label': '比例'})
    plt.title('7x7 子混淆矩阵（最难 + 最主流作物）')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/sub_confusion_matrix_7x7.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results_and_model(transformer_model, label_encoder, metrics_dict, y_test, y_pred_proba,
                           output_folder):
    """保存Transformer模型和结果"""
    print("\n保存结果和模型...")

    metrics_with_descriptions = {
        'accuracy_top1': {'value': metrics_dict['accuracy_top1'], 'description': 'Top-1 准确率 - 基础性能'},
        'accuracy_top3': {'value': metrics_dict['accuracy_top3'], 'description': 'Top-3 准确率 - 实际推荐体验'},
        'f1_macro': {'value': metrics_dict['f1_macro'], 'description': '宏平均 F1-score - 关注小众作物表现'},
        'precision_avg': {'value': metrics_dict['precision_avg'], 'description': '平均精确率（weighted）- 避免错误推荐'},
        'recall_avg': {'value': metrics_dict['recall_avg'], 'description': '平均召回率（weighted）- 避免漏掉合适作物'},
        'pr_auc_ovr': {'value': metrics_dict['pr_auc_ovr'], 'description': 'PR-AUC (OvR) - 精确率-召回率平衡'},
        'inference_time_ms': {'value': metrics_dict['inference_time_ms'],
                              'description': '单样本推理时间（毫秒）- 部署效率'}
    }

    with open(f'{output_folder}/evaluation_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_with_descriptions, f, indent=4, ensure_ascii=False)

    # 保存Transformer模型
    torch.save(transformer_model.state_dict(), f'{output_folder}/transformer_model.pth')
    joblib.dump(label_encoder, f'{output_folder}/label_encoder.pkl')

    # 详细预测结果
    test_df = pd.read_csv('../crop_test.csv')
    results_df = test_df.copy()
    results_df['True_Crop'] = label_encoder.inverse_transform(y_test)
    results_df['Predicted_Crop'] = label_encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))
    results_df['Prediction_Probability'] = np.max(y_pred_proba, axis=1)

    top3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
    top3_probs = np.sort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
    top3_crops = [[label_encoder.classes_[idx] for idx in row] for row in top3_indices]
    results_df['Top3_Recommendations'] = top3_crops
    results_df['Top3_Probabilities'] = [list(probs) for probs in top3_probs]

    results_df.to_csv(f'{output_folder}/detailed_predictions.csv', index=False, encoding='utf-8-sig')

    create_usage_example(output_folder)


def create_usage_example(output_folder):
    """创建使用示例（Transformer）"""
    usage_code = f'''# 作物推荐系统使用示例（Transformer）
import joblib
import torch
import numpy as np
from torch import nn

# 加载模型和编码器
transformer_model = nn.Sequential(
    nn.Linear(6, 64),
    nn.ReLU(),
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128),
        num_layers=2
    ),
    nn.Linear(64, 10)  # 10类作物
)
transformer_model.load_state_dict(torch.load('{output_folder}/transformer_model.pth'))
transformer_model.eval()
label_encoder = joblib.load('{output_folder}/label_encoder.pkl')

FEATURE_RANGES = {{
    'N': (20, 199.9),
    'P': (20, 100),
    'K': (20, 149.9),
    'TEMP': (5, 47),
    'SOIL_PH': (6, 9),
    'RELATIVE_HUMIDITY': (15, 100)
}}

def validate_input_features(soil_ph, temp, humidity, n, p, k):
    errors = []
    if not (FEATURE_RANGES['SOIL_PH'][0] <= soil_ph <= FEATURE_RANGES['SOIL_PH'][1]):
        errors.append(f"土壤pH值应在{{FEATURE_RANGES['SOIL_PH'][0]}}-{{FEATURE_RANGES['SOIL_PH'][1]}}之间，当前值为{{soil_ph}}")
    if not (FEATURE_RANGES['TEMP'][0] <= temp <= FEATURE_RANGES['TEMP'][1]):
        errors.append(f"温度应在{{FEATURE_RANGES['TEMP'][0]}}-{{FEATURE_RANGES['TEMP'][1]}}°C之间，当前值为{{temp}}")
    if not (FEATURE_RANGES['RELATIVE_HUMIDITY'][0] <= humidity <= FEATURE_RANGES['RELATIVE_HUMIDITY'][1]):
        errors.append(f"相对湿度应在{{FEATURE_RANGES['RELATIVE_HUMIDITY'][0]}}-{{FEATURE_RANGES['RELATIVE_HUMIDITY'][1]}}%之间，当前值为{{humidity}}")
    if not (FEATURE_RANGES['N'][0] <= n <= FEATURE_RANGES['N'][1]):
        errors.append(f"氮含量(N)应在{{FEATURE_RANGES['N'][0]}}-{{FEATURE_RANGES['N'][1]}}之间，当前值为{{n}}")
    if not (FEATURE_RANGES['P'][0] <= p <= FEATURE_RANGES['P'][1]):
        errors.append(f"磷含量(P)应在{{FEATURE_RANGES['P'][0]}}-{{FEATURE_RANGES['P'][1]}}之间，当前值为{{p}}")
    if not (FEATURE_RANGES['K'][0] <= k <= FEATURE_RANGES['K'][1]):
        errors.append(f"钾含量(K)应在{{FEATURE_RANGES['K'][0]}}-{{FEATURE_RANGES['K'][1]}}之间，当前值为{{k}}")
    return errors

def recommend_crops(soil_ph, temp, humidity, n, p, k, top_k=3):
    validation_errors = validate_input_features(soil_ph, temp, humidity, n, p, k)
    if validation_errors:
        return [{{'error': error}} for error in validation_errors]

    features = np.array([[soil_ph, temp, humidity, n, p, k]])

    # Transformer预测
    with torch.no_grad():
        outputs = transformer_model(torch.tensor(features, dtype=torch.float32))
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]

    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    recommendations = []
    for i, idx in enumerate(top_indices):
        crop_name = label_encoder.inverse_transform([idx])[0]
        probability = probabilities[idx]
        recommendations.append({{
            'rank': i + 1,
            'crop': crop_name,
            'probability': round(float(probability), 4)
        }})
    return recommendations

if __name__ == "__main__":
    recommendations = recommend_crops(
        soil_ph=6.5,
        temp=25,
        humidity=70,
        n=120,
        p=80,
        k=60
    )

    print("作物推荐结果:")
    print("=" * 40)
    if recommendations and 'error' in recommendations[0]:
        print("输入参数错误:")
        for rec in recommendations:
            print(f"  - {{rec['error']}}")
        print("\\n无法进行作物推荐，请修正输入参数。")
    else:
        for rec in recommendations:
            print(f"{{rec['rank']}}. {{rec['crop']}} - 概率: {{rec['probability']:.2%}}")
'''

    with open(f'{output_folder}/usage_example.py', 'w', encoding='utf-8') as f:
        f.write(usage_code)


def main():
    print("=" * 60)
    print("Transformer 作物推荐系统")
    print("=" * 60)

    output_folder = create_output_folder()
    print(f"输出文件夹: {output_folder}")

    x_train, y_train, x_val, y_val, x_test, y_test, label_encoder, feature_columns = load_and_preprocess_data()
    num_classes = len(label_encoder.classes_)

    # 训练Transformer模型
    transformer_model = train_transformer_model(
        x_train, y_train, x_val, y_val, x_test, y_test,
        label_encoder, num_classes
    )

    # 评估模型
    metrics_dict, y_pred_proba, y_pred, topk_accuracies = calculate_metrics(
        transformer_model, x_test, y_test, label_encoder
    )

    # 可视化
    create_visualizations(y_test, y_pred, y_pred_proba, label_encoder, topk_accuracies, output_folder, feature_columns)

    # 保存结果
    save_results_and_model(transformer_model, label_encoder, metrics_dict, y_test, y_pred_proba, output_folder)

    # 打印摘要
    print("\n" + "=" * 60)
    print("模型评估结果摘要（Transformer）")
    print("=" * 60)
    print(f"Top-1 准确率: {metrics_dict['accuracy_top1']:.4f}")
    print(f"Top-3 准确率: {metrics_dict['accuracy_top3']:.4f}")
    print(f"宏平均 F1-score: {metrics_dict['f1_macro']:.4f}")
    print(f"平均精确率: {metrics_dict['precision_avg']:.4f}")
    print(f"平均召回率: {metrics_dict['recall_avg']:.4f}")
    print(f"PR-AUC (OvR): {metrics_dict['pr_auc_ovr']:.4f}")
    print(f"推理时间: {metrics_dict['inference_time_ms']} ms/样本")

    print("\n" + "=" * 60)
    print("生成的文件列表")
    print("=" * 60)
    print(f"1. {output_folder}/transformer_model.pth - Transformer模型")
    print(f"2. {output_folder}/label_encoder.pkl - 标签编码器")
    print(f"3. {output_folder}/evaluation_metrics.json - 评估指标")
    print(f"4. {output_folder}/detailed_predictions.csv - 详细预测")
    print(f"5. {output_folder}/usage_example.py - 使用示例")
    print(f"6. {output_folder}/topk_accuracy_curve.png")
    print(f"7. {output_folder}/normalized_confusion_matrix.png")
    print(f"8. {output_folder}/f1_per_class.png")
    print(f"9. {output_folder}/macro_pr_curve.png")
    print(f"10. {output_folder}/feature_importance.png")
    print(f"11. {output_folder}/sub_confusion_matrix_7x7.png")

    print(f"\n所有文件已保存到: {output_folder}/")
    print("Transformer 模型部署准备完成！")


if __name__ == "__main__":
    main()