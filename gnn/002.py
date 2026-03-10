import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import time
import warnings
import matplotlib
import os
from datetime import datetime

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"crop_recommendation_gnn_output_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def validate_input_features(soil_ph, temp, humidity, n, p, k):
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
    print("正在读取数据...")
    train_df = pd.read_csv('../crop_train.csv', encoding='utf-8-sig')
    val_df = pd.read_csv('../crop_val.csv', encoding='utf-8-sig')
    test_df = pd.read_csv('../crop_test.csv', encoding='utf-8-sig')

    feature_columns = ['SOIL_PH', 'TEMP', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']

    # 1. 确保所有特征列都是数值类型
    for df in [train_df, val_df, test_df]:
        # 尝试将所有特征列转换为数值类型
        for col in feature_columns:
            # 尝试转换为浮点数，非数值转为NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. 填充缺失值（用各列均值）
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())

        # 3. 确保列类型为float（关键修复）
        for col in feature_columns:
            df[col] = df[col].astype(float)

    print(f"训练集大小: {train_df.shape}")
    print(f"验证集大小: {val_df.shape}")
    print(f"测试集大小: {test_df.shape}")

    target_column = 'CROPS'

    label_encoder = LabelEncoder()
    all_data = pd.concat([train_df, val_df, test_df])
    label_encoder.fit(all_data[target_column])

    return train_df, val_df, test_df, label_encoder, feature_columns


def df_to_pyg_data(df, feature_columns, labels=None):
    """将DataFrame转换为PyG单节点图列表"""
    data_list = []
    for idx, row in df.iterrows():
        # 确保特征是数值类型
        features = row[feature_columns].values.astype(float)
        x = torch.tensor(features, dtype=torch.float).unsqueeze(0)  # [1, num_features]
        # 每个样本是一个孤立节点，无边
        edge_index = torch.empty((2, 0), dtype=torch.long)
        if labels is not None:
            y = torch.tensor([labels[idx]], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y=y)
        else:
            data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
    return data_list


class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        if num_layers >= 2:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        # 全局平均池化（虽然每个图只有一个节点，但保留接口）
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


def train_gnn_model(train_loader, val_loader, num_classes, device, epochs=100, lr=0.001):
    print("\n正在训练GNN模型...")
    model = GNNClassifier(in_channels=6, hidden_channels=64, num_classes=num_classes, num_layers=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 20
    trigger_times = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        # 验证
        model.eval()
        correct = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
        val_acc = correct / len(val_loader.dataset)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
            torch.save(model.state_dict(), 'best_gnn_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader.dataset):.4f}, Val Acc: {val_acc:.4f}')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_gnn_model.pth', map_location=device))
    os.remove('best_gnn_model.pth')  # 清理临时文件
    return model


def calculate_metrics_gnn(model, test_loader, label_encoder, device):
    print("\n计算评估指标...")
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            start_time = time.time()
            out = model(data.x, data.edge_index, data.batch)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / data.num_graphs)

            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    y_test = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_pred_proba = np.array(all_probs)
    avg_inference_time_ms = np.mean(inference_times) * 1000

    # Top-k准确率
    topk_accuracies = {}
    for k in range(1, 6):
        correct = 0
        for i in range(len(y_test)):
            top_k_indices = np.argsort(y_pred_proba[i])[-k:]
            if y_test[i] in top_k_indices:
                correct += 1
        topk_accuracies[k] = correct / len(y_test)

    precision_avg = precision_score(y_test, y_pred, average='weighted')
    recall_avg = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    n_classes = len(label_encoder.classes_)
    pr_auc_scores = []
    for class_idx in range(n_classes):
        y_true_binary = (y_test == class_idx).astype(int)
        y_score = y_pred_proba[:, class_idx]
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        pr_auc = auc(recall, precision)
        pr_auc_scores.append(pr_auc)
    pr_auc_ovr = np.mean(pr_auc_scores)

    metrics_dict = {
        'accuracy_top1': float(round(topk_accuracies[1], 4)),
        'accuracy_top3': float(round(topk_accuracies[3], 4)),
        'f1_macro': float(round(f1_macro, 4)),
        'precision_avg': float(round(precision_avg, 4)),
        'recall_avg': float(round(recall_avg, 4)),
        'pr_auc_ovr': float(round(pr_auc_ovr, 4)),
        'inference_time_ms': float(round(avg_inference_time_ms, 2))
    }

    return metrics_dict, y_pred_proba, y_pred, topk_accuracies, y_test


def create_visualizations(y_test, y_pred, y_pred_proba, label_encoder, topk_accuracies, output_folder):
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
    colors = plt.get_cmap('viridis')((f1_per_class - f1_min) / (f1_max - f1_min)) if f1_max > f1_min else [
                                                                                                              'skyblue'] * len(
        f1_per_class)
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

    # 4. PR曲线（每类）
    n_classes = len(label_encoder.classes_)
    plt.figure(figsize=(10, 8))
    for class_idx in range(n_classes):
        y_true_binary = (y_test == class_idx).astype(int)
        y_score = y_pred_proba[:, class_idx]
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{label_encoder.classes_[class_idx]} (AUC = {pr_auc:.3f})')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('多分类PR曲线 (每类别)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(f'{output_folder}/pr_curve_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 宏平均PR曲线
    plt.figure(figsize=(10, 8))
    precision_macro = []
    for class_idx in range(n_classes):
        y_true_binary = (y_test == class_idx).astype(int)
        y_score = y_pred_proba[:, class_idx]
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        recall_interp = np.linspace(0, 1, 100)
        precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1])
        precision_macro.append(precision_interp)
    precision_mean = np.mean(precision_macro, axis=0)
    recall_mean = np.linspace(0, 1, 100)
    pr_auc_mean = auc(recall_mean, precision_mean)
    plt.plot(recall_mean, precision_mean, lw=3, color='red', label=f'宏平均PR曲线 (AUC = {pr_auc_mean:.3f})')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('宏平均PR曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(f'{output_folder}/macro_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results_and_model(model, label_encoder, metrics_dict, y_test, y_pred_proba, output_folder, device):
    print("\n保存结果和模型...")

    # 保存指标
    metrics_with_descriptions = {
        'accuracy_top1': {'value': metrics_dict['accuracy_top1'], 'description': 'Top-1 准确率'},
        'accuracy_top3': {'value': metrics_dict['accuracy_top3'], 'description': 'Top-3 准确率'},
        'f1_macro': {'value': metrics_dict['f1_macro'], 'description': '宏平均 F1-score'},
        'precision_avg': {'value': metrics_dict['precision_avg'], 'description': '平均精确率（weighted）'},
        'recall_avg': {'value': metrics_dict['recall_avg'], 'description': '平均召回率（weighted）'},
        'pr_auc_ovr': {'value': metrics_dict['pr_auc_ovr'], 'description': 'PR-AUC (OvR)'},
        'inference_time_ms': {'value': metrics_dict['inference_time_ms'], 'description': '单样本推理时间（毫秒）'}
    }
    with open(f'{output_folder}/evaluation_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_with_descriptions, f, indent=4, ensure_ascii=False)

    # 保存模型（整个模型）
    torch.save(model.state_dict(), f'{output_folder}/crop_recommendation_gnn_model.pth')
    joblib.dump(label_encoder, f'{output_folder}/label_encoder.pkl')

    # 保存详细预测（需重新加载测试数据）
    test_df = pd.read_csv('../crop_test.csv', encoding='utf-8-sig')
    results_df = test_df.copy()
    results_df['True_Crop'] = label_encoder.inverse_transform(y_test)
    results_df['Predicted_Crop'] = label_encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))
    results_df['Prediction_Probability'] = np.max(y_pred_proba, axis=1)

    top3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
    top3_probs = np.sort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
    top3_crops = [[label_encoder.classes_[idx] for idx in row] for row in top3_indices]
    top3_probabilities = [list(row) for row in top3_probs]

    results_df['Top3_Recommendations'] = top3_crops
    results_df['Top3_Probabilities'] = top3_probabilities
    results_df.to_csv(f'{output_folder}/detailed_predictions.csv', index=False, encoding='utf-8-sig')

    create_usage_example_gnn(output_folder)


def create_usage_example_gnn(output_folder):
    usage_code = f'''# GNN作物推荐系统使用示例
import torch
import torch.nn.functional as F
import joblib
import numpy as np
from torch_geometric.data import Data

# 加载模型和编码器
label_encoder = joblib.load('{output_folder}/label_encoder.pkl')

# 定义GNN模型结构（必须与训练时一致）
class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2):
        super().__init__()
        from torch_geometric.nn import SAGEConv, global_mean_pool
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        if num_layers >= 2:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)
        self.global_mean_pool = global_mean_pool

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.global_mean_pool(x, batch)
        return self.classifier(x)

# 初始化并加载权重
device = torch.device('cpu')
model = GNNClassifier(in_channels=6, hidden_channels=64, num_classes=len(label_encoder.classes_), num_layers=2)
model.load_state_dict(torch.load('{output_folder}/crop_recommendation_gnn_model.pth', map_location=device))
model.eval()

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

    # 构造单节点图
    x = torch.tensor([[soil_ph, temp, humidity, n, p, k]], dtype=torch.float)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.zeros(1, dtype=torch.long)  # batch index for the single node

    with torch.no_grad():
        out = model(x, edge_index, batch)
        probabilities = F.softmax(out, dim=1).cpu().numpy()[0]

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
    recommendations = recommend_crops(soil_ph=6.5, temp=25, humidity=70, n=120, p=80, k=60)
    print("作物推荐结果:")
    print("=" * 40)
    if recommendations and 'error' in recommendations[0]:
        print("输入参数错误:")
        for rec in recommendations:
            print(f"  - {{rec['error']}}")
    else:
        for rec in recommendations:
            print(f"{{rec['rank']}}. {{rec['crop']}} - 概率: {{rec['probability']:.2%}}")
'''
    with open(f'{output_folder}/usage_example.py', 'w', encoding='utf-8') as f:
        f.write(usage_code)


def main():
    print("=" * 60)
    print("GNN作物推荐系统")
    print("=" * 60)

    output_folder = create_output_folder()
    print(f"输出文件夹: {output_folder}")

    # 加载数据
    train_df, val_df, test_df, label_encoder, feature_columns = load_and_preprocess_data()

    # 编码标签
    y_train = label_encoder.transform(train_df['CROPS'])
    y_val = label_encoder.transform(val_df['CROPS'])
    y_test = label_encoder.transform(test_df['CROPS'])

    # 转换为PyG数据
    train_data = df_to_pyg_data(train_df, feature_columns, y_train)
    val_data = df_to_pyg_data(val_df, feature_columns, y_val)
    test_data = df_to_pyg_data(test_df, feature_columns, y_test)

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 训练GNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(label_encoder.classes_)
    model = train_gnn_model(train_loader, val_loader, num_classes, device, epochs=100, lr=0.001)

    # 评估
    metrics_dict, y_pred_proba, y_pred, topk_accuracies, y_test_array = calculate_metrics_gnn(
        model, test_loader, label_encoder, device)

    # 可视化
    create_visualizations(y_test_array, y_pred, y_pred_proba, label_encoder, topk_accuracies, output_folder)

    # 保存
    save_results_and_model(model, label_encoder, metrics_dict, y_test_array, y_pred_proba, output_folder, device)

    # 打印结果
    print("\n" + "=" * 60)
    print("模型评估结果摘要")
    print("=" * 60)
    print(f"Top-1 准确率: {metrics_dict['accuracy_top1']:.4f}")
    print(f"Top-3 准确率: {metrics_dict['accuracy_top3']:.4f}")
    print(f"宏平均 F1-score: {metrics_dict['f1_macro']:.4f}")
    print(f"平均精确率: {metrics_dict['precision_avg']:.4f}")
    print(f"平均召回率: {metrics_dict['recall_avg']:.4f}")
    print(f"PR-AUC (OvR): {metrics_dict['pr_auc_ovr']:.4f}")
    print(f"推理时间: {metrics_dict['inference_time_ms']} ms/样本")

    print("\n所有文件已保存到:", output_folder)


if __name__ == "__main__":
    main()