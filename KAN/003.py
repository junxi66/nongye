import pandas as pd
import numpy as np
from kan import KAN  # ✅ pykan 0.2.8 使用 KAN（多分类时实际是 MultKAN）
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
import torch

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


def create_output_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"crop_recommendation_output_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def validate_input_features(soil_ph, temp, humidity, n, p, k):
    errors = []
    if not (FEATURE_RANGES['SOIL_PH'][0] <= soil_ph <= FEATURE_RANGES['SOIL_PH'][1]):
        errors.append(f"土壤pH值应在{FEATURE_RANGES['SOIL_PH'][0]}-{FEATURE_RANGES['SOIL_PH'][1]}之间，当前值为{soil_ph}")
    if not (FEATURE_RANGES['TEMP'][0] <= temp <= FEATURE_RANGES['TEMP'][1]):
        errors.append(f"温度应在{FEATURE_RANGES['TEMP'][0]}-{FEATURE_RANGES['TEMP'][1]}°C之间，当前值为{temp}")
    if not (FEATURE_RANGES['RELATIVE_HUMIDITY'][0] <= humidity <= FEATURE_RANGES['RELATIVE_HUMIDITY'][1]):
        errors.append(f"相对湿度应在{FEATURE_RANGES['RELATIVE_HUMIDITY'][0]}-{FEATURE_RANGES['RELATIVE_HUMIDITY'][1]}%之间，当前值为{humidity}")
    if not (FEATURE_RANGES['N'][0] <= n <= FEATURE_RANGES['N'][1]):
        errors.append(f"氮含量(N)应在{FEATURE_RANGES['N'][0]}-{FEATURE_RANGES['N'][1]}之间，当前值为{n}")
    if not (FEATURE_RANGES['P'][0] <= p <= FEATURE_RANGES['P'][1]):
        errors.append(f"磷含量(P)应在{FEATURE_RANGES['P'][0]}-{FEATURE_RANGES['P'][1]}之间，当前值为{p}")
    if not (FEATURE_RANGES['K'][0] <= k <= FEATURE_RANGES['K'][1]):
        errors.append(f"钾含量(K)应在{FEATURE_RANGES['K'][0]}-{FEATURE_RANGES['K'][1]}之间，当前值为{k}")
    return errors


def load_and_preprocess_data():
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

    x_train = train_df[feature_columns].values.astype(np.float32)
    y_train = label_encoder.transform(train_df[target_column])
    x_val = val_df[feature_columns].values.astype(np.float32)
    y_val = label_encoder.transform(val_df[target_column])
    x_test = test_df[feature_columns].values.astype(np.float32)
    y_test = label_encoder.transform(test_df[target_column])

    return x_train, y_train, x_val, y_val, x_test, y_test, label_encoder, feature_columns


def train_kan_model(x_train, y_train, x_val, y_val, num_classes):
    """训练 KAN 模型（多分类时实际为 MultKAN，必须传入字典格式 dataset）"""
    print("正在训练 KAN 模型...")

    # 转换为 PyTorch 张量
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # 构建 KAN 网络：6 → 10 → num_classes
    model = KAN(width=[6, 10, num_classes], grid=3, k=3, seed=42)

    # ⚠️ 关键修复：MultKAN.fit() 必须传入字典！
    dataset = {
        'train_input': x_train_tensor,
        'train_label': y_train_tensor,
        'test_input': x_val_tensor,
        'test_label': y_val_tensor
    }

    model.fit(
        dataset,
        steps=100,
        stop_grid_update_step=50,
        loss_fn=torch.nn.CrossEntropyLoss()
    )

    # 手动验证（训练后评估）
    model.eval()
    with torch.no_grad():
        val_logits = model(x_val_tensor)
        val_pred = val_logits.argmax(dim=1)
        val_acc = (val_pred == y_val_tensor).float().mean().item()
    print(f"✅ 验证集准确率: {val_acc:.4f}")

    return model


def calculate_metrics(kan_model, x_test, y_test, label_encoder):
    print("\n计算评估指标（KAN 模型）...")

    start_time = time.time()
    test_tensor = torch.tensor(x_test, dtype=torch.float32)

    with torch.no_grad():
        logits = kan_model(test_tensor)
        proba = torch.softmax(logits, dim=1).numpy()

    inference_time_ms = (time.time() - start_time) * 1000 / len(x_test)
    y_pred = np.argmax(proba, axis=1)

    # Top-k 准确率
    topk_accuracies = {}
    for k in range(1, 6):
        correct = 0
        for i in range(len(y_test)):
            top_k_indices = np.argsort(proba[i])[-k:]
            if y_test[i] in top_k_indices:
                correct += 1
        topk_accuracies[k] = correct / len(y_test)

    precision_avg = precision_score(y_test, y_pred, average='weighted')
    recall_avg = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # PR-AUC (OvR)
    n_classes = len(label_encoder.classes_)
    pr_auc_scores = []
    for class_idx in range(n_classes):
        y_true_binary = np.array(y_test == class_idx, dtype=int)
        y_score = proba[:, class_idx]
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

    return metrics_dict, proba, y_pred, topk_accuracies


def create_visualizations(y_test, y_pred, y_pred_proba, label_encoder, topk_accuracies, output_folder, kan_model,
                          feature_columns):
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
    colors = [plt.get_cmap('viridis')((val - f1_min) / (f1_max - f1_min)) if f1_max > f1_min else 'skyblue' for val in f1_per_class]
    bars = plt.bar(x_pos, f1_per_class, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('作物类别')
    plt.ylabel('F1分数')
    plt.title('每类作物F1分数')
    plt.xticks(x_pos, classes, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.1)
    for bar, value in zip(bars, f1_per_class):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    mean_f1 = np.mean(f1_per_class)
    plt.axhline(y=mean_f1, color='red', linestyle='--', alpha=0.8, label=f'平均F1: {mean_f1:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_folder}/f1_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 宏平均PR曲线
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
    plt.axhline(y=avg_pos_ratio, color='gray', linestyle='--', lw=2, label=f'随机基线 (平均正例率 = {avg_pos_ratio:.3f})')
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

    # 5. 7x7 子混淆矩阵
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


def save_results_and_model(kan_model, label_encoder, metrics_dict, y_test, y_pred_proba, output_folder):
    print("\n保存结果和模型...")

    metrics_with_descriptions = {
        'accuracy_top1': {'value': metrics_dict['accuracy_top1'], 'description': 'Top-1 准确率 - 基础性能'},
        'accuracy_top3': {'value': metrics_dict['accuracy_top3'], 'description': 'Top-3 准确率 - 实际推荐体验'},
        'f1_macro': {'value': metrics_dict['f1_macro'], 'description': '宏平均 F1-score - 关注小众作物表现'},
        'precision_avg': {'value': metrics_dict['precision_avg'], 'description': '平均精确率（weighted）- 避免错误推荐'},
        'recall_avg': {'value': metrics_dict['recall_avg'], 'description': '平均召回率（weighted）- 避免漏掉合适作物'},
        'pr_auc_ovr': {'value': metrics_dict['pr_auc_ovr'], 'description': 'PR-AUC (OvR) - 精确率-召回率平衡'},
        'inference_time_ms': {'value': metrics_dict['inference_time_ms'], 'description': '单样本推理时间（毫秒）- 部署效率'}
    }

    with open(f'{output_folder}/evaluation_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_with_descriptions, f, indent=4, ensure_ascii=False)

    # 保存 KAN 模型
    kan_model.saveckpt(f'{output_folder}/kan_model.pt')
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
    usage_code = f'''# 作物推荐系统使用示例（KAN 模型）
import joblib
import torch
import numpy as np
from kan import KAN

# 加载模型和编码器
kan_model = KAN(width=[6, 10, 10])  # 注意：输出维度需与训练时一致
kan_model.loadckpt('{output_folder}/kan_model.pt')
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

    features = np.array([[soil_ph, temp, humidity, n, p, k]], dtype=np.float32)
    features_tensor = torch.tensor(features)

    with torch.no_grad():
        logits = kan_model(features_tensor)
        proba = torch.softmax(logits, dim=1).numpy()[0]

    top_indices = np.argsort(proba)[-top_k:][::-1]
    recommendations = []
    for i, idx in enumerate(top_indices):
        crop_name = label_encoder.inverse_transform([idx])[0]
        probability = proba[idx]
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
    print("KAN 单独模型作物推荐系统")
    print("=" * 60)

    output_folder = create_output_folder()
    print(f"输出文件夹: {output_folder}")

    x_train, y_train, x_val, y_val, x_test, y_test, label_encoder, feature_columns = load_and_preprocess_data()
    num_classes = len(label_encoder.classes_)

    kan_model = train_kan_model(x_train, y_train, x_val, y_val, num_classes)

    metrics_dict, y_pred_proba, y_pred, topk_accuracies = calculate_metrics(
        kan_model, x_test, y_test, label_encoder)

    create_visualizations(y_test, y_pred, y_pred_proba, label_encoder, topk_accuracies, output_folder, kan_model,
                          feature_columns)

    save_results_and_model(kan_model, label_encoder, metrics_dict, y_test, y_pred_proba, output_folder)

    print("\n" + "=" * 60)
    print("模型评估结果摘要（KAN 模型）")
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
    files = [
        "kan_model.pt", "label_encoder.pkl", "evaluation_metrics.json",
        "detailed_predictions.csv", "usage_example.py",
        "topk_accuracy_curve.png", "normalized_confusion_matrix.png",
        "f1_per_class.png", "macro_pr_curve.png", "sub_confusion_matrix_7x7.png"
    ]
    for i, f in enumerate(files, 1):
        print(f"{i}. {output_folder}/{f}")

    print(f"\n所有文件已保存到: {output_folder}/")
    print("KAN 模型部署准备完成！")


if __name__ == "__main__":
    main()