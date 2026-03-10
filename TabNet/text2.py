import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
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
from sklearn.utils import class_weight
import torch

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12


def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在读取数据...")
    train_df = pd.read_csv('crop_train.csv')
    val_df = pd.read_csv('crop_val.csv')
    test_df = pd.read_csv('crop_test.csv')

    print(f"训练集大小: {train_df.shape}")
    print(f"验证集大小: {val_df.shape}")
    print(f"测试集大小: {test_df.shape}")

    # 特征列和目标列
    feature_columns = ['SOIL_PH', 'TEMP', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']
    target_column = 'CROPS'

    # 标签编码
    label_encoder = LabelEncoder()
    all_data = pd.concat([train_df, val_df, test_df])
    label_encoder.fit(all_data[target_column])

    # 准备特征和目标变量
    x_train = train_df[feature_columns].values
    y_train = label_encoder.transform(train_df[target_column])
    x_val = val_df[feature_columns].values
    y_val = label_encoder.transform(val_df[target_column])
    x_test = test_df[feature_columns].values
    y_test = label_encoder.transform(test_df[target_column])

    return x_train, y_train, x_val, y_val, x_test, y_test, label_encoder, feature_columns


def train_tabnet_model(x_train, y_train, x_val, y_val):
    """训练TabNet模型"""
    print("\n正在训练TabNet模型...")

    # 修复：创建每个样本的权重，而不是类别权重
    # 计算类别权重
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # 将类别权重转换为样本权重
    sample_weights = np.array([class_weights[label] for label in y_train])

    # 定义TabNet模型参数 - 根据你的数据规模调整参数
    tabnet_params = {
        'n_d': 8,  # 决策预测维度
        'n_a': 8,  # 注意力维度
        'n_steps': 3,  # 决策步骤数
        'gamma': 1.3,  # 特征重用系数
        'lambda_sparse': 1e-4,  # 稀疏性正则化
        'optimizer_fn': torch.optim.Adam,  # 优化器
        'optimizer_params': {'lr': 2e-2, 'weight_decay': 1e-5},  # 添加权重衰减
        'mask_type': 'sparsemax',  # 注意力mask类型
        'scheduler_params': {'step_size': 10, 'gamma': 0.9},
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'verbose': 10,  # 每10个epoch显示一次进度
        'seed': 42
    }

    # 创建TabNet分类器
    model = TabNetClassifier(**tabnet_params)

    # 训练模型
    print("开始训练TabNet模型...")
    start_time = time.time()

    model.fit(
        X_train=x_train, y_train=y_train,
        eval_set=[(x_val, y_val)],
        eval_name=['val'],
        eval_metric=['accuracy'],
        max_epochs=50,  # 减少epochs数量，避免过拟合
        patience=10,  # 减少早停轮数
        batch_size=256,  # 调整batch_size
        virtual_batch_size=64,
        weights=sample_weights,  # 使用样本权重
        drop_last=False
    )

    training_time = time.time() - start_time
    print(f"TabNet模型训练完成，耗时: {training_time:.2f}秒")

    return model


def calculate_metrics(model, x_test, y_test, label_encoder):
    """计算所有评估指标"""
    print("\n计算评估指标...")

    # 预测概率和推理时间测量
    start_time = time.time()
    y_pred_proba = model.predict_proba(x_test)
    inference_time_ms = (time.time() - start_time) * 1000 / len(x_test)

    y_pred = model.predict(x_test)

    # Top-k准确率计算
    topk_accuracies = {}
    for k in range(1, 6):
        correct = 0
        for i in range(len(y_test)):
            top_k_indices = np.argsort(y_pred_proba[i])[-k:]
            if y_test[i] in top_k_indices:
                correct += 1
        topk_accuracies[k] = correct / len(y_test)

    # 基础指标
    precision_avg = precision_score(y_test, y_pred, average='weighted')
    recall_avg = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # PR-AUC (One-vs-Rest) - 宏平均
    n_classes = len(label_encoder.classes_)
    pr_auc_scores = []

    # 对每个类别计算PR-AUC
    for class_idx in range(n_classes):
        y_true_binary = np.array(y_test == class_idx, dtype=int)
        y_score = y_pred_proba[:, class_idx]
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        pr_auc = auc(recall, precision)
        pr_auc_scores.append(pr_auc)

    pr_auc_ovr = np.mean(pr_auc_scores)  # 宏平均PR-AUC

    metrics_dict = {
        'accuracy_top1': float(round(topk_accuracies[1], 4)),
        'accuracy_top3': float(round(topk_accuracies[3], 4)),
        'f1_macro': float(round(f1_macro, 4)),
        'precision_avg': float(round(precision_avg, 4)),
        'recall_avg': float(round(recall_avg, 4)),
        'pr_auc_ovr': float(round(pr_auc_ovr, 4)),
        'inference_time_ms': float(round(inference_time_ms, 2))
    }

    return metrics_dict, y_pred_proba, y_pred, topk_accuracies


def create_visualizations(y_test, y_pred, y_pred_proba, label_encoder, topk_accuracies):
    """创建所有可视化图表"""
    print("\n生成可视化图表...")

    # 1. Top-k准确率曲线
    plt.figure(figsize=(10, 6))
    k_values = list(range(1, 6))
    accuracies = [topk_accuracies[k] for k in k_values]

    plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('k值 (Top-k)')
    plt.ylabel('准确率')
    plt.title('TabNet模型 - Top-k 准确率曲线')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    # 添加数值标签
    for k, acc in zip(k_values, accuracies):
        plt.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.savefig('topk_accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 归一化混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': '比例'})
    plt.title('TabNet模型 - 归一化混淆矩阵 (按行归一化)')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig('normalized_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 每类F1分数柱状图
    f1_per_class = f1_score(y_test, y_pred, average=None)
    plt.figure(figsize=(12, 6))
    classes = label_encoder.classes_
    x_pos = np.arange(len(classes))

    f1_min = min(f1_per_class)
    f1_max = max(f1_per_class)
    if f1_max > f1_min:
        cmap = plt.get_cmap('viridis')
        colors = [cmap((val - f1_min) / (f1_max - f1_min)) for val in f1_per_class]
    else:
        colors = ['skyblue'] * len(f1_per_class)

    bars = plt.bar(x_pos, f1_per_class, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('作物类别')
    plt.ylabel('F1分数')
    plt.title('TabNet模型 - 每类作物F1分数')
    plt.xticks(x_pos, classes, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.1)

    # 添加数值标签
    for bar, value in zip(bars, f1_per_class):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    # 添加平均线
    mean_f1 = np.mean(f1_per_class)
    plt.axhline(y=mean_f1, color='red', linestyle='--', alpha=0.8,
                label=f'平均F1: {mean_f1:.3f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('f1_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. PR曲线（多分类，每类一条曲线）
    n_classes = len(label_encoder.classes_)
    plt.figure(figsize=(10, 8))

    for class_idx in range(n_classes):
        y_true_binary = np.array(y_test == class_idx, dtype=int)
        y_score = y_pred_proba[:, class_idx]

        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, lw=2,
                 label=f'{label_encoder.classes_[class_idx]} (AUC = {pr_auc:.3f})')

    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('TabNet模型 - 多分类PR曲线 (每类别)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig('pr_curve_per_class.png', dpi=300, bbox_inches='tight')  # 修复这行
    plt.close()

    # 5. 宏平均PR曲线
    plt.figure(figsize=(10, 8))

    precision_macro = []
    for class_idx in range(n_classes):
        y_true_binary = np.array(y_test == class_idx, dtype=int)
        y_score = y_pred_proba[:, class_idx]
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)

        recall_interp = np.linspace(0, 1, 100)
        precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1])
        precision_macro.append(precision_interp)

    precision_mean = np.mean(precision_macro, axis=0)
    recall_mean = np.linspace(0, 1, 100)
    pr_auc_mean = auc(recall_mean, precision_mean)

    plt.plot(recall_mean, precision_mean, lw=3, color='red',
             label=f'宏平均PR曲线 (AUC = {pr_auc_mean:.3f})')

    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('TabNet模型 - 宏平均PR曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig('macro_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results_and_model(model, label_encoder, metrics_dict, y_test, y_pred_proba):
    """保存结果和模型"""
    print("\n保存结果和模型...")

    metrics_with_descriptions = {
        'accuracy_top1': {
            'value': metrics_dict['accuracy_top1'],
            'description': 'Top-1 准确率 - 基础性能'
        },
        'accuracy_top3': {
            'value': metrics_dict['accuracy_top3'],
            'description': 'Top-3 准确率 - 实际推荐体验'
        },
        'f1_macro': {
            'value': metrics_dict['f1_macro'],
            'description': '宏平均 F1-score - 关注小众作物表现'
        },
        'precision_avg': {
            'value': metrics_dict['precision_avg'],
            'description': '平均精确率（weighted）- 避免错误推荐'
        },
        'recall_avg': {
            'value': metrics_dict['recall_avg'],
            'description': '平均召回率（weighted）- 避免漏掉合适作物'
        },
        'pr_auc_ovr': {
            'value': metrics_dict['pr_auc_ovr'],
            'description': 'PR-AUC (OvR) - 精确率-召回率平衡'
        },
        'inference_time_ms': {
            'value': metrics_dict['inference_time_ms'],
            'description': '单样本推理时间（毫秒）- 部署效率'
        }
    }

    json_str = json.dumps(metrics_with_descriptions, indent=4, ensure_ascii=False)
    with open('evaluation_metrics.json', 'w', encoding='utf-8') as file_obj:
        file_obj.write(json_str)

    # 保存模型和编码器
    model.save_model('crop_recommendation_tabnet_model.zip')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # 创建详细的预测结果文件
    test_df = pd.read_csv('crop_test.csv')
    results_df = test_df.copy()
    results_df['True_Crop'] = label_encoder.inverse_transform(y_test)
    results_df['Predicted_Crop'] = label_encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))
    results_df['Prediction_Probability'] = np.max(y_pred_proba, axis=1)

    # 添加Top-3推荐
    top3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
    top3_probs = np.sort(y_pred_proba, axis=1)[:, -3:][:, ::-1]

    top3_crops = []
    top3_probabilities = []

    for i in range(len(top3_indices)):
        crops = [label_encoder.classes_[idx] for idx in top3_indices[i]]
        probs = top3_probs[i]
        top3_crops.append(crops)
        top3_probabilities.append(probs)

    results_df['Top3_Recommendations'] = top3_crops
    results_df['Top3_Probabilities'] = top3_probabilities

    results_df.to_csv('detailed_predictions.csv', index=False, encoding='utf-8-sig')
    create_usage_example()


def create_usage_example():
    """创建使用示例"""
    usage_code = '''# 作物推荐系统使用示例（TabNet模型）
import joblib
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

# 加载模型和编码器
model = TabNetClassifier()
model.load_model('crop_recommendation_tabnet_model.zip')
label_encoder = joblib.load('label_encoder.pkl')

def recommend_crops(soil_ph, temp, humidity, n, p, k, top_k=3):
    """
    推荐最适合的作物

    参数:
    soil_ph: 土壤pH值
    temp: 温度
    humidity: 相对湿度
    n: 氮含量
    p: 磷含量
    k: 钾含量
    top_k: 返回前K个推荐 (默认3)

    返回:
    recommendations: 推荐作物列表，包含作物名和概率
    """
    # 准备输入特征
    features = np.array([[soil_ph, temp, humidity, n, p, k]])

    # 预测概率
    probabilities = model.predict_proba(features)[0]

    # 获取Top-K推荐
    top_indices = np.argsort(probabilities)[-top_k:][::-1]

    recommendations = []
    for i, idx in enumerate(top_indices):
        crop_name = label_encoder.inverse_transform([idx])[0]
        probability = probabilities[idx]
        recommendations.append({
            'rank': i + 1,
            'crop': crop_name,
            'probability': round(float(probability), 4)
        })

    return recommendations

# 使用示例
if __name__ == "__main__":
    # 示例输入
    recommendations = recommend_crops(
        soil_ph=6.5,      # 土壤pH
        temp=25,          # 温度 (°C)
        humidity=70,      # 相对湿度 (%)
        n=120,            # 氮含量
        p=80,             # 磷含量
        k=60              # 钾含量
    )

    print("TabNet模型作物推荐结果:")
    print("=" * 50)
    for rec in recommendations:
        print(f"{rec['rank']}. {rec['crop']} - 概率: {rec['probability']:.2%}")
'''

    with open('usage_example.py', 'w', encoding='utf-8') as file_obj:
        file_obj.write(usage_code)


def main():
    """主函数"""
    print("=" * 60)
    print("TabNet作物推荐系统")
    print("=" * 60)

    # 1. 加载和预处理数据
    x_train, y_train, x_val, y_val, x_test, y_test, label_encoder, feature_columns = load_and_preprocess_data()

    # 2. 训练TabNet模型
    model = train_tabnet_model(x_train, y_train, x_val, y_val)

    # 3. 计算指标
    metrics_dict, y_pred_proba, y_pred, topk_accuracies = calculate_metrics(
        model, x_test, y_test, label_encoder)

    # 4. 创建可视化
    create_visualizations(y_test, y_pred, y_pred_proba, label_encoder, topk_accuracies)

    # 5. 保存结果和模型
    save_results_and_model(model, label_encoder, metrics_dict, y_test, y_pred_proba)

    # 6. 打印结果摘要
    print("\n" + "=" * 60)
    print("TabNet模型评估结果摘要")
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
    print("1. crop_recommendation_tabnet_model.zip - 训练好的TabNet模型")
    print("2. label_encoder.pkl - 标签编码器")
    print("3. evaluation_metrics.json - 评估指标文件")
    print("4. detailed_predictions.csv - 详细预测结果")
    print("5. usage_example.py - 使用示例代码")
    print("6. topk_accuracy_curve.png - Top-k准确率曲线")
    print("7. normalized_confusion_matrix.png - 归一化混淆矩阵")
    print("8. f1_per_class.png - 每类F1分数柱状图")
    print("9. pr_curve_per_class.png - 每类PR曲线")
    print("10. macro_pr_curve.png - 宏平均PR曲线")

    print("\nTabNet模型系统部署准备完成！")


if __name__ == "__main__":
    main()