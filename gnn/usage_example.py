# GNN作物推荐系统使用示例
import torch
import torch.nn.functional as F
import joblib
import numpy as np
from torch_geometric.data import Data

# 加载模型和编码器
label_encoder = joblib.load('gnn/label_encoder.pkl')

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
model.load_state_dict(torch.load('gnn/crop_recommendation_gnn_model.pth', map_location=device))
model.eval()

FEATURE_RANGES = {
    'N': (20, 199.9),
    'P': (20, 100),
    'K': (20, 149.9),
    'TEMP': (5, 47),
    'SOIL_PH': (6, 9),
    'RELATIVE_HUMIDITY': (15, 100)
}

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

def recommend_crops(soil_ph, temp, humidity, n, p, k, top_k=3):
    validation_errors = validate_input_features(soil_ph, temp, humidity, n, p, k)
    if validation_errors:
        return [{'error': error} for error in validation_errors]

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
        recommendations.append({
            'rank': i + 1,
            'crop': crop_name,
            'probability': round(float(probability), 4)
        })
    return recommendations

if __name__ == "__main__":
    recommendations = recommend_crops(soil_ph=6.5, temp=25, humidity=70, n=120, p=80, k=60)
    print("作物推荐结果:")
    print("=" * 40)
    if recommendations and 'error' in recommendations[0]:
        print("输入参数错误:")
        for rec in recommendations:
            print(f"  - {rec['error']}")
    else:
        for rec in recommendations:
            print(f"{rec['rank']}. {rec['crop']} - 概率: {rec['probability']:.2%}")
