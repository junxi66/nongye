# 作物推荐系统使用示例（CatBoost 0.7 + Transformer 0.3 融合）
import joblib
import torch
import numpy as np
import torch.nn as nn

# 定义优化的Transformer模型结构
class OptimizedTransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout_rate=0.2):
        super(OptimizedTransformerClassifier, self).__init__()
        nhead = 4
        if hidden_size % nhead != 0:
            hidden_size = (hidden_size // nhead) * nhead
            if hidden_size < nhead:
                hidden_size = nhead
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=nhead, 
                dim_feedforward=hidden_size*2,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.dropout(x.squeeze(1))
        x = self.fc(x)
        return x

# 加载模型和编码器
catboost_model = joblib.load('catboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 加载Transformer模型
transformer_model = OptimizedTransformerClassifier(
    input_size=6, 
    hidden_size=64,
    num_classes=len(label_encoder.classes_)
)
transformer_model.load_state_dict(torch.load('transformer_model.pth'))
transformer_model.eval()

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

    features = np.array([[soil_ph, temp, humidity, n, p, k]])

    # 融合预测（CatBoost 0.7 + Transformer 0.3）
    catboost_proba = catboost_model.predict_proba(features)[0]
    transformer_proba = transformer_model(torch.tensor(features, dtype=torch.float32).unsqueeze(1)).detach().numpy()[0]
    fused_proba = 0.7 * catboost_proba + 0.3 * transformer_proba

    top_indices = np.argsort(fused_proba)[-top_k:][::-1]
    recommendations = []
    for i, idx in enumerate(top_indices):
        crop_name = label_encoder.inverse_transform([idx])[0]
        score = int(round(fused_proba[idx] * 100))  # 转为0-100评分
        recommendations.append({
            'rank': i + 1,
            'crop': crop_name,
            'score': score  # 0-100评分
        })
    return recommendations

if __name__ == "__main__":
    recommendations = recommend_crops(
        soil_ph=6.5,
        temp=25,
        humidity=70,
        n=40,
        p=80,
        k=60
    )

    print("作物推荐结果:")
    print("=" * 40)
    if recommendations and 'error' in recommendations[0]:
        print("输入参数错误:")
        for rec in recommendations:
            print(f"  - {rec['error']}")
        print("\n无法进行作物推荐，请修正输入参数。")
    else:
        for rec in recommendations:
            print(f"{rec['rank']}. {rec['crop']} - 评分: {rec['score']}/100")
