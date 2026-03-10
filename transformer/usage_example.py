# 作物推荐系统使用示例（Transformer）
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
transformer_model.load_state_dict(torch.load('transformer/transformer_model.pth'))
transformer_model.eval()
label_encoder = joblib.load('transformer/label_encoder.pkl')

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

    # Transformer预测
    with torch.no_grad():
        outputs = transformer_model(torch.tensor(features, dtype=torch.float32))
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]

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
            print(f"  - {rec['error']}")
        print("\n无法进行作物推荐，请修正输入参数。")
    else:
        for rec in recommendations:
            print(f"{rec['rank']}. {rec['crop']} - 概率: {rec['probability']:.2%}")
