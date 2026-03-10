# 作物推荐系统使用示例（KAN 模型）
import joblib
import torch
import numpy as np
from kan import KAN

# 加载模型和编码器
kan_model = KAN(width=[6, 10, 10])  # 注意：输出维度需与训练时一致
kan_model.loadckpt('KAN/kan_model.pt')
label_encoder = joblib.load('KAN/label_encoder.pkl')

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
