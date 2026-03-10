# 作物推荐系统使用示例（TabNet模型）
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
