import streamlit as st
import joblib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import requests
from datetime import datetime

# 设置页面
st.set_page_config(
    page_title="🌱 智能农业助手",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background-color: #f0f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .parameter-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .error-message {
        background-color: #ffe6e6;
        color: #d63031;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #d63031;
    }
    .success-message {
        background-color: #e6f7e6;
        color: #2E8B57;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

# 定义特征范围（与训练时一致）
FEATURE_RANGES = {
    'N': (20, 199.9),
    'P': (20, 100),
    'K': (20, 149.9),
    'TEMP': (5, 47),
    'SOIL_PH': (6, 9),
    'RELATIVE_HUMIDITY': (15, 100)
}


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
                dim_feedforward=hidden_size * 2,
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


# 加载模型函数
@st.cache_resource
def load_models():
    """加载预训练的模型"""
    try:
        # 加载CatBoost模型
        catboost_model = joblib.load('CatTrans/catboost_model.pkl')

        # 加载标签编码器
        label_encoder = joblib.load('CatTrans/label_encoder.pkl')

        # 加载Transformer模型
        transformer_model = OptimizedTransformerClassifier(
            input_size=6,
            hidden_size=64,
            num_classes=len(label_encoder.classes_)
        )
        transformer_model.load_state_dict(torch.load('CatTrans/transformer_model.pth', map_location='cpu'))
        transformer_model.eval()

        return catboost_model, transformer_model, label_encoder
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.info("请确保以下文件在当前目录中：catboost_model.pkl, transformer_model.pth, label_encoder.pkl")
        return None, None, None


# 输入验证函数
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


# 天气功能模块
def get_weather_location():
    """使用百度地图API获取当前位置"""
    try:
        baidu_ak = "WC6s35CQwWvjpg5txGj20ricDgPLp2US"
        url = f"https://api.map.baidu.com/location/ip?ak={baidu_ak}&coor=bd09ll"
        response = requests.get(url)
        data = response.json()
        
        if data.get('status') == 0:
            return data['content']['address_detail']['city']
        else:
            return None
    except:
        return None

def get_current_weather(city_name):
    """获取当前天气"""
    try:
        seniverse_key = "SEH8S1zMkjgD49Bsi"
        # 使用3天预报API获取今天的详细信息
        url = f"https://api.seniverse.com/v3/weather/daily.json?key={seniverse_key}&location={city_name}&language=zh-Hans&unit=c&days=3"
        response = requests.get(url)
        data = response.json()
        
        if data.get('results'):
            daily_forecasts = data['results'][0]['daily']
            location = data['results'][0]['location']['name']
            
            # 获取今天的数据（第一条记录）
            today_weather = daily_forecasts[0]
            
            return {
                'location': location,
                'temperature_low': today_weather['low'],
                'temperature_high': today_weather['high'],
                'text_day': today_weather['text_day'],
                'text_night': today_weather['text_night'],
                'wind_direction': today_weather.get('wind_direction', '未知'),
                'wind_speed': today_weather.get('wind_speed', '未知'),
                'humidity': today_weather.get('humidity', '未知')
            }
        else:
            return None
    except:
        return None

def get_weather_indices(city_name):
    """获取天气指数"""
    try:
        seniverse_key = "SEH8S1zMkjgD49Bsi"
        url = f"https://api.seniverse.com/v3/life/suggestion.json?key={seniverse_key}&location={city_name}&language=zh-Hans"
        response = requests.get(url)
        data = response.json()
        
        if data.get('results'):
            suggestions = data['results'][0]['suggestion']
            
            index_mapping = {
                'air_condition': '空调指数',
                'car_washing': '洗车指数',
                'cold': '感冒指数',
                'comfort': '舒适度指数',
                'dress': '穿衣指数',
                'exercise': '运动指数',
                'travel': '旅游指数',
                'uv': '紫外线指数'
            }
            
            indices = []
            for key, suggestion in suggestions.items():
                index_name = index_mapping.get(key, key)
                # 确保所有字段都存在且不为空
                if suggestion and 'brief' in suggestion and 'details' in suggestion:
                    indices.append({
                        'name': index_name,
                        'brief': suggestion['brief'],
                        'details': suggestion['details']
                    })
            
            return indices if indices else None
        else:
            return None
    except Exception as e:
        print(f"获取天气指数错误: {e}")
        return None

def get_weekly_forecast(city_name):
    """获取3天天气预报"""
    try:
        seniverse_key = "SEH8S1zMkjgD49Bsi"
        url = f"https://api.seniverse.com/v3/weather/daily.json?key={seniverse_key}&location={city_name}&language=zh-Hans&unit=c&days=3"
        response = requests.get(url)
        data = response.json()
        
        if data.get('results'):
            daily_forecasts = data['results'][0]['daily']
            
            forecasts = []
            for i, day in enumerate(daily_forecasts):
                date = datetime.strptime(day['date'], '%Y-%m-%d')
                weekday = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][date.weekday()]
                
                if i == 0:
                    day_label = "今天"
                elif i == 1:
                    day_label = "明天"
                else:
                    day_label = weekday
                
                forecasts.append({
                    'day_label': day_label,
                    'date': day['date'],
                    'low': day['low'],
                    'high': day['high'],
                    'text_day': day['text_day'],
                    'text_night': day['text_night'],
                    'wind_direction': day.get('wind_direction', '未知'),
                    'wind_speed': day.get('wind_speed', '未知'),
                    'humidity': day.get('humidity', '未知')
                })
            
            return forecasts
        else:
            return None
    except:
        return None


# 预测函数
def recommend_crops(soil_ph, temp, humidity, n, p, k, top_k=3):
    """推荐作物主函数"""
    # 验证输入
    validation_errors = validate_input_features(soil_ph, temp, humidity, n, p, k)
    if validation_errors:
        return [{'error': error} for error in validation_errors]

    # 加载模型
    catboost_model, transformer_model, label_encoder = load_models()
    if catboost_model is None:
        return [{'error': '模型加载失败，无法进行预测'}]

    try:
        # 准备特征
        features = np.array([[soil_ph, temp, humidity, n, p, k]])

        # 融合预测（CatBoost 0.7 + Transformer 0.3）
        catboost_proba = catboost_model.predict_proba(features)[0]
        with torch.no_grad():
            transformer_logits = transformer_model(torch.tensor(features, dtype=torch.float32).unsqueeze(1))
            transformer_proba = torch.softmax(transformer_logits, dim=1).numpy()[0]

        fused_proba = 0.7 * catboost_proba + 0.3 * transformer_proba

        # 获取Top-K推荐
        top_indices = np.argsort(fused_proba)[-top_k:][::-1]
        recommendations = []
        for i, idx in enumerate(top_indices):
            crop_name = label_encoder.inverse_transform([idx])[0]
            score = int(round(fused_proba[idx] * 100))  # 转为0-100评分
            recommendations.append({
                'rank': i + 1,
                'crop': crop_name,
                'score': score
            })
        return recommendations
    except Exception as e:
        return [{'error': f'预测过程中出现错误: {str(e)}'}]


# 侧边栏导航
with st.sidebar:
    st.header("🌱 智能农业助手")
    
    # 页面选择
    page = st.selectbox(
        "选择功能页面",
        ["🌿 作物推荐", "🌤️ 天气预报"],
        index=0
    )
    
    st.header("ℹ️ 系统信息")
    if page == "🌿 作物推荐":
        st.markdown("""
        **技术架构：**
        - 🚀 CatBoost + Transformer 融合模型
        - 📊 贝叶斯优化超参数调优
        - 🎯 Top-3 准确率 > 99%
        - ⚡ 实时推理 < 10ms
        """)

        st.header("📋 参数范围说明")
        for param, (min_val, max_val) in FEATURE_RANGES.items():
            st.markdown(f"**{param}**: {min_val} - {max_val}")

        st.header("🎯 使用说明")
        st.markdown("""
        1. 输入6个环境参数
        2. 点击'获取作物推荐'按钮
        3. 查看Top-3推荐结果
        4. 0-100分表示匹配度
        """)
    else:
        st.markdown("""
        **天气功能：**
        - 📍 自动定位当前位置
        - 🌡️ 实时天气信息
        - 📊 生活指数建议
        - 📅 3天天气预报
        """)

# 根据选择的页面显示不同内容
if page == "🌿 作物推荐":
    # 作物推荐页面
    st.markdown('<div class="main-header">🌿 智能作物推荐系统</div>', unsafe_allow_html=True)
    st.markdown("""
    **基于 CatBoost + Transformer 融合模型的智能农业推荐系统**

    输入6个环境参数，系统将为您推荐最适合种植的3种作物，并提供0-100的匹配评分。
    """)

    # 主界面 - 输入表单
    st.header("📊 输入环境参数")

    # 使用两列布局
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌡️ 环境参数")
        soil_ph = st.slider(
            "土壤pH值",
            min_value=6.0,
            max_value=9.0,
            value=7.0,
            step=0.1,
            help=f"取值范围: {FEATURE_RANGES['SOIL_PH'][0]} - {FEATURE_RANGES['SOIL_PH'][1]}"
        )
        temp = st.slider(
            "温度 (°C)",
            min_value=5.0,
            max_value=47.0,
            value=25.0,
            step=0.1,
            help=f"取值范围: {FEATURE_RANGES['TEMP'][0]} - {FEATURE_RANGES['TEMP'][1]}"
        )
        humidity = st.slider(
            "相对湿度 (%)",
            min_value=15,
            max_value=100,
            value=70,
            help=f"取值范围: {FEATURE_RANGES['RELATIVE_HUMIDITY'][0]} - {FEATURE_RANGES['RELATIVE_HUMIDITY'][1]}"
        )

    with col2:
        st.subheader("🧪 土壤养分")
        n = st.slider(
            "氮含量 (N)",
            min_value=20,
            max_value=199,
            value=120,
            help=f"取值范围: {FEATURE_RANGES['N'][0]} - {FEATURE_RANGES['N'][1]}"
        )
        p = st.slider(
            "磷含量 (P)",
            min_value=20,
            max_value=100,
            value=80,
            help=f"取值范围: {FEATURE_RANGES['P'][0]} - {FEATURE_RANGES['P'][1]}"
        )
        k = st.slider(
            "钾含量 (K)",
            min_value=20,
            max_value=149,
            value=60,
            help=f"取值范围: {FEATURE_RANGES['K'][0]} - {FEATURE_RANGES['K'][1]}"
        )

    # 推荐按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🌿 获取作物推荐",
            type="primary",
            use_container_width=True
        )

    # 显示输入参数摘要
    st.markdown("### 📋 输入参数汇总")
    param_cols = st.columns(6)
    params = [
        ("pH值", f"{soil_ph}"),
        ("温度", f"{temp}°C"),
        ("湿度", f"{humidity}%"),
        ("氮含量", f"{n}"),
        ("磷含量", f"{p}"),
        ("钾含量", f"{k}")
    ]
    for i, (name, value) in enumerate(params):
        with param_cols[i]:
            st.markdown(f'<div class="parameter-card"><strong>{name}</strong><br>{value}</div>', unsafe_allow_html=True)

    # 预测和结果显示
    if predict_button:
        with st.spinner('🔍 正在分析最佳作物...'):
            recommendations = recommend_crops(soil_ph, temp, humidity, n, p, k)

        # 显示结果
        if recommendations and 'error' in recommendations[0]:
            st.markdown("### ❌ 输入错误")
            for rec in recommendations:
                st.markdown(f'<div class="error-message">{rec["error"]}</div>', unsafe_allow_html=True)
            st.info("请调整输入参数至有效范围内后重试。")
        else:
            st.markdown("### 🎯 推荐作物 Top 3")
            st.markdown('<div class="success-message">✅ 推荐完成！以下是系统为您推荐的最佳作物：</div>',
                        unsafe_allow_html=True)

            # 显示推荐结果
            for rec in recommendations:
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h3>🏆 第 {rec['rank']} 名：{rec['crop']}</h3>
                        <p><strong>匹配评分：</strong>{rec['score']}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(rec['score'] / 100)

            # 显示技术信息
            with st.expander("🔬 技术信息"):
                st.markdown("""
                **模型信息：**
                - 使用CatBoost + Transformer融合模型
                - 融合权重：CatBoost 70% + Transformer 30%
                - 基于大量农业数据进行训练
                - Top-3准确率超过95%
                """)

else:
    # 天气预报页面
    st.markdown('<div class="main-header">🌤️ 天气预报系统</div>', unsafe_allow_html=True)
    st.markdown("""
    **基于百度地图定位 + 心知天气API的智能天气预报系统**
    
    自动获取当前位置，提供实时天气信息、生活指数建议和3天天气预报。
    """)
    
    # 天气功能控制
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        refresh_weather = st.button(
            "🔄 刷新天气信息",
            type="primary",
            use_container_width=True
        )
    
    if refresh_weather or 'weather_data' not in st.session_state:
        with st.spinner('🌍 正在获取位置和天气信息...'):
            # 获取位置
            current_city = get_weather_location()
            
            if current_city:
                # 获取天气数据
                current_weather = get_current_weather(current_city)
                weather_indices = get_weather_indices(current_city)
                weekly_forecast = get_weekly_forecast(current_city)
                
                # 存储到session state
                st.session_state.weather_data = {
                    'city': current_city,
                    'current': current_weather,
                    'indices': weather_indices,
                    'forecast': weekly_forecast,
                    'last_update': datetime.now()
                }
            else:
                st.error("❌ 无法获取当前位置，请检查网络连接")
                st.session_state.weather_data = None
    
    # 显示天气信息
    if st.session_state.weather_data:
        weather_data = st.session_state.weather_data
        
        # 当前天气
        if weather_data['current']:
            st.header("🌡️ 当前天气")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h3>📍 {weather_data['current']['location']}</h3>
                    <p><strong>温度：</strong>{weather_data['current']['temperature_low']}°C ~ {weather_data['current']['temperature_high']}°C</p>
                    <p><strong>白天：</strong>{weather_data['current']['text_day']}</p>
                    <p><strong>夜间：</strong>{weather_data['current']['text_night']}</p>
                    <p><strong>风向：</strong>{weather_data['current']['wind_direction']}</p>
                    <p><strong>风速：</strong>{weather_data['current']['wind_speed']} km/h</p>
                    <p><strong>湿度：</strong>{weather_data['current']['humidity']}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.info(f"🕐 更新时间：{weather_data['last_update'].strftime('%H:%M:%S')}")
        
        # 天气指数
        if weather_data['indices']:
            st.header("📊 生活指数建议")
            cols = st.columns(2)
            
            for i, index in enumerate(weather_data['indices']):
                with cols[i % 2]:
                    # 确保数据完整性
                    if index and 'name' in index and 'brief' in index and 'details' in index:
                        st.markdown(f"""
                        <div class="parameter-card">
                            <h4>{index['name']}</h4>
                            <p><strong>等级：</strong>{index['brief']}</p>
                            <p><strong>建议：</strong>{index['details']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("📊 暂无生活指数数据")
        
        # 3天预报
        if weather_data['forecast']:
            st.header("📅 3天天气预报")
            
            for day in weather_data['forecast'][:3]:
                # 直接显示天气卡片
                st.markdown(f"""
                <div class="recommendation-card">
                    <h3>📅 {day['day_label']} ({day['date']})</h3>
                    <p><strong>🌡️ 温度：</strong>{day['low']}°C ~ {day['high']}°C</p>
                    <p><strong>🌤️ 白天：</strong>{day['text_day']}</p>
                    <p><strong>🌙 夜间：</strong>{day['text_night']}</p>
                    <p><strong>💨 风向：</strong>{day['wind_direction']}</p>
                    <p><strong>🌪️ 风速：</strong>{day['wind_speed']} km/h</p>
                    <p><strong>💧 湿度：</strong>{day['humidity']}%</p>
                </div>
                """, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🌱 智能农业助手 | 集成作物推荐与天气预报的综合农业决策平台"
    "</div>",
    unsafe_allow_html=True
)