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
    """加载预训练的模型 - 部署安全版本"""
    try:
        import os
        import sys
        
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建模型文件的完整路径
        catboost_path = os.path.join(current_dir, 'catboost_model.pkl')
        transformer_path = os.path.join(current_dir, 'transformer_model.pth')
        encoder_path = os.path.join(current_dir, 'label_encoder.pkl')
        
        # 调试信息
        st.write(f"🔍 当前工作目录: {os.getcwd()}")
        st.write(f"🔍 脚本目录: {current_dir}")
        st.write(f"🔍 查找CatBoost模型: {catboost_path}")
        st.write(f"🔍 查找Transformer模型: {transformer_path}")
        st.write(f"🔍 查找标签编码器: {encoder_path}")
        
        # 检查文件是否存在
        if not os.path.exists(catboost_path):
            st.error(f"❌ CatBoost模型文件不存在：{catboost_path}")
            # 尝试在当前目录查找
            if os.path.exists('catboost_model.pkl'):
                st.write("✅ 在当前目录找到CatBoost模型")
                catboost_path = 'catboost_model.pkl'
            else:
                return None, None, None
                
        if not os.path.exists(transformer_path):
            st.error(f"❌ Transformer模型文件不存在：{transformer_path}")
            # 尝试在当前目录查找
            if os.path.exists('transformer_model.pth'):
                st.write("✅ 在当前目录找到Transformer模型")
                transformer_path = 'transformer_model.pth'
            else:
                return None, None, None
                
        if not os.path.exists(encoder_path):
            st.error(f"❌ 标签编码器文件不存在：{encoder_path}")
            # 尝试在当前目录查找
            if os.path.exists('label_encoder.pkl'):
                st.write("✅ 在当前目录找到标签编码器")
                encoder_path = 'label_encoder.pkl'
            else:
                return None, None, None
        
        # 加载CatBoost模型
        catboost_model = joblib.load(catboost_path)
        st.success("✅ CatBoost模型加载成功")

        # 加载标签编码器
        label_encoder = joblib.load(encoder_path)
        st.success("✅ 标签编码器加载成功")

        # 加载Transformer模型
        transformer_model = OptimizedTransformerClassifier(
            input_size=6,
            hidden_size=64,
            num_classes=len(label_encoder.classes_)
        )
        transformer_model.load_state_dict(torch.load(transformer_path, map_location='cpu'))
        transformer_model.eval()
        st.success("✅ Transformer模型加载成功")

        return catboost_model, transformer_model, label_encoder
        
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.write(f"🔍 错误详情: {type(e).__name__}")
        st.info("请检查模型文件是否完整或重新上传")
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
SENIVERSE_KEY = "SEH8S1zMkjgD49Bsi"

def make_api_request(url):
    """通用的API请求函数"""
    try:
        response = requests.get(url, timeout=10)
        return response.json()
    except:
        return None

def get_weather_location():
    """使用多个API获取当前位置，提高定位准确性"""
    # 首先尝试ipinfo.io
    try:
        url = "https://ipinfo.io/json"
        data = make_api_request(url)
        
        if data and 'city' in data:
            city = data.get('city', '未知城市')
            region = data.get('region', '')
            country = data.get('country', '')
            
            # 如果城市信息不详细，尝试使用region
            if not city or city == '未知城市':
                city = region if region else '未知城市'
            
            # 如果还是没有，使用country
            if not city or city == '未知城市':
                city = country if country else '未知城市'
            
            # 如果定位到的是服务器位置（如The Dalles），尝试备用API
            if city in ['The Dalles', 'Unknown', '未知城市']:
                return get_weather_location_backup()
                
            return city
        else:
            return get_weather_location_backup()
    except Exception as e:
        st.error(f"获取位置失败: {e}")
        return get_weather_location_backup()

def get_weather_location_backup():
    """备用定位API"""
    try:
        # 使用ip-api.com作为备用
        url = "http://ip-api.com/json/"
        data = make_api_request(url)
        
        if data and 'city' in data and data['status'] == 'success':
            city = data.get('city', '未知城市')
            region = data.get('regionName', '')
            country = data.get('country', '')
            
            # 优先使用城市，其次是地区，最后是国家
            if city and city not in ['The Dalles', 'Unknown']:
                return city
            elif region and region not in ['The Dalles', 'Unknown']:
                return region
            elif country and country not in ['The Dalles', 'Unknown']:
                return country
            else:
                return "北京"  # 最后的备用选项
        else:
            return "北京"  # 默认城市
    except Exception as e:
        st.warning(f"备用定位也失败，使用默认城市: {e}")
        return "北京"  # 默认城市

def get_weather_data(city_name):
    """一次性获取所有天气数据，避免重复API调用"""
    # 使用同一个API获取天气和预报数据
    url = f"https://api.seniverse.com/v3/weather/daily.json?key={SENIVERSE_KEY}&location={city_name}&language=zh-Hans&unit=c&days=3"
    weather_data = make_api_request(url)
    
    if not weather_data or not weather_data.get('results'):
        return None, None, None
    
    result = weather_data['results'][0]
    daily_forecasts = result['daily']
    location = result['location']['name']
    
    # 当前天气（今天的数据）
    today_weather = daily_forecasts[0]
    current_weather = {
        'location': location,
        'temperature_low': today_weather['low'],
        'temperature_high': today_weather['high'],
        'text_day': today_weather['text_day'],
        'text_night': today_weather['text_night'],
        'wind_direction': today_weather.get('wind_direction', '未知'),
        'wind_speed': today_weather.get('wind_speed', '未知'),
        'humidity': today_weather.get('humidity', '未知')
    }
    
    # 3天预报
    forecasts = []
    weekday = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    
    for i, day in enumerate(daily_forecasts):
        date = datetime.strptime(day['date'], '%Y-%m-%d')
        
        if i == 0:
            day_label = "今天"
        elif i == 1:
            day_label = "明天"
        else:
            day_label = weekday[date.weekday()]
        
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
    
    # 获取天气指数
    indices_url = f"https://api.seniverse.com/v3/life/suggestion.json?key={SENIVERSE_KEY}&location={city_name}&language=zh-Hans"
    indices_data = make_api_request(indices_url)
    
    indices = None
    if indices_data and indices_data.get('results'):
        suggestions = indices_data['results'][0]['suggestion']
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
            if suggestion and 'brief' in suggestion and 'details' in suggestion:
                indices.append({
                    'name': index_name,
                    'brief': suggestion['brief'],
                    'details': suggestion['details']
                })
    
    return current_weather, indices if indices else None, forecasts

def get_current_weather(city_name):
    """获取当前天气（保持向后兼容）"""
    current_weather, _, _ = get_weather_data(city_name)
    return current_weather

def get_weather_indices(city_name):
    """获取天气指数（保持向后兼容）"""
    _, indices, _ = get_weather_data(city_name)
    return indices

def get_weekly_forecast(city_name):
    """获取3天天气预报（保持向后兼容）"""
    _, _, forecasts = get_weather_data(city_name)
    return forecasts


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
    
    # 初始化页面状态
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🌿 作物推荐"
    if 'last_page' not in st.session_state:
        st.session_state.last_page = None
    
    page = st.session_state.current_page
    
    # 页面选择按钮
    st.markdown("### 📋 选择功能")
    
    # 作物推荐按钮
    crop_btn = st.button(
        "🌿 作物推荐",
        use_container_width=True,
        type="primary" if page == "🌿 作物推荐" else "secondary"
    )
    
    # 天气预报按钮
    weather_btn = st.button(
        "🌤️ 天气预报", 
        use_container_width=True,
        type="primary" if page == "🌤️ 天气预报" else "secondary"
    )
    
    # 处理按钮点击
    if crop_btn:
        st.session_state.current_page = "🌿 作物推荐"
        st.rerun()
    elif weather_btn:
        st.session_state.current_page = "🌤️ 天气预报"
        # 点击天气预报按钮时清除天气数据缓存，强制刷新
        if 'weather_data' in st.session_state:
            del st.session_state.weather_data
        st.rerun()
    
    st.markdown("---")
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
    **基于ipinfo.io定位 + 心知天气API的智能天气预报系统**
    
    自动获取当前位置，提供实时天气信息、生活指数建议和3天天气预报。
    """)
    
    # 城市选择功能
    st.header("📍 城市选择")
    
    # 城市列表（全国所有城市）
    cities = [
        # 直辖市
        "北京", "上海", "天津", "重庆",
        
        # 河北省
        "石家庄", "唐山", "秦皇岛", "邯郸", "邢台", "保定", "张家口", "承德", "沧州", "廊坊", "衡水",
        
        # 山西省
        "太原", "大同", "阳泉", "长治", "晋城", "朔州", "晋中", "运城", "忻州", "临汾", "吕梁",
        
        # 内蒙古自治区
        "呼和浩特", "包头", "乌海", "赤峰", "通辽", "鄂尔多斯", "呼伦贝尔", "巴彦淖尔", "乌兰察布",
        
        # 辽宁省
        "沈阳", "大连", "鞍山", "抚顺", "本溪", "丹东", "锦州", "营口", "阜新", "辽阳", "盘锦", "铁岭", "朝阳", "葫芦岛",
        
        # 吉林省
        "长春", "吉林", "四平", "辽源", "通化", "白山", "松原", "白城", "延边",
        
        # 黑龙江省
        "哈尔滨", "齐齐哈尔", "鸡西", "鹤岗", "双鸭山", "大庆", "伊春", "佳木斯", "七台河", "牡丹江", "黑河", "绥化", "大兴安岭",
        
        # 江苏省
        "南京", "无锡", "徐州", "常州", "苏州", "南通", "连云港", "淮安", "盐城", "扬州", "镇江", "泰州", "宿迁",
        
        # 浙江省
        "杭州", "宁波", "温州", "嘉兴", "湖州", "绍兴", "金华", "衢州", "舟山", "台州", "丽水",
        
        # 安徽省
        "合肥", "芜湖", "蚌埠", "淮南", "马鞍山", "淮北", "铜陵", "安庆", "黄山", "滁州", "阜阳", "宿州", "六安", "亳州", "池州", "宣城",
        
        # 福建省
        "福州", "厦门", "莆田", "三明", "泉州", "漳州", "南平", "龙岩", "宁德",
        
        # 江西省
        "南昌", "景德镇", "萍乡", "九江", "新余", "鹰潭", "赣州", "吉安", "宜春", "抚州", "上饶",
        
        # 山东省
        "济南", "青岛", "淄博", "枣庄", "东营", "烟台", "潍坊", "济宁", "泰安", "威海", "日照", "莱芜", "临沂", "德州", "聊城", "滨州", "菏泽",
        
        # 河南省
        "郑州", "开封", "洛阳", "平顶山", "安阳", "鹤壁", "新乡", "焦作", "濮阳", "许昌", "漯河", "三门峡", "南阳", "商丘", "信阳", "周口", "驻马店",
        
        # 湖北省
        "武汉", "黄石", "十堰", "宜昌", "襄阳", "鄂州", "荆门", "孝感", "荆州", "黄冈", "咸宁", "随州", "恩施",
        
        # 湖南省
        "长沙", "株洲", "湘潭", "衡阳", "邵阳", "岳阳", "常德", "张家界", "益阳", "郴州", "永州", "怀化", "娄底", "湘西",
        
        # 广东省
        "广州", "深圳", "珠海", "汕头", "韶关", "佛山", "江门", "湛江", "茂名", "肇庆", "惠州", "梅州", "汕尾", "河源", "阳江", "清远", "东莞", "中山", "潮州", "揭阳", "云浮",
        
        # 广西壮族自治区
        "南宁", "柳州", "桂林", "梧州", "北海", "防城港", "钦州", "贵港", "玉林", "百色", "贺州", "河池", "来宾", "崇左",
        
        # 海南省
        "海口", "三亚", "三沙", "儋州", "五指山", "文昌", "琼海", "万宁", "东方", "定安", "屯昌", "澄迈", "临高", "白沙", "昌江", "乐东", "陵水", "保亭", "琼中",
        
        # 四川省
        "成都", "自贡", "攀枝花", "泸州", "德阳", "绵阳", "广元", "遂宁", "内江", "乐山", "南充", "眉山", "宜宾", "广安", "达州", "雅安", "巴中", "资阳", "阿坝", "甘孜", "凉山",
        
        # 贵州省
        "贵阳", "六盘水", "遵义", "安顺", "毕节", "铜仁", "黔西南", "黔东南", "黔南",
        
        # 云南省
        "昆明", "曲靖", "玉溪", "保山", "昭通", "丽江", "普洱", "临沧", "楚雄", "红河", "文山", "西双版纳", "大理", "德宏", "怒江", "迪庆",
        
        # 西藏自治区
        "拉萨", "日喀则", "昌都", "林芝", "山南", "那曲", "阿里",
        
        # 陕西省
        "西安", "铜川", "宝鸡", "咸阳", "渭南", "延安", "汉中", "榆林", "安康", "商洛",
        
        # 甘肃省
        "兰州", "嘉峪关", "金昌", "白银", "天水", "武威", "张掖", "平凉", "酒泉", "庆阳", "定西", "陇南", "临夏", "甘南",
        
        # 青海省
        "西宁", "海东", "海北", "黄南", "海南", "果洛", "玉树", "海西",
        
        # 宁夏回族自治区
        "银川", "石嘴山", "吴忠", "固原", "中卫",
        
        # 新疆维吾尔自治区
        "乌鲁木齐", "克拉玛依", "吐鲁番", "哈密", "昌吉", "博尔塔拉", "巴音郭楞", "阿克苏", "克孜勒苏", "喀什", "和田", "伊犁", "塔城", "阿勒泰",
        
        # 特别行政区
        "香港", "澳门",
        
        # 台湾省
        "台北", "高雄", "台中", "台南", "新竹", "基隆", "嘉义", "宜兰", "桃园", "新北", "台东", "花莲", "彰化", "云林", "屏东", "苗栗", "南投", "澎湖", "金门", "马祖"
    ]
    
    # 创建两列布局，调整对齐
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 城市选择框
        selected_city = st.selectbox(
            "选择城市：",
            options=cities,
            index=0,
            key="city_selector",
            help=f"共 {len(cities)} 个城市可选"
        )
    
    with col2:
        # 定位并刷新按钮，垂直居中对齐
        st.markdown("<br>", unsafe_allow_html=True)  # 添加一些垂直间距
        refresh_weather = st.button("🔄 定位并刷新", use_container_width=True)
    
    # 获取天气数据 - 使用更安全的方式
    refresh_triggered = refresh_weather
    
    # 页面切换检测
    if page == "🌤️ 天气预报":
        if 'weather_data' not in st.session_state:
            refresh_triggered = True
        elif 'last_page' in st.session_state and st.session_state.last_page != "🌤️ 天气预报":
            refresh_triggered = True
    
    # 记录当前页面
    st.session_state.last_page = page
    
    # 城市变更检测
    city_changed = False
    if 'weather_data' in st.session_state and selected_city != st.session_state.weather_data.get('city', ''):
        city_changed = True
    
    # 执行天气数据获取
    if refresh_triggered or city_changed:
        with st.spinner('🌍 正在获取位置和天气信息...'):
            if refresh_triggered and not city_changed:
                # 自动定位
                current_city = get_weather_location()
                if current_city:
                    st.success(f"自动定位成功：{current_city}")
                else:
                    st.warning("自动定位失败，使用选择城市")
                    current_city = selected_city
            else:
                # 使用选择的城市
                current_city = selected_city
            
            if current_city:
                # 获取天气数据
                current_weather, weather_indices, weekly_forecast = get_weather_data(current_city)
                
                # 安全地更新session state
                new_weather_data = {
                    'city': current_city,
                    'current': current_weather,
                    'indices': weather_indices,
                    'forecast': weekly_forecast,
                    'last_update': datetime.now()
                }
                
                # 一次性更新，避免多次DOM操作
                st.session_state.weather_data = new_weather_data
            else:
                st.error("❌ 无法获取天气信息，请检查网络连接")
                if 'weather_data' in st.session_state:
                    del st.session_state.weather_data
    
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