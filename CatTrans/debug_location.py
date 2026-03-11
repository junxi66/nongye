import streamlit as st
import requests
import json

def test_location_apis():
    """测试所有定位API的响应"""
    st.header("🔍 定位API测试")
    
    apis = [
        {
            "name": "ipinfo.io",
            "url": "https://ipinfo.io/json",
            "parser": "ipinfo"
        },
        {
            "name": "ip-api.com", 
            "url": "http://ip-api.com/json/",
            "parser": "ipapi_com"
        },
        {
            "name": "ipapi.co",
            "url": "https://ipapi.co/json/", 
            "parser": "ipapi_co"
        },
        {
            "name": "ipgeolocation.io",
            "url": "https://api.ipgeolocation.io/ipgeo?apiKey=free",
            "parser": "ipgeolocation"
        }
    ]
    
    for api in apis:
        st.subheader(f"📍 {api['name']}")
        
        try:
            response = requests.get(api['url'], timeout=10)
            data = response.json()
            
            st.json(data)
            
            # 解析位置信息
            if api['parser'] == 'ipinfo':
                city = data.get('city', 'N/A')
                region = data.get('region', 'N/A') 
                country = data.get('country', 'N/A')
                ip = data.get('ip', 'N/A')
                
            elif api['parser'] == 'ipapi_com':
                city = data.get('city', 'N/A')
                region = data.get('regionName', 'N/A')
                country = data.get('country', 'N/A')
                ip = data.get('query', 'N/A')
                
            elif api['parser'] == 'ipapi_co':
                city = data.get('city', 'N/A')
                region = data.get('region', 'N/A')
                country = data.get('country_name', 'N/A')
                ip = data.get('ip', 'N/A')
                
            elif api['parser'] == 'ipgeolocation':
                city = data.get('city', 'N/A')
                region = data.get('state_prov', 'N/A')
                country = data.get('country_name', 'N/A')
                ip = data.get('ip', 'N/A')
            
            st.write(f"**IP地址:** {ip}")
            st.write(f"**城市:** {city}")
            st.write(f"**地区:** {region}")
            st.write(f"**国家:** {country}")
            
            # 检查是否为无效位置
            invalid_locations = ['The Dalles', 'Oregon', 'Washington', 'California']
            if city in invalid_locations:
                st.error(f"❌ 检测到服务器位置: {city}")
            elif city == 'N/A':
                st.warning("⚠️ 无法获取城市信息")
            else:
                st.success(f"✅ 有效位置: {city}")
                
        except Exception as e:
            st.error(f"❌ API请求失败: {e}")
        
        st.markdown("---")

def test_weather_api():
    """测试天气API"""
    st.header("🌤️ 天气API测试")
    
    # 测试几个城市的天气API
    test_cities = ["北京", "上海", "广州", "深圳"]
    
    for city in test_cities:
        st.subheader(f"📍 {city}")
        
        try:
            url = f"https://api.seniverse.com/v3/weather/daily.json?key=SEH8S1zMkjgD49Bsi&location={city}&language=zh-Hans&unit=c&days=1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('results'):
                result = data['results'][0]
                location = result['location']['name']
                weather = result['daily'][0]
                
                st.write(f"**位置:** {location}")
                st.write(f"**天气:** {weather['text_day']}")
                st.write(f"**温度:** {weather['low']}°C ~ {weather['high']}°C")
                st.success("✅ 天气API正常")
            else:
                st.error("❌ 无天气数据")
                
        except Exception as e:
            st.error(f"❌ 天气API请求失败: {e}")
        
        st.markdown("---")

def main():
    st.set_page_config(
        page_title="定位调试工具",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Streamlit Cloud 定位调试工具")
    st.markdown("这个工具用于诊断在Streamlit Cloud上的定位问题")
    
    # 显示当前环境信息
    st.header("🌐 环境信息")
    st.write(f"**当前时间:** {st.session_state.get('time', 'N/A')}")
    st.write(f"**用户代理:** {st.session_state.get('user_agent', 'N/A')}")
    
    # 测试定位API
    test_location_apis()
    
    # 测试天气API
    test_weather_api()

if __name__ == "__main__":
    main()
