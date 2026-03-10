import requests
import json
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import threading

class WeatherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("天气预报应用")
        self.root.geometry("800x600")
        
        
        self.baidu_ak = "WC6s35CQwWvjpg5txGj20ricDgPLp2US"  
        self.seniverse_key = "SEH8S1zMkjgD49Bsi"  
        
        self.current_location = None
        self.current_city = None
        
        self.setup_ui()
        self.get_location()
    
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="天气预报", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 位置信息
        self.location_label = ttk.Label(main_frame, text="正在获取位置...", font=("Arial", 12))
        self.location_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # 刷新按钮
        refresh_btn = ttk.Button(main_frame, text="刷新天气", command=self.refresh_weather)
        refresh_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # 创建笔记本控件用于分页显示
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # 当天天气标签页
        today_frame = ttk.Frame(notebook)
        notebook.add(today_frame, text="当天天气")
        
        # 当天天气指数标签页
        indices_frame = ttk.Frame(notebook)
        notebook.add(indices_frame, text="天气指数")
        
        # 未来3天天气标签页
        forecast_frame = ttk.Frame(notebook)
        notebook.add(forecast_frame, text="3天预报")
        
        # 设置当天天气界面
        self.setup_today_weather(today_frame)
        
        # 设置天气指数界面
        self.setup_weather_indices(indices_frame)
        
        # 设置3天预报界面
        self.setup_weekly_forecast(forecast_frame)
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
    
    def setup_today_weather(self, parent):
        # 当天天气显示区域
        self.today_info = tk.Text(parent, wrap=tk.WORD, width=70, height=20, font=("Arial", 10))
        self.today_info.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.today_info.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.today_info.configure(yscrollcommand=scrollbar.set)
        
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
    
    def setup_weather_indices(self, parent):
        # 天气指数显示区域
        self.indices_info = tk.Text(parent, wrap=tk.WORD, width=70, height=20, font=("Arial", 10))
        self.indices_info.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.indices_info.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.indices_info.configure(yscrollcommand=scrollbar.set)
        
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
    
    def setup_weekly_forecast(self, parent):
        # 3天预报显示区域
        self.forecast_info = tk.Text(parent, wrap=tk.WORD, width=70, height=20, font=("Arial", 10))
        self.forecast_info.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.forecast_info.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.forecast_info.configure(yscrollcommand=scrollbar.set)
        
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
    
    def get_location(self):
        """使用百度地图API获取当前位置"""
        def get_location_thread():
            try:
                # 使用IP定位获取大致位置
                url = f"https://api.map.baidu.com/location/ip?ak={self.baidu_ak}&coor=bd09ll"
                response = requests.get(url)
                data = response.json()
                
                if data.get('status') == 0:
                    self.current_location = data['content']
                    self.current_city = data['content']['address_detail']['city']
                    
                    # 更新位置标签
                    self.root.after(0, lambda: self.location_label.config(
                        text=f"当前位置: {self.current_city}"
                    ))
                    
                    # 获取天气信息
                    self.get_weather_data()
                else:
                    self.root.after(0, lambda: messagebox.showerror(
                        "错误", f"获取位置失败: {data.get('message', '未知错误')}"
                    ))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"获取位置时发生错误: {str(e)}"
                ))
        
        threading.Thread(target=get_location_thread, daemon=True).start()
    
    def get_weather_data(self):
        """获取天气数据"""
        if not self.current_city:
            return
        
        def get_weather_thread():
            try:
                # 心知天气直接使用城市名称
                # 获取实时天气
                self.get_current_weather(self.current_city)
                
                # 获取天气指数
                self.get_weather_indices(self.current_city)
                
                # 获取3天预报
                self.get_weekly_forecast(self.current_city)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"获取天气数据时发生错误: {str(e)}"
                ))
        
        threading.Thread(target=get_weather_thread, daemon=True).start()
    
    def get_current_weather(self, city_name):
        """获取当前天气"""
        try:
            # 使用3天预报API获取今天的详细信息
            url = f"https://api.seniverse.com/v3/weather/daily.json?key={self.seniverse_key}&location={city_name}&language=zh-Hans&unit=c&days=3"
            response = requests.get(url)
            data = response.json()
            
            if data.get('results'):
                daily_forecasts = data['results'][0]['daily']
                location = data['results'][0]['location']['name']
                
                # 获取今天的数据（第一条记录）
                today_weather = daily_forecasts[0]
                
                # 格式化显示
                weather_info = f"""
═══════════════════════════════════════
            当前天气信息
═══════════════════════════════════════

📍 位置: {location}
🌡️ 温度: {today_weather['low']}°C ~ {today_weather['high']}°C
🌤️ 白天: {today_weather['text_day']}
🌙 夜间: {today_weather['text_night']}
💨 风向: {today_weather.get('wind_direction', '未知')}
🌪️ 风速: {today_weather.get('wind_speed', '未知')}km/h
💧 湿度: {today_weather.get('humidity', '未知')}%

📅 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                
                self.root.after(0, lambda: self.update_today_display(weather_info))
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", "获取当前天气失败: 无数据返回"
                ))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "错误", f"获取当前天气时发生错误: {str(e)}"
            ))
    
    def get_weather_indices(self, city_name):
        """获取天气指数"""
        try:
            url = f"https://api.seniverse.com/v3/life/suggestion.json?key={self.seniverse_key}&location={city_name}&language=zh-Hans"
            response = requests.get(url)
            data = response.json()
            
            if data.get('results'):
                indices_info = "═══════════════════════════════════════\n            今日生活指数\n═══════════════════════════════════════\n\n"
                
                suggestions = data['results'][0]['suggestion']
                
                # 指数类型映射
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
                
                for key, suggestion in suggestions.items():
                    index_name = index_mapping.get(key, key)
                    brief = suggestion['brief']
                    details = suggestion['details']
                    
                    indices_info += f"📊 {index_name}\n"
                    indices_info += f"   等级: {brief}\n"
                    indices_info += f"   建议: {details}\n\n"
                
                self.root.after(0, lambda: self.update_indices_display(indices_info))
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", "获取天气指数失败: 无数据返回"
                ))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "错误", f"获取天气指数时发生错误: {str(e)}"
            ))
    def get_weekly_forecast(self, city_name):
        """获取3天天气预报"""
        try:
            url = f"https://api.seniverse.com/v3/weather/daily.json?key={self.seniverse_key}&location={city_name}&language=zh-Hans&unit=c&days=3"
            response = requests.get(url)
            data = response.json()
            
            if data.get('results'):
                daily_forecasts = data['results'][0]['daily']
                
                # 调试信息：检查实际返回的天数
                print(f"API返回的天数: {len(daily_forecasts)}")
                for i, day in enumerate(daily_forecasts):
                    print(f"第{i+1}天: {day['date']} - {day['text_day']}")
                
                forecast_info = f"═══════════════════════════════════════\n            未来3天天气预报\n═══════════════════════════════════════\n\n"
                
                for i, day in enumerate(daily_forecasts):
                    date = datetime.strptime(day['date'], '%Y-%m-%d')
                    weekday = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][date.weekday()]
                    
                    if i == 0:
                        day_label = "今天"
                    elif i == 1:
                        day_label = "明天"
                    else:
                        day_label = weekday
                    
                    forecast_info += f"📅 {day_label} ({day['date']})\n"
                    forecast_info += f"   🌡️ 温度: {day['low']}°C ~ {day['high']}°C\n"
                    forecast_info += f"   🌤️ 天气: {day['text_day']}\n"
                    forecast_info += f"   🌙 夜间: {day['text_night']}\n"
                    forecast_info += f"   💨 风向: {day.get('wind_direction', '未知')}\n"
                    forecast_info += f"   🌪️ 风速: {day.get('wind_speed', '未知')}km/h\n"
                    forecast_info += f"   💧 湿度: {day.get('humidity', '未知')}%\n"
                    forecast_info += f"   ──────────────────────────────────\n\n"
                
                self.root.after(0, lambda: self.update_forecast_display(forecast_info))
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", "获取3天预报失败: 无数据返回"
                ))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "错误", f"获取3天预报时发生错误: {str(e)}"
            ))
    
    def update_today_display(self, info):
        """更新当天天气显示"""
        self.today_info.delete(1.0, tk.END)
        self.today_info.insert(tk.END, info)
    
    def update_indices_display(self, info):
        """更新天气指数显示"""
        self.indices_info.delete(1.0, tk.END)
        self.indices_info.insert(tk.END, info)
    
    def update_forecast_display(self, info):
        """更新3天预报显示"""
        self.forecast_info.delete(1.0, tk.END)
        self.forecast_info.insert(tk.END, info)
    
    def refresh_weather(self):
        """刷新天气信息"""
        self.location_label.config(text="正在刷新...")
        self.get_location()

def main():
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()