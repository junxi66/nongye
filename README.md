# 🌱 智能农业助手

基于 CatBoost + Transformer 融合模型的智能作物推荐系统

## 🎯 功能特性

- 🌿 **智能作物推荐** - 基于环境参数推荐最适合的作物
- 🌤️ **天气预报系统** - 实时天气信息和3天预报
- 🤖 **融合模型** - CatBoost + Transformer 高精度预测
- 📊 **可视化界面** - 美观的 Streamlit Web 界面

## 📋 技术栈

- **Python 3.10**
- **Streamlit** - Web 框架
- **CatBoost** - 梯度提升模型
- **PyTorch** - 深度学习框架
- **scikit-learn** - 机器学习工具
- **pandas/numpy** - 数据处理

## 🚀 部署到 Streamlit Cloud

### 1. 准备 GitHub 仓库

```bash
# 创建 .gitignore 文件
echo "__pycache__/
*.pyc
venv*/
.env
.DS_Store
*.log" > .gitignore

# 初始化 Git
git init
git add .
git commit -m "Initial commit: Smart Agriculture Assistant"
```

### 2. 推送到 GitHub

```bash
# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/智能农业助手.git
git branch -M main
git push -u origin main
```

### 3. 部署到 Streamlit Cloud

1. 访问 [Streamlit Cloud](https://share.streamlit.io/)
2. 使用 GitHub 账号登录
3. 点击 "New app" 
4. 选择你的 GitHub 仓库
5. 配置部署设置：
   - **Repository**: 你的用户名/智能农业助手
   - **Branch**: main
   - **Main file path**: CatTrans/app.py
   - **Python version**: 3.10

6. 点击 "Deploy!"

## 📁 项目结构

```
6-AIC/
├── CatTrans/
│   ├── app.py              # 主应用文件
│   ├── catboost_model.pkl   # CatBoost 模型
│   ├── transformer_model.pth # Transformer 模型
│   └── label_encoder.pkl   # 标签编码器
├── crop_train.csv          # 训练数据
├── crop_val.csv           # 验证数据
├── crop_test.csv          # 测试数据
├── requirements.txt        # 依赖包
└── README.md             # 项目说明
```

## ⚙️ 部署前检查清单

### ✅ 必需文件
- [ ] `CatTrans/app.py` - 主应用
- [ ] `CatTrans/catboost_model.pkl` - CatBoost 模型
- [ ] `CatTrans/transformer_model.pth` - Transformer 模型  
- [ ] `CatTrans/label_encoder.pkl` - 标签编码器
- [ ] `requirements.txt` - 依赖包列表

### ✅ requirements.txt 内容
```txt
streamlit>=1.28.0
catboost>=1.2
torch>=2.0.1
scikit-learn>=1.3.0
pandas>=2.0.3
numpy>=1.24.3
joblib>=1.3.0
requests>=2.31.0
```

### ✅ 模型文件路径检查
确保 `app.py` 中的模型路径正确：
```python
catboost_model = joblib.load('CatTrans/catboost_model.pkl')
label_encoder = joblib.load('CatTrans/label_encoder.pkl')
transformer_model.load_state_dict(torch.load('CatTrans/transformer_model.pth', map_location='cpu'))
```

## 🔧 常见部署问题

### 1. 模型文件过大
Streamlit Cloud 有 1GB 限制，如果模型过大：
- 考虑模型压缩
- 使用模型量化
- 删除不必要的文件

### 2. 依赖包冲突
确保 requirements.txt 中的版本兼容：
```txt
numpy>=1.24.3,<2.0.0  # 避免与 PyTorch 冲突
torch>=2.0.1,<2.1.0
```

### 3. 内存不足
Streamlit Cloud 免费版内存限制：
- 优化模型加载方式
- 使用 `@st.cache_resource` 缓存模型
- 考虑升级到付费计划

## 🌐 访问地址

部署成功后，你的应用将在：
`https://你的用户名-智能农业助手.streamlit.app`

## 📞 支持

如果遇到部署问题：
1. 检查 Streamlit Cloud 部署日志
2. 确认所有必需文件都已上传
3. 验证 requirements.txt 版本兼容性
4. 检查模型文件路径是否正确

---

🌱 **智能农业助手** - 让农业决策更智能！
