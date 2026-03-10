# 🚀 Streamlit Cloud 部署指南

## 📋 部署前检查清单

### ✅ 必需文件
确保以下文件都在项目根目录中：
- `app.py` - 主应用文件
- `catboost_model.pkl` - CatBoost模型文件 (15.9MB)
- `transformer_model.pth` - PyTorch模型文件 (294KB)
- `label_encoder.pkl` - 标签编码器文件 (1.4KB)
- `requirements.txt` - Python依赖
- `.streamlit/config.toml` - Streamlit配置

### 📦 依赖文件
- `requirements.txt` - 包含所有Python包依赖
- `packages.txt` - 系统依赖（空文件，但需要存在）

## 🔧 部署步骤

### 1. 准备GitHub仓库
1. 将所有文件推送到GitHub仓库
2. 确保仓库是公开的
3. 检查文件大小限制（单个文件最大200MB）

### 2. Streamlit Cloud配置
1. 登录 [Streamlit Cloud](https://share.streamlit.io/)
2. 点击 "New app"
3. 选择GitHub仓库
4. 分支：`main` 或 `master`
5. 主文件路径：`app.py`
6. 点击 "Deploy"

### 3. 环境变量（可选）
如果需要，可以设置以下环境变量：
- `SENIVERSE_KEY` - 心知天气API密钥

## 🐛 常见问题解决

### ❌ 模型加载失败
**错误信息**：`No such file or directory: 'catboost_model.pkl'`

**解决方案**：
1. 确认模型文件已上传到GitHub
2. 检查文件路径是否正确
3. 确认文件大小不超过200MB

### 🌍 定位问题
**问题**：显示"The Dalles"而不是实际城市

**解决方案**：
- 应用已配置备用定位API
- 如果都失败，会默认使用"北京"
- 用户可以手动选择城市

### ⚡ 性能优化
- 模型使用`@st.cache_resource`缓存
- 天气API调用有超时限制
- 支持手动城市选择

## 📊 部署后测试

### 功能测试清单
- [ ] 页面正常加载
- [ ] 侧边栏按钮工作
- [ ] 作物推荐功能
- [ ] 天气预报功能
- [ ] 自动定位功能
- [ ] 手动城市选择

### 预期行为
1. **首次访问**：默认显示作物推荐页面
2. **天气预报**：自动定位或默认北京
3. **页面切换**：点击天气预报按钮自动刷新
4. **错误处理**：优雅的错误提示

## 🎯 部署成功指标
- ✅ 应用在30秒内启动
- ✅ 所有模型文件加载成功
- ✅ 天气API正常工作
- ✅ 用户界面响应流畅

## 📞 技术支持
如果遇到问题，请检查：
1. Streamlit Cloud部署日志
2. GitHub仓库文件完整性
3. 依赖版本兼容性
