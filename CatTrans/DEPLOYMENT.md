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

## 🔧 模型文件路径问题解决

### ❌ 常见错误
```
❌ 模型加载失败: [Errno 2] No such file or directory: 'catboost_model.pkl'
```

### ✅ 解决方案
**问题原因：** Streamlit Cloud的工作目录可能与本地不同

**修复方法：**
1. **使用绝对路径**：应用已更新为使用`os.path.dirname(os.path.abspath(__file__))`获取脚本目录
2. **多重路径检查**：先检查脚本目录，再检查当前工作目录
3. **调试信息**：显示实际的文件路径，便于排查问题

### 🔍 调试信息
部署后应用会显示：
```
🔍 当前工作目录: /mount/src/your-app
🔍 脚本目录: /mount/src/your-app
🔍 查找CatBoost模型: /mount/src/your-app/catboost_model.pkl
🔍 查找Transformer模型: /mount/src/your-app/transformer_model.pth
🔍 查找标签编码器: /mount/src/your-app/label_encoder.pkl
```

## 🚀 部署步骤

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
1. ✅ 应用已更新为使用绝对路径
2. ✅ 添加了多重路径检查
3. ✅ 显示详细的调试信息
4. ✅ 如果仍然失败，检查GitHub仓库中的文件

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
- [ ] 查看调试信息（确认路径正确）
- [ ] 模型加载成功（显示绿色✅）
- [ ] 侧边栏按钮工作
- [ ] 作物推荐功能
- [ ] 天气预报功能
- [ ] 自动定位功能
- [ ] 手动城市选择

### 预期行为
1. **首次访问**：显示调试信息和模型加载状态
2. **模型加载**：显示三个✅成功消息
3. **天气预报**：自动定位或默认北京
4. **页面切换**：点击天气预报按钮自动刷新
5. **错误处理**：优雅的错误提示

## 🎯 部署成功指标
- ✅ 应用在30秒内启动
- ✅ 调试信息显示正确路径
- ✅ 所有模型文件加载成功
- ✅ 天气API正常工作
- ✅ 用户界面响应流畅

## 📞 故障排除

### 如果模型仍然无法加载：
1. **检查GitHub仓库**：确认3个模型文件已上传
2. **查看部署日志**：在Streamlit Cloud查看详细错误
3. **检查文件大小**：确保每个文件 < 200MB
4. **重新部署**：删除应用后重新创建

### 调试步骤：
1. 部署后查看页面上的调试信息
2. 确认文件路径是否正确
3. 如果路径错误，可能需要调整仓库结构
4. 检查Streamlit Cloud的工作目录设置

## 📋 最终检查清单

部署前确认：
- [ ] 所有3个模型文件在GitHub仓库中
- [ ] requirements.txt包含所有依赖
- [ ] .streamlit/config.toml存在
- [ ] app.py已更新为使用绝对路径
- [ ] 本地测试无错误

部署后检查：
- [ ] 调试信息显示正确路径
- [ ] 三个模型都显示✅加载成功
- [ ] 作物推荐功能正常
- [ ] 天气预报功能正常
