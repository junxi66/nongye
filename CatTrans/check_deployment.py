#!/usr/bin/env python3
"""
Streamlit Cloud 部署前检查脚本
检查所有必需文件是否存在并可访问
"""

import os
import sys

def check_deployment_files():
    """检查部署必需的文件"""
    print("🔍 检查Streamlit Cloud部署文件...")
    
    # 必需的文件列表
    required_files = [
        'app.py',
        'catboost_model.pkl',
        'transformer_model.pth', 
        'label_encoder.pkl',
        'requirements.txt',
        '.streamlit/config.toml'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            existing_files.append(f"✅ {file_path} ({size:,} bytes)")
        else:
            missing_files.append(f"❌ {file_path}")
    
    print("\n📋 文件检查结果:")
    for file_info in existing_files:
        print(file_info)
    
    if missing_files:
        print("\n⚠️  缺失文件:")
        for file_info in missing_files:
            print(file_info)
        return False
    else:
        print("\n✅ 所有必需文件都存在!")
        return True

def check_model_files():
    """检查模型文件的完整性"""
    print("\n🔍 检查模型文件完整性...")
    
    try:
        import joblib
        import torch
        
        # 检查CatBoost模型
        try:
            catboost_model = joblib.load('catboost_model.pkl')
            print("✅ CatBoost模型加载成功")
        except Exception as e:
            print(f"❌ CatBoost模型加载失败: {e}")
            return False
        
        # 检查标签编码器
        try:
            label_encoder = joblib.load('label_encoder.pkl')
            print("✅ 标签编码器加载成功")
        except Exception as e:
            print(f"❌ 标签编码器加载失败: {e}")
            return False
        
        # 检查Transformer模型
        try:
            transformer_state = torch.load('transformer_model.pth', map_location='cpu')
            print("✅ Transformer模型加载成功")
        except Exception as e:
            print(f"❌ Transformer模型加载失败: {e}")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ 导入依赖失败: {e}")
        return False

def check_requirements():
    """检查requirements.txt"""
    print("\n🔍 检查requirements.txt...")
    
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        print("✅ requirements.txt内容:")
        print(requirements)
        return True
    else:
        print("❌ requirements.txt不存在")
        return False

def main():
    """主检查函数"""
    print("🚀 Streamlit Cloud 部署检查开始\n")
    
    # 检查文件存在性
    files_ok = check_deployment_files()
    
    # 检查模型文件
    if files_ok:
        models_ok = check_model_files()
    else:
        models_ok = False
    
    # 检查requirements
    req_ok = check_requirements()
    
    print("\n" + "="*50)
    if files_ok and models_ok and req_ok:
        print("🎉 所有检查通过! 可以部署到Streamlit Cloud")
        print("\n📝 部署步骤:")
        print("1. 推送所有文件到GitHub")
        print("2. 在Streamlit Cloud中连接仓库")
        print("3. 选择app.py作为主文件")
        print("4. 点击Deploy")
    else:
        print("❌ 检查失败! 请修复问题后重试")
        sys.exit(1)

if __name__ == "__main__":
    main()
