import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 导入您的模块
from dataset.uw import UWDataset
from model.dino_backbone import MyDinoV3Model
from model.mymodel import DINOFreqDualFusionClassifier
from model.efficientnet.model import Detector

# --- 路径修复  ---
# 获取当前文件 (train.py) 的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 'model' 文件夹的路径
model_dir = os.path.join(current_dir, 'model')
# 将 'model' 文件夹添加到 Python 搜索路径中
# 这样 python 就能直接找到 'dinov3' 包，满足它内部的绝对导入需求
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# --- 1. 配置参数 ---

# 数据路径 (与 train.py 保持一致)
DATA_DIR = 'D://z_twb//Data//anti-deepfake-data and code//data_split'
# *** 注意：您要求测试“训练集”，所以这里指向 train 文件夹 ***
# 如果想测试验证集，请改为: os.path.join(DATA_DIR, 'val')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

# 模型权重路径
# 您可以选择加载 'efficientnet-b4_best_f1.pth' 或 'efficientnet-b4_last.pth'
# WEIGHT_PATH = 'checkpoints/dinov3-base16_last.pth'
WEIGHT_PATH = 'checkpoints/mymodel-v2_last.pth'

# 参数设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "efficientnet-b4"
BATCH_SIZE = 16  # 测试时显存占用较少，可以适当调大
NUM_WORKERS = 1


def test_model():
    print(f"--- 开始测试 ---")
    print(f"测试设备: {DEVICE}")
    print(f"测试数据集路径: {TEST_DATA_DIR}")
    print(f"加载权重: {WEIGHT_PATH}")

    # --- 2. 准备数据集 ---
    # 注意：测试时通常关闭 augment_other (随机翻转等)，以获得确定性的结果
    test_dataset = UWDataset(
        root_dir=TEST_DATA_DIR, 
        augment_other=False,
        blur_sigma=5
    )

    if len(test_dataset) == 0:
        print(f"错误：未在 {TEST_DATA_DIR} 找到数据。")
        return

    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --- 3. 加载模型 ---
    TEST_WEIGHTS = "D://z_twb//dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    # efficient net b4
    # model = Detector(name=MODEL_NAME).to(DEVICE)

    # dino v3
    # model = MyDinoV3Model(
    #     weights_path="D://z_twb//dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    #     num_classes=2,
    #     model_name='base'
    # ).to(DEVICE)

    # my model
    model = DINOFreqDualFusionClassifier(weights_path=TEST_WEIGHTS, freq_ch=128, attn_embed=256, heads=4,
                                         num_classes=2).to(DEVICE)
    #

    if os.path.exists(WEIGHT_PATH):
        # 加载权重
        # 注意：train.py 保存时使用的是 {'model': model.state_dict()}
        checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
        
        # 兼容处理：检查 checkpoint 是直接的 state_dict 还是包含 'model' 键的字典
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
            
        print("模型权重加载成功！")
    else:
        print(f"错误：找不到权重文件 {WEIGHT_PATH}，请检查路径。")
        return

    # --- 4. 推理循环 ---
    model.eval() # 切换到评估模式
    
    all_preds = []
    all_labels = []
    
    # 进度条
    progress_bar = tqdm(test_loader, desc="推理中", unit="batch")

    with torch.no_grad(): # 不计算梯度
        for inputs, labels in progress_bar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # 前向传播
            outputs = model(inputs)
            
            # 获取预测结果 (0 或 1)
            _, preds = torch.max(outputs, 1)

            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 5. 计算并打印指标 ---
    print("\n--- 测试结果 ---")
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    print(f"测试样本总数: {len(all_labels)}")
    print(f"{'Acc (准确率)':<15}: {accuracy:.4f}")
    print(f"{'Precision':<15}: {precision:.4f}")
    print(f"{'Recall':<15}: {recall:.4f}")
    print(f"{'F1 Score':<15}: {f1:.4f}")

    # 额外的混淆矩阵信息
    # 统计真实标签中 0 和 1 的数量，以及预测中 0 和 1 的数量
    import numpy as np
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    print("\n--- 详细统计 ---")
    print(f"真实数据分布 -> Authentic(0): {np.sum(all_labels == 0)}, Fake(1): {np.sum(all_labels == 1)}")
    print(f"模型预测分布 -> Authentic(0): {np.sum(all_preds == 0)}, Fake(1): {np.sum(all_preds == 1)}")


if __name__ == '__main__':
    test_model()
