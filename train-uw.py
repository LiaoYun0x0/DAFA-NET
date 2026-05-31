import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# scikit-learn 评价指标
from sklearn.metrics import precision_score, recall_score, f1_score

# 导入您自己的模块
from dataset.uw import UWDataset
from dataset.rs import RSFakeDataset
from model.mymodel import DINOFreqDualFusionClassifier
# from model.dino_backbone import MyDinoV3Model
# from model.efficientnet.model import Detector

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

# 路径设置
DATA_DIR = 'D://z_twb//Data//anti-deepfake-data and code//data_split'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
MODEL_SAVE_PATH = 'checkpoints'
LOG_FILE = 'logs/training_log.csv'

# 训练超参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "mymodel-v2"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
NUM_WORKERS = 1
SEED = 42


def set_seed(seed=42):
    """
    固定所有随机种子以确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 保证 cudnn 算法的确定性 (会稍微降低训练速度，但保证结果一致)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"已设置随机种子: {seed}")


def train_model():
    """完整的主训练和验证函数"""
    set_seed(SEED)

    print(f"--- 开始训练 ---")
    print(f"使用设备: {DEVICE}")

    # --- 2. 准备数据集和 DataLoader ---
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # 确保日志目录存在
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    train_dataset = UWDataset(root_dir=TRAIN_DIR, augment_other=True)
    val_dataset = UWDataset(root_dir=VAL_DIR, augment_other=False)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f"错误：训练集或验证集为空。")
        return

    print(f"训练集图像数: {len(train_dataset)}")
    print(f"验证集图像数: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # --- 3. 初始化模型、损失函数和优化器 ---
    # model = Detector(name=MODEL_NAME).to(DEVICE)
    # model = MyDinoV3Model(
    #     weights_path="D://z_twb//dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    #     num_classes=2,
    #     model_name='base'
    # ).to(DEVICE)
    TEST_WEIGHTS = "D://z_twb//dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    # 实例化您的主干 (确保它支持 get_intermediate_layers)
    model = DINOFreqDualFusionClassifier(weights_path=TEST_WEIGHTS, freq_ch=128, attn_embed=256, heads=4,
                                         num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 训练和验证循环 ---
    best_val_f1 = 0.0
    history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # ===========================
        #        训练阶段
        # ===========================
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        train_all_labels = []
        train_all_preds = []

        progress_bar = tqdm(train_loader, desc="训练中", unit="batch")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += labels.size(0)

            train_all_labels.extend(labels.cpu().numpy())
            train_all_preds.extend(preds.cpu().numpy())

            progress_bar.set_postfix(
                loss=running_loss / total_preds,
                acc=correct_preds.double().item() / total_preds
            )

        # 计算训练集指标 (添加 zero_division 防止警告)
        train_loss = running_loss / total_preds
        train_acc = correct_preds.double() / total_preds
        train_precision = precision_score(train_all_labels, train_all_preds, average='binary', zero_division=0)
        train_recall = recall_score(train_all_labels, train_all_preds, average='binary', zero_division=0)
        train_f1 = f1_score(train_all_labels, train_all_preds, average='binary', zero_division=0)

        print(f"Epoch {epoch + 1} 训练: Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

        # ===========================
        #        验证阶段
        # ===========================
        model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        val_all_labels = []
        val_all_preds = []

        progress_bar_val = tqdm(val_loader, desc="验证中", unit="batch")

        with torch.no_grad():
            for inputs, labels in progress_bar_val:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels.data)
                total_preds += labels.size(0)

                val_all_labels.extend(labels.cpu().numpy())
                val_all_preds.extend(preds.cpu().numpy())

        # 计算验证集指标
        val_loss = running_loss / total_preds
        val_acc = correct_preds.double() / total_preds
        val_precision = precision_score(val_all_labels, val_all_preds, average='binary', zero_division=0)
        val_recall = recall_score(val_all_labels, val_all_preds, average='binary', zero_division=0)
        val_f1 = f1_score(val_all_labels, val_all_preds, average='binary', zero_division=0)

        print(f"Epoch {epoch + 1} 验证: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # ===========================
        #        模型保存
        # ===========================

        # 1. 保存最佳模型 (Best F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_best_f1.pth")
            torch.save({'model': model.state_dict()}, save_path)
            print(f"*** 新的最佳模型(F1)已保存到 {save_path} (F1: {best_val_f1:.4f}) ***")

        # 2. 保存最新模型 (Latest / Last Epoch) - 总是执行
        last_save_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_last.pth")
        torch.save({'model': model.state_dict()}, last_save_path)
        # print(f"已更新最新模型: {last_save_path}") # 可以注释掉以免刷屏

        # ===========================
        #        日志记录
        # ===========================
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc.item(),  # 转换为纯数字
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc.item(),  # 转换为纯数字
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        }
        history.append(epoch_data)

        try:
            df = pd.DataFrame(history)
            df.to_csv(LOG_FILE, index=False)
            print(f"日主已记录到 CSV 文件中")
        except Exception as e:
            print(f"警告：保存 CSV 日志失败: {e}")

    print("\n--- 训练完成 ---")
    print(f"最佳验证 F1 分数: {best_val_f1:.4f}")
    print(f"最佳模型保存在: {os.path.join(MODEL_SAVE_PATH, f'{MODEL_NAME}_best_f1.pth')}")
    print(f"最新模型保存在: {os.path.join(MODEL_SAVE_PATH, f'{MODEL_NAME}_last.pth')}")
    print(f"训练日志已保存到: {LOG_FILE}")


if __name__ == '__main__':
    train_model()
