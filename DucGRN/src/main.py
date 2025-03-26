import argparse
import torch
from src.utils.config import load_config
from src.train import train
from src.test import test

def main():
    # 读取配置
    config = load_config()
    DEVICE = torch.device(config["device"])

    print(f"[INFO] 使用设备: {DEVICE}")

    # 创建命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        print("[INFO] 开始模型训练...")
        train(config["training"]["epochs"])
        print("[INFO] 训练完成！")
    elif args.mode == 'test':
        print("[INFO] 开始模型测试...")
        test()
        print("[INFO] 测试完成！")

if __name__ == "__main__":
    main()
