import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet


class Detector(nn.Module):
    def __init__(self, name="efficientnet-b4"):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained(name, advprop=True, num_classes=2)

    def forward(self, x):
        x = self.net(x)
        return x


def load_efficient(name="efficientnet-b4", device="cuda", weight_path=None):
    """
    加载 SBI 训练的 Detector 模型
    """
    model = Detector(name)
    model = model.to(device)

    cnn_sd = torch.load(weight_path, map_location=device)["model"]

    model.load_state_dict(cnn_sd)
    model.eval()
    return model
