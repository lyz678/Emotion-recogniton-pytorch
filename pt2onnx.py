import torch
import torchvision.models as models
from models.cnn import *  # 假设你的 MiniXCEPTION 模型在 models/cnn.py 中定义
import onnx
from onnxsim import simplify

input_shape = (1, 48, 48)
num_classes = 7

# 创建一个空模型实例
model = ResNet18(input_shape, num_classes)  # MiniXception ResNet18 

# 加载参数
checkpoint = torch.load('./models/best_model.pth')
model.load_state_dict(checkpoint)

# 将模型设置为评估模式
model.eval()

# 定义示例输入，根据模型的输入形状来定义
dummy_input = torch.randn(1, 1, 48, 48)  # 1张图片，1通道（灰度图），大小为 48x48

# 导出模型为 ONNX 格式
torch.onnx.export(model, dummy_input, "ResNet18.onnx", verbose=True)

# 读取原始的 ONNX 模型
onnx_model = onnx.load("ResNet18.onnx")

# 简化模型
simplified_model, check = simplify(onnx_model)

# 将简化后的模型保存到文件
onnx.save(simplified_model, "ResNet18.sim.onnx")