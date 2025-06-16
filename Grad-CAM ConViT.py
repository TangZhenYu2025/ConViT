
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image


import timm
def grad_cam(model, image, target_layer):
    model.eval()

    # Hook the feature map
    def hook_fn(module, input, output):
        global feature_map
        feature_map = output

    hook = target_layer.register_forward_hook(hook_fn)

    # Forward pass
    output = model(image.unsqueeze(0))
    class_idx = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    loss = output[0, class_idx]
    loss.backward()

    # Get gradients and feature maps
    gradients = target_layer.weight.grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activation_map = feature_map[0].detach().cpu().numpy()

    # Convert pooled_gradients to numpy array
    pooled_gradients = pooled_gradients.detach().cpu().numpy()

    for i in range(len(pooled_gradients)):
        activation_map[i, :, :] *= pooled_gradients[i]

    heatmap = np.mean(activation_map, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    hook.remove()
    return heatmap

def show_grad_cam(image, heatmap):
    image = image.permute(1, 2, 0).numpy()
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.show()


# Example usage
pretrained_cfg=timm.create_model('ConViT').default_cfg
pretrained_cfg['file']='/tmp/pycharm_project_744/porous media/GradCAM/CNNKAN/pre_path_ok/ConViT.pth'
# model_convit = timm.create_model('convit_tiny', pretrained=True, num_classes=1,pretrained_cfg=pretrained_cfg,global_pool=
model = timm.create_model('convit_tiny', pretrained=True,pretrained_cfg=pretrained_cfg, num_classes=1000,global_pool='avg')
model_name='convit_tiny'
print(model)
# model.head = nn.Sequential(
#     nn.ReLU(),
#     nn.BatchNorm1d(192),
#     nn.Dropout(0.1),
#     nn.Linear(in_features=192, out_features=1, bias=False),
#     )

# 加载保存的模型权重
weight_path = '/tmp/pycharm_project_744/porous media/GradCAM/CNNKAN/pre_path_ok/convit_tiny.pth'  # 替换为你权重文件的路径
model.load_state_dict(torch.load(weight_path))


target_layer = model.blocks[1].norm1 # You may need to adjust the layer
image_path = 'img.png'  # Replace with your image path
image = Image.open(image_path).convert('RGB')

# Transform image to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize if necessary
    transforms.ToTensor(),
])
image = transform(image)
# image = transforms.ToTensor()('img.png')  # Replace 'your_image' with your image
heatmap = grad_cam(model, image, target_layer)
show_grad_cam(image, heatmap)