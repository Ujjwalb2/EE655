import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== PROGRESS TRACKER =====
progress = {
    "step": 0,
    "total": 1,
    "status": "idle"
}

# ===== CONFIG =====
IMAGE_SIZE = 512 if torch.cuda.is_available() else 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

loader = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

unloader = transforms.Compose([
    transforms.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD]
    ),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# ===== VGG =====
class VGG19Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

        self.style_layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        self.content_layers = {'21': 'conv4_2'}

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.style_layers:
                features[self.style_layers[name]] = x
            if name in self.content_layers:
                features[self.content_layers[name]] = x
        return features

vgg = VGG19Features()

# ===== LOSSES =====
def gram_matrix(feature_maps):
    b, c, h, w = feature_maps.shape
    F = feature_maps.view(b, c, h * w)
    G = torch.bmm(F, F.transpose(1, 2))
    return G / (c * h * w)

class TotalVariationLoss(nn.Module):
    def forward(self, img):
        h_var = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        v_var = torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        return (h_var + v_var) / img.numel()

# ===== CONFIG =====
CONFIG = {
    'content_weight': 1e4,
    'style_weight': 1e7,
    'tv_weight': 1e-2,
    'num_steps': 300
}

# ===== STYLE TRANSFER =====
def run_style_transfer(content_img, style_img):
    progress["step"] = 0
    progress["total"] = CONFIG['num_steps']
    progress["status"] = "running"

    tv_loss_fn = TotalVariationLoss().to(device)

    with torch.no_grad():
        content_features = vgg(content_img)
        style_features = vgg(style_img)

    input_img = content_img.clone().requires_grad_(True)

    optimizer = optim.LBFGS([input_img])

    run = [0]
    while run[0] <= CONFIG['num_steps']:
        def closure():
            optimizer.zero_grad()

            features = vgg(input_img)

            content_loss = torch.mean(
                (features['conv4_2'] - content_features['conv4_2']) ** 2
            )

            style_loss = 0
            for layer in ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']:
                G = gram_matrix(features[layer])
                A = gram_matrix(style_features[layer])
                style_loss += torch.mean((G - A) ** 2)

            tv_loss = tv_loss_fn(input_img)

            loss = (
                CONFIG['content_weight'] * content_loss +
                CONFIG['style_weight'] * style_loss +
                CONFIG['tv_weight'] * tv_loss
            )

            loss.backward()

            run[0] += 1
            progress["step"] = run[0]

            return loss

        optimizer.step(closure)

    progress["status"] = "done"

    return input_img


def stylize_image(content_path, style_path, output_path):
    content = load_image(content_path)
    style = load_image(style_path)

    output = run_style_transfer(content, style)

    output = unloader(output.squeeze(0).cpu())
    save_image(output, output_path)