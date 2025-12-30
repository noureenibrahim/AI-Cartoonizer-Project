import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import os
# ==========================================
# PART 1: CLASSICAL CARTOONIZER (ETHAR AND ARWA)
# ==========================================
def classical_cartoonize_image(image):
    """
    Uses Bilateral Filter + Adaptive Thresholding
    """
    # 1. Edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    # 2. Color Smoothing
    color = image.copy()
    for _ in range(5):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=200, sigmaSpace=200)

    # 3. Combine
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.addWeighted(src1=color, alpha=0.5, src2=edges_rgb, beta=0.5, gamma=0.0)
    return cartoon

# ==========================================
# PART 1: NEURAL NETWORK CLASSES (From your friend's code)
# ==========================================
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=0)
        self.gn = nn.GroupNorm(1, out_ch)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.res1 = ConvLayer(channels, channels, 3, 1, 1)
        self.res2 = ConvLayer(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.res2(self.res1(x))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.block_a = nn.Sequential(
            ConvLayer(3, 32, 7, 1, 3),
            ConvLayer(32, 64, 3, 2, 1),
            ConvLayer(64, 64, 3, 1, 1)
        )
        self.block_b = nn.Sequential(
            ConvLayer(64, 128, 3, 2, 1),
            ConvLayer(128, 128, 3, 1, 1)
        )
        self.block_c_list = [ConvLayer(128, 128, 3, 1, 1)]
        for i in range(8):
            self.block_c_list.append(ResBlock(128))
        self.block_c_list.append(ConvLayer(128, 128, 3, 1, 1))
        self.block_c = nn.Sequential(*self.block_c_list)
        self.block_d = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvLayer(128, 64, 3, 1, 1),
            ConvLayer(64, 64, 3, 1, 1)
        )
        self.block_e = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvLayer(64, 32, 3, 1, 1),
            ConvLayer(32, 32, 3, 1, 1),
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, 7, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        return x


# ==========================================
# PART 2: HELPER FUNCTIONS
# ==========================================

_ai_model = None


def load_ai_model():
    global _ai_model
    if _ai_model is not None:
        return _ai_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Generator().to(device)

    # We look for paprika.pt in the current folder
    weights_path = "paprika.pt"

    if os.path.exists(weights_path):
        try:
            # Safe loading with CPU support
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            current_model_dict = model.state_dict()
            new_state_dict = {}
            for key in checkpoint.keys():
                if key in current_model_dict:
                    if checkpoint[key].shape == current_model_dict[key].shape:
                        new_state_dict[key] = checkpoint[key]

            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            _ai_model = model
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None


def flat_toon_processor(frame):
    # 1. K-Means Color Quantization
    pixel_data = frame.reshape((-1, 3))
    pixel_data = np.float32(pixel_data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    num_colors = 12

    ret, label, center = cv2.kmeans(pixel_data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    flat_img = res.reshape((frame.shape))

    # 2. Bilateral Filtering
    smooth_img = cv2.bilateralFilter(flat_img, d=9, sigmaColor=200, sigmaSpace=200)

    # 3. Create Cartoon Outlines
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    cartoon_edges = cv2.adaptiveThreshold(img_blur, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 9, 3)

    # Combine
    final_output = cv2.bitwise_and(smooth_img, smooth_img, mask=cartoon_edges)

    return final_output


# ==========================================
# PART 3: MAIN FUNCTIONS FOR UI
# ==========================================

def classical_cartoonize_image(image):
    # Member 1's basic logic (kept for the "Classical" button)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    color = image.copy()
    for _ in range(5):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=200, sigmaSpace=200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(src1=color, alpha=0.5, src2=edges_rgb, beta=0.5, gamma=0.0)


def ai_cartoonize_image(image):
    model = load_ai_model()

    if model is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # AI step: Resize
        height, width = image.shape[:2]
        new_h = (height // 32) * 32
        new_w = (width // 32) * 32
        img_input = cv2.resize(image, (new_w, new_h))

        # Convert to Tensor
        img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)
        img_tensor = img_tensor * 2 - 1

        # Run through network
        with torch.no_grad():
            prediction = model(img_tensor)
            prediction = (prediction.squeeze(0).cpu() + 1) / 2

    # Apply the classical CV flat processing to the ORIGINAL image
    final_result = flat_toon_processor(image)

    return final_result