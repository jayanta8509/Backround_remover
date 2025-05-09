import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Normalize


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size = im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    im_tensor = torch.divide(im_tensor, 255.0)
    
    # Corrected normalization using torch
    normalizer = Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])  # Create normalizer instance
    im_tensor = normalizer(im_tensor)
    
    return im_tensor

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array