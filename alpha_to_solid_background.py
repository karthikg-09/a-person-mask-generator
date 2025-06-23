import torch
import numpy as np
from PIL import Image


class AlphaToSolidBackground:
    """
    Simple node to remove alpha channel and replace transparency with solid color background.
    Perfect for preparing transparent cutout images for person mask generators.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "background_color": (["white", "black", "gray", "custom"], {"default": "white"}),
            },
            "optional": {
                "custom_r": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "custom_g": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "custom_b": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_background",)
    FUNCTION = "remove_alpha_add_background"
    CATEGORY = "🎭 Alpha Processing"

    def remove_alpha_add_background(
        self,
        image,
        background_color: str = "white",
        custom_r: int = 255,
        custom_g: int = 255,
        custom_b: int = 255,
    ):
        """Remove alpha channel and add solid color background"""
        
        # Convert tensor to PIL image
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            i = 255.0 * image.cpu().numpy()
            pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        else:
            pil_image = image
        
        # Get background color
        if background_color == "white":
            bg_color = (255, 255, 255)
        elif background_color == "black":
            bg_color = (0, 0, 0)
        elif background_color == "gray":
            bg_color = (128, 128, 128)
        else:  # custom
            bg_color = (custom_r, custom_g, custom_b)
        
        # Handle different image modes
        if pil_image.mode == 'RGBA':
            # Create background with same size
            background = Image.new('RGB', pil_image.size, bg_color)
            
            # Composite image over background using alpha channel
            result = Image.alpha_composite(
                background.convert('RGBA'), 
                pil_image
            ).convert('RGB')
            
        elif pil_image.mode == 'LA':  # Grayscale with alpha
            # Convert to RGBA first
            rgba_image = pil_image.convert('RGBA')
            background = Image.new('RGB', pil_image.size, bg_color)
            
            result = Image.alpha_composite(
                background.convert('RGBA'), 
                rgba_image
            ).convert('RGB')
            
        else:
            # No alpha channel, just convert to RGB if needed
            result = pil_image.convert('RGB')
        
        # Convert back to tensor
        result_array = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_array).unsqueeze(0)
        
        return (result_tensor,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AlphaToSolidBackground": AlphaToSolidBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaToSolidBackground": "Alpha to Solid Background",
} 