from .a_person_mask_generator_comfyui import APersonMaskGenerator
from .a_person_face_landmark_mask_generator_comfyui import APersonFaceLandmarkMaskGenerator
from .alpha_to_solid_background import AlphaToSolidBackground

NODE_CLASS_MAPPINGS = {
    "APersonMaskGenerator": APersonMaskGenerator,
    "APersonFaceLandmarkMaskGenerator": APersonFaceLandmarkMaskGenerator,
    "AlphaToSolidBackground": AlphaToSolidBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APersonMaskGenerator": "A Person Mask Generator",
    "APersonFaceLandmarkMaskGenerator": "A Person Face Landmark Mask Generator",
    "AlphaToSolidBackground": "Alpha to Solid Background",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
