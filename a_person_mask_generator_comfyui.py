import math
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from functools import reduce
import cv2
import torch
import numpy as np
from PIL import Image
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import folder_paths


def get_a_person_mask_generator_model_path() -> str:
    model_folder_name = "mediapipe"
    model_name = "selfie_multiclass_256x256.tflite"

    model_folder_path = os.path.join(folder_paths.models_dir, model_folder_name)
    model_file_path = os.path.join(model_folder_path, model_name)

    if not os.path.exists(model_file_path):
        import wget

        model_url = f"https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/{model_name}"
        print(f"Downloading '{model_name}' model")
        os.makedirs(model_folder_path, exist_ok=True)
        wget.download(model_url, model_file_path)

    return model_file_path


class APersonMaskGenerator:

    def __init__(self):
        # download the model if we need it
        get_a_person_mask_generator_model_path()

    @classmethod
    def INPUT_TYPES(self):
        false_widget = (
            "BOOLEAN",
            {"default": False, "label_on": "enabled", "label_off": "disabled"},
        )
        true_widget = (
            "BOOLEAN",
            {"default": True, "label_on": "enabled", "label_off": "disabled"},
        )

        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "face_mask": true_widget,
                "background_mask": false_widget,
                "hair_mask": false_widget,
                "body_mask": false_widget,
                "clothes_mask": false_widget,
                "eyeglasses_mask": false_widget,
                "hat_mask": false_widget,
                "earrings_mask": false_widget,
                "other_accessories_mask": false_widget,
                "confidence": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "refine_mask": true_widget,
            },
        }

    CATEGORY = "A Person Mask Generator - David Bielejeski"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)

    FUNCTION = "generate_mask"

    def get_mediapipe_image(self, image: Image) -> mp.Image:
        # Convert image to NumPy array
        numpy_image = np.asarray(image)

        image_format = mp.ImageFormat.SRGB

        # Convert BGR to RGB (if necessary)
        if numpy_image.shape[-1] == 4:
            image_format = mp.ImageFormat.SRGBA
        elif numpy_image.shape[-1] == 3:
            image_format = mp.ImageFormat.SRGB
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

        return mp.Image(image_format=image_format, data=numpy_image)

    def get_bbox_for_mask(self, mask_image: Image):
        # Convert the image to grayscale
        grayscale_image = mask_image.convert("L")

        # Convert the PIL image to a NumPy array
        image_array = np.array(grayscale_image)

        # Find contours in the binary image
        contours, _ = cv2.findContours(
            image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Find the bounding box of the largest contour (assuming it's the main subject)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            return (x, y, x + w, y + h)

        return None

    def _filter_accessories_by_type(self, accessories_mask: np.ndarray, image: Image, 
                                   eyeglasses: bool, hat: bool, earrings: bool, 
                                   other_accessories: bool) -> np.ndarray:
        """
        Filter accessories mask to specific types using position and context analysis
        """
        if not any([eyeglasses, hat, earrings, other_accessories]):
            return np.zeros_like(accessories_mask)
            
        if other_accessories:
            # Return full accessories mask if "other_accessories" is enabled
            return accessories_mask
            
        # Get image dimensions
        height, width = accessories_mask.shape
        
        # Create filtered mask
        filtered_mask = np.zeros_like(accessories_mask)
        
        # Find connected components in accessories mask
        from scipy import ndimage
        labeled_mask, num_components = ndimage.label(accessories_mask > 0.3)
        
        for component_id in range(1, num_components + 1):
            component_mask = (labeled_mask == component_id)
            
            # Get component properties
            component_coords = np.where(component_mask)
            if len(component_coords[0]) == 0:
                continue
                
            min_y, max_y = np.min(component_coords[0]), np.max(component_coords[0])
            min_x, max_x = np.min(component_coords[1]), np.max(component_coords[1])
            center_y = (min_y + max_y) // 2
            center_x = (min_x + max_x) // 2
            
            # Normalize positions (0-1)
            norm_center_y = center_y / height
            norm_center_x = center_x / width
            norm_height = (max_y - min_y) / height
            norm_width = (max_x - min_x) / width
            
            # Classification logic based on position and size
            component_area = np.sum(component_mask)
            
            # EYEGLASSES: Central face area, horizontal shape, medium size
            if eyeglasses and (0.25 < norm_center_y < 0.65) and (0.2 < norm_center_x < 0.8):
                if norm_width > norm_height * 1.2 and 0.02 < component_area / (height * width) < 0.15:
                    filtered_mask += component_mask.astype(np.float32)
                    continue
            
            # HAT: Top area of image, larger size
            if hat and norm_center_y < 0.4:
                if component_area / (height * width) > 0.01:
                    filtered_mask += component_mask.astype(np.float32)
                    continue
                    
            # EARRINGS: Side areas near face level, small size
            if earrings and (0.3 < norm_center_y < 0.7):
                if (norm_center_x < 0.3 or norm_center_x > 0.7) and component_area / (height * width) < 0.01:
                    filtered_mask += component_mask.astype(np.float32)
                    continue
        
        return np.clip(filtered_mask, 0, 1)

    def __get_mask(
            self,
            image: Image,
            segmenter,
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            eyeglasses_mask: bool,
            hat_mask: bool,
            earrings_mask: bool,
            other_accessories_mask: bool,
            confidence: float,
            refine_mask: bool,
    ) -> Image:
        # Retrieve the masks for the segmented image
        media_pipe_image = self.get_mediapipe_image(image=image)
        if any([face_mask, background_mask, hair_mask, body_mask, clothes_mask, 
                eyeglasses_mask, hat_mask, earrings_mask, other_accessories_mask]):
            segmented_masks = segmenter.segment(media_pipe_image)

        # https://developers.google.com/mediapipe/solutions/vision/image_segmenter#multiclass-model
        # 0 - background
        # 1 - hair
        # 2 - body - skin
        # 3 - face - skin
        # 4 - clothes
        # 5 - others(accessories)
        masks = []
        if background_mask:
            masks.append(segmented_masks.confidence_masks[0])
        if hair_mask:
            masks.append(segmented_masks.confidence_masks[1])
        if body_mask:
            masks.append(segmented_masks.confidence_masks[2])
        if face_mask:
            masks.append(segmented_masks.confidence_masks[3])
        if clothes_mask:
            masks.append(segmented_masks.confidence_masks[4])
            
        # Handle accessories with smart filtering
        if any([eyeglasses_mask, hat_mask, earrings_mask, other_accessories_mask]):
            if len(segmented_masks.confidence_masks) > 5:
                accessories_raw = segmented_masks.confidence_masks[5].numpy_view()
                filtered_accessories = self._filter_accessories_by_type(
                    accessories_raw, image, 
                    eyeglasses_mask, hat_mask, earrings_mask, other_accessories_mask
                )
                
                # Convert filtered accessories back to MediaPipe mask format
                class FilteredAccessoriesMask:
                    def __init__(self, array):
                        self._array = array
                    def numpy_view(self):
                        return self._array
                
                masks.append(FilteredAccessoriesMask(filtered_accessories))

        image_data = media_pipe_image.numpy_view()
        image_shape = image_data.shape

        # convert the image shape from "rgb" to "rgba" aka add the alpha channel
        if image_shape[-1] == 3:
            image_shape = (image_shape[0], image_shape[1], 4)

        mask_background_array = np.zeros(image_shape, dtype=np.uint8)
        mask_background_array[:] = (0, 0, 0, 255)

        mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
        mask_foreground_array[:] = (255, 255, 255, 255)

        mask_arrays = []

        if len(masks) == 0:
            mask_arrays.append(mask_background_array)
        else:
            for i, mask in enumerate(masks):
                condition = (
                        np.stack((mask.numpy_view(),) * image_shape[-1], axis=-1)
                        > confidence
                )
                mask_array = np.where(
                    condition, mask_foreground_array, mask_background_array
                )
                mask_arrays.append(mask_array)

        # Merge our masks taking the maximum from each
        merged_mask_arrays = reduce(np.maximum, mask_arrays)

        # Create the image
        mask_image = Image.fromarray(merged_mask_arrays)

        # refine the mask by zooming in on the area where we detected our segments
        if refine_mask:
            bbox = self.get_bbox_for_mask(mask_image=mask_image)
            if bbox != None:
                cropped_image_pil = image.crop(bbox)

                cropped_mask_image = self.__get_mask(image=cropped_image_pil,
                                                   segmenter=segmenter,
                                                   face_mask=face_mask,
                                                   background_mask=background_mask,
                                                   hair_mask=hair_mask,
                                                   body_mask=body_mask,
                                                   clothes_mask=clothes_mask,
                                                   eyeglasses_mask=eyeglasses_mask,
                                                   hat_mask=hat_mask,
                                                   earrings_mask=earrings_mask,
                                                   other_accessories_mask=other_accessories_mask,
                                                   confidence=confidence,
                                                   refine_mask=False,
                                                   )

                updated_mask_image = Image.new('RGBA', image.size, (0, 0, 0))
                updated_mask_image.paste(cropped_mask_image, bbox)
                mask_image = updated_mask_image

        return mask_image

    def get_mask_images(
            self,
            images, # tensors
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            eyeglasses_mask: bool,
            hat_mask: bool,
            earrings_mask: bool,
            other_accessories_mask: bool,
            confidence: float,
            refine_mask: bool,
    ) -> list[Image]:
        a_person_mask_generator_model_path = get_a_person_mask_generator_model_path()
        a_person_mask_generator_model_buffer = None

        with open(a_person_mask_generator_model_path, "rb") as f:
            a_person_mask_generator_model_buffer = f.read()

        image_segmenter_base_options = BaseOptions(
            model_asset_buffer=a_person_mask_generator_model_buffer
        )
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=image_segmenter_base_options,
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True,
        )

        mask_images: list[Image] = []

        # Create the image segmenter
        with ImageSegmenter.create_from_options(options) as segmenter:
            for tensor_image in images:
                # Convert the Tensor to a PIL image
                i = 255.0 * tensor_image.cpu().numpy()

                # The media pipe library does a much better job with images with an alpha channel for some reason.
                if i.shape[-1] == 3:  # If the image is RGB
                    # Add a fully transparent alpha channel (255)
                    i = np.dstack((i, np.full((i.shape[0], i.shape[1]), 255)))  # Create an RGBA image

                image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                mask_image = self.__get_mask(
                    image=image_pil,
                    segmenter=segmenter,
                    face_mask=face_mask,
                    background_mask=background_mask,
                    hair_mask=hair_mask,
                    body_mask=body_mask,
                    clothes_mask=clothes_mask,
                    eyeglasses_mask=eyeglasses_mask,
                    hat_mask=hat_mask,
                    earrings_mask=earrings_mask,
                    other_accessories_mask=other_accessories_mask,
                    confidence=confidence,
                    refine_mask=refine_mask,
                )
                mask_images.append(mask_image)

        return mask_images

    def generate_mask(
            self,
            images,
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            eyeglasses_mask: bool,
            hat_mask: bool,
            earrings_mask: bool,
            other_accessories_mask: bool,
            confidence: float,
            refine_mask: bool,
    ):
        """Create a segmentation mask from an image

        Args:
            image (torch.Tensor): The image to create the mask for.
            face_mask (bool): create a mask for the face.
            background_mask (bool): create a mask for the background.
            hair_mask (bool): create a mask for the hair.
            body_mask (bool): create a mask for the body.
            clothes_mask (bool): create a mask for the clothes.
            eyeglasses_mask (bool): create a mask for eyeglasses.
            hat_mask (bool): create a mask for hats.
            earrings_mask (bool): create a mask for earrings.
            other_accessories_mask (bool): create a mask for other accessories.
            confidence (float): how confident the model is that the detected item is there.
            refine_mask (bool): refine the mask by cropping and re-processing.

        Returns:
            torch.Tensor: The segmentation masks.
        """

        mask_images = self.get_mask_images(
            images=images,
            face_mask=face_mask,
            background_mask=background_mask,
            hair_mask=hair_mask,
            body_mask=body_mask,
            clothes_mask=clothes_mask,
            eyeglasses_mask=eyeglasses_mask,
            hat_mask=hat_mask,
            earrings_mask=earrings_mask,
            other_accessories_mask=other_accessories_mask,
            confidence=confidence,
            refine_mask=refine_mask,
        )

        tensor_masks = []
        for mask_image in mask_images:
            # convert PIL image to tensor image
            tensor_mask = mask_image.convert("RGB")
            tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
            tensor_mask = torch.from_numpy(tensor_mask)[None,]
            tensor_mask = tensor_mask.squeeze(3)[..., 0]

            tensor_masks.append(tensor_mask)

        return (torch.cat(tensor_masks, dim=0),)
