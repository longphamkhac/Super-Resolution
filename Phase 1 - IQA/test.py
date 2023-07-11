from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import random
import cv2

import warnings
warnings.filterwarnings("ignore")

# Add random sunflare
transform = A.Compose([
    A.RandomSunFlare(flare_roi=(0, 0, 0.05, 0.05), 
                              angle_lower=0, angle_upper=0.2, 
                              num_flare_circles_lower=6, 
                              num_flare_circles_upper=10, 
                              src_radius=124, src_color=(255, 255, 255),
                              always_apply=True)
])

# IAA Affine (Shearing)
transform1 = A.Compose([
    A.augmentations.geometric.transforms.Affine(
        scale = 1,
        translate_percent = 0,
        translate_px = None,
        rotate = 90,
        shear = 35,
        keep_ratio = True,
        always_apply = True
    )
])

# ColorJitter (Contrast)
transform2 = A.Compose([
    A.ColorJitter(brightness = 0, contrast = 0.3, saturation = 0, hue = 0, always_apply = True)
])


# Optical Distortion
transform3 = A.Compose([
    A.OpticalDistortion(
        distort_limit = (0.75, 0.75),
        shift_limit = (0.75, 0.75),
        always_apply = True
    )
])


# IAAA Addictive Gaussian Noise
transform4 = A.Compose([
    A.IAAAdditiveGaussianNoise(
        loc = 30,
        scale = (2.5500000000000003, 12.75),
        always_apply = True
    )
])

# Rotate 90
transform5 = A.augmentations.geometric.rotate.Rotate(limit = (90, 90), always_apply = True)

# Rotate 180
transform6 = A.Rotate(limit = (180, 180), always_apply = True)

# Brightness
transform7 = A.ColorJitter(
    brightness = (1.25, 1.25),
    contrast = 0, saturation = 0, hue = 0, always_apply = True
)

# Equalize
transform8 = A.Equalize(
    mode = "cv", by_channels = True, always_apply = True
)

# CLAHE
def clahe(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit = 20.0, tileGridSize = (30, 30))
    v = clahe.apply(v)

    hsv_img = np.dstack((h, s, v))

    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    return rgb

# IAA Sharpen for HQ
transform10 = A.IAASharpen(
    alpha = (0.5, 0.5),
    lightness = (0.5, 0.5),
    always_apply = True
)

# ISO Noise
transform11 = A.ISONoise(
    color_shift = (0.05, 0.05),
    intensity = (0.5, 0.5),
    always_apply = True
)


##### Augmentation for HQ label #####
HQ_tranforms = A.Compose([
    A.HorizontalFlip(always_apply = True),
    A.VerticalFlip(always_apply = True),
    A.Rotate((270, 270), always_apply = True),
    A.OpticalDistortion(
        distort_limit = (0.75, 0.75),
        shift_limit = (0.75, 0.75),
        always_apply = True
    ),
    A.augmentations.geometric.transforms.Affine(
        scale = 1,
        translate_percent = 0,
        translate_px = None,
        rotate = 90,
        shear = 35,
        keep_ratio = True,
        always_apply = True
    ),
])


##### Augmentation for Dark label #####
Dark_tranforms = A.Compose([
    A.HorizontalFlip(always_apply = True),
    A.ColorJitter(
        brightness = (0.75, 0.75),
        contrast = 0, saturation = 0, hue = 0, always_apply = True
    ),
    A.OpticalDistortion(
        distort_limit = (0.75, 0.75),
        shift_limit = (0.75, 0.75),
        always_apply = True
    ),
    A.augmentations.geometric.transforms.Affine(
            scale = 1,
            translate_percent = 0,
            translate_px = None,
            rotate = 90,
            shear = 35,
            keep_ratio = True,
            always_apply = True
    )
])


data_transform = {
    "train_bright_dark_1": A.Compose([
        A.HorizontalFlip(always_apply = True),
        A.Rotate((90, 90), always_apply = True),
    ]),

    "train_bright_dark_2": A.Compose([
        A.VerticalFlip(always_apply = True),
        A.Rotate((180, 180), always_apply = True)
    ]),

    "train_brightness_3": A.ColorJitter(
        brightness = (1.25, 1.25),
        contrast = 0, saturation = 0, hue = 0, always_apply = True
    ),

    "train_dark_3": A.ColorJitter(
        brightness = (0.75, 0.75),
        contrast = 0, saturation = 0, hue = 0, always_apply = True
    ),

    "train_bright_4": A.RandomSunFlare(
        flare_roi = (0, 0, 0.05, 0.05), 
        angle_lower = 0, angle_upper = 0.2, 
        num_flare_circles_lower = 6, 
        num_flare_circles_upper = 10, 
        src_radius = 124, src_color = (255, 255, 255),
        always_apply = True
    ),

    "train_dark_4": A.VerticalFlip(always_apply = True),

    "train_bright_dark_5": A.augmentations.geometric.transforms.Affine(
        scale = 1,
        translate_percent = 0,
        translate_px = None,
        rotate = 90,
        shear = 35,
        keep_ratio = True,
        always_apply = True
    ),

    "train_bright_6": A.OpticalDistortion(
        distort_limit = (0.75, 0.75),
        shift_limit = (0.75, 0.75),
        always_apply = True
    ),

    "train_bright_7": A.IAAAdditiveGaussianNoise(
        loc = 30,
        scale = (2.5500000000000003, 12.75),
        always_apply = True
    ),

    "train_HQ": A.HorizontalFlip(always_apply = True),
}



if __name__ == "__main__":
    import torch

    # input_path = "/home/long/IQA/data/train_dataset/Analysis_dataset/Patch_training/Brightness/Im_good3203.png"
    input_path = "/home/long/IQA/data/train_dataset/Analysis_dataset/Patch_training/HQ/Im_good3448.png"
    # input_path = "/home/long/IQA/data/train_dataset/Analysis_dataset/Patch_training/Dark/Im_good3190.png"

    img = Image.open(input_path)
    img = np.array(img)

    plt.imshow(img)
    plt.show()

    # CLAHE
    # img_transformed = clahe(img)

    # augs = Dark_tranforms(image = img)
    # img_transformed = augs["image"]

    # plt.imshow(img_transformed)
    # plt.show()
