import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from dataset import IQADataset, label_map
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.manual_seed(seed = 42)

BATCH_SIZE    = 16
LEARNING_RATE = 0.0003 # 0.003
MOMENT        = 0.6
l2_lambda = 0
l1_lambda = 0.001


input_path = 'D:\\IQA_Data\\Best_Dataset_Update'


mmean = [0.485, 0.456, 0.406]
sstd  = [0.229, 0.224, 0.225]


data_transforms = {
    "resize": A.Resize(height = 224, width = 224, always_apply = True),

    "train_horizontal_flip": A.HorizontalFlip(always_apply = True),

    "train_vertical_flip": A.VerticalFlip(always_apply = True),

    "train_rotate_1": A.Rotate((90, 90), always_apply = True),

    "train_rotate_2": A.Rotate((180, 180), always_apply = True),
    
    "train_rotate_3": A.Rotate((270, 270), always_apply = True),
    
    "train_increase_light": A.ColorJitter(
        brightness = (1.25, 1.25),
        contrast = 0, saturation = 0, hue = 0, always_apply = True
    ),

    "train_decrease_light": A.ColorJitter(
        brightness = (0.7, 0.7),
        contrast = 0, saturation = 0, hue = 0, always_apply = True
    ),

    "train_add_sunflare": A.RandomSunFlare(
        flare_roi = (0, 0, 0.1, 0.1), 
        angle_lower = 0, angle_upper = 0.2, 
        num_flare_circles_lower = 6, 
        num_flare_circles_upper = 10, 
        src_radius = 124, src_color = (255, 255, 255),
        always_apply = True
    ),

    "train_shearing": A.augmentations.geometric.transforms.Affine(
        scale = 1,
        translate_percent = 0,
        translate_px = None,
        rotate = 90,
        shear = 35,
        keep_ratio = True,
        always_apply = True
    ),

    "train_contrast": A.ColorJitter(
        brightness = 0.5, contrast = 0.5, saturation = 0, hue = 0, always_apply = True
    ),

    "train_optical_distortion": A.OpticalDistortion(
        distort_limit = (0.75, 0.75),
        shift_limit = (0.75, 0.75),
        always_apply = True
    ),

    "train_gaussian_noise": A.IAAAdditiveGaussianNoise(
        loc = 30,
        scale = (2.5500000000000003, 12.75),
        always_apply = True
    ),

    "train_equalize": A.Equalize(
        mode = "cv", by_channels = True, always_apply = True
    ),

    "HQ_transforms": A.Compose([
        A.HorizontalFlip(always_apply = True),
    ]),

    "val": A.Compose([
        A.Normalize(mean = mmean, std = sstd, always_apply = True),
        ToTensorV2()
    ])
}



image_datasets = {
    "train": IQADataset(input_path, status = "Patch_training", transform = data_transforms),
    "val": IQADataset(input_path, status = "Patch_validation", transform = data_transforms)
}

dataloaders = {
    "train": torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size = BATCH_SIZE,
                                shuffle = True,
                                num_workers = 8),

    "val": torch.utils.data.DataLoader(image_datasets['val'],
                                batch_size = BATCH_SIZE,
                                shuffle = False,
                                num_workers = 8)
}
# ====================================================================================


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# class_names = image_datasets['train'].classes
class_names = [label for label in label_map.keys()]

device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

device = torch.device(device_name)


def weight_mse(output, target, weight=torch.Tensor([[0.166, 0.167, 0.5, 0.167]])):
    output = nn.Softmax(dim=1)(output)
    target = target.to(torch.long)
    target2= torch.zeros(output.shape).to(device)
    for t in range(0, len(target)):
        target2[t, target[t]] = 1
    weight= weight.repeat(output.shape[0], 1).to(device)
    return torch.sum(weight * (output - target2)**2)

def showImg(inp, title=None):
    """
        show image from Tensor
    """
    global mmean, sstd
    inp  = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mmean)
    std  = np.array(sstd)
    inp  = std*inp + mean
    inp  = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


if __name__ == "__main__":
    print('Length:', dataset_sizes)
    print('Class names:', class_names)
    print('Device name:', device_name)
    print('Len class names:', len(class_names))


    # from torch.nn.functional import one_hot
    # for inputs, labels in dataloaders['train']:
    #     print(inputs.shape)
    #     print(labels.shape)
    #     print(labels)

    #     labels_bce = one_hot(labels, num_classes = 3)
    #     print(labels_bce.shape)
    #     print(labels_bce)
    #     break


    