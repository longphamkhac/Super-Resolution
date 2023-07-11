import cv2
import os
import torch.nn as nn
import torch
# import skvideo.io
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from resnet.resnet import resnet18, resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

label_map = {
    0: "Brightness",
    1: "Dark",
    2: "HQ",
    3: "Motion"
}

backbone = 'ResNet18'
sizeSub = 128
mmean = [0.485, 0.456, 0.406]
sstd  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = A.Compose([
        A.Resize(height = 224, width = 224, always_apply = True), # For viT or Swin
        A.Normalize(mmean, sstd, always_apply = True),
        ToTensorV2()
])

PATH_MODEL = r"viT\net-epoch-2-0.928.pth.tar"
save_root = "VIT_TEST_IMAGES_0.6"

def imshow(im):
    plt.imshow(im)
    plt.show()

def get_model(type_model = "resnet18", path_trained = ""):
    if type_model == 'resnet18':
        PATH_PRETRAINED = 'pretrained/resnet18-5c106cde.pth'
        model    = resnet18(pretrained=False, path_pretrained=PATH_PRETRAINED)
        num_ftrs = model.fc.in_features
        print('Number features were extracted by backbone ResNet18: ', num_ftrs)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 4))
        model.to(device)
    
    elif type_model == 'resnet50':
        PATH_PRETRAINED = 'pretrained/resnet50-19c8e357.pth'
        model    = resnet50(pretrained=False, path_pretrained=PATH_PRETRAINED)
        num_ftrs = model.fc.in_features
        print('Number features were extracted by backbone ResNet50: ', num_ftrs)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4))
        model.to(device)

    elif type_model == 'viT':
        from timm import models
        model = models.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=4)
        nfeats= model.head.in_features
        print('Number features were extracted by backbone ViT: ', nfeats)
        model.head = nn.Sequential(
            nn.Linear(nfeats, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 4)
        )
        model.to(device)

    model.load_state_dict(torch.load(path_trained, map_location = device))
    print('[1] Loading model successful from ', path_trained)
    return model


def inferenceOnPatch(model, image):
    motion_threshold = 0.6
    model.eval()

    image = np.array(image)
    input = test_transforms(image = image)["image"].unsqueeze(0)

    input = Variable(input).to(device)
    
    output = model(input)
    
    out1 = output[:, :3]
    out2 = output[:, 3:]

    out1 = F.softmax(out1)
    out2 = F.sigmoid(out2)

    output = torch.cat([out1, out2], dim = 1).squeeze(-1)
    # print(output[0])

    if output[0][-1] > motion_threshold:
        pred = 3
        pred = torch.tensor(pred)
    else:
        _, pred = torch.max(output[0][:3], dim = 0)

    return pred

def show_res(image, results, save_path):
    from matplotlib import patches

    fig = plt.figure()
    ax = fig.gca()
    plt.imshow(image)
    for res in results:
        x1, y1, x2, y2, type = res[0][0], res[0][1], res[0][2], res[0][3], label_map[res[0][4]]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, alpha=0.7, linestyle="dashed",
                              edgecolor='red', facecolor='none')
        ax.add_patch(p)
        ax.text(x1, y1, type, color = "red")

    plt.savefig(save_path)
    # plt.show()

def inferenceOnFrame(original, save_path = ""):
    start = time.time()
    # original = original.crop((56, 24, 696, 550))
    original = original.crop((40, 24, 680, 550))

    width, height = original.size

    Npatch_per_row = height // sizeSub
    Npatch_per_col = width // sizeSub

    coordPatches = [[0, 0]]
    nHQ = 0
    res = []
    draw_res = []

    for i in range(0, Npatch_per_row):
        for j in range(0, Npatch_per_col):
            nr, nc = sizeSub, sizeSub

            br, bc = coordPatches[-1][0], coordPatches[-1][1]
            patch = original.crop((bc, br, (bc + nc), (br + nr)))

            predict = inferenceOnPatch(model, patch)
            pred = predict.item()
            res.append(pred)
            draw_res.append([
                (bc, br, bc + nc, br + nr, pred)
            ])
            if pred == 2:
                nHQ += 1
            
            coordPatches.append([coordPatches[-1][0], coordPatches[-1][1] + sizeSub])

        coordPatches.append([coordPatches[-1][0] + sizeSub, 0])

    show_res(original, draw_res, save_path)

if __name__ == "__main__":
    model = get_model(type_model = "viT", path_trained = PATH_MODEL)

    images_test_path = r"C:\IQA\Test_Images"

    image_names = os.listdir(images_test_path)
    image_names.sort()
    for image_name in image_names:
        img = Image.open(os.path.join(images_test_path, image_name))
        save_path = os.path.join(save_root, image_name)

        inferenceOnFrame(img, save_path = save_path)

    print("Done!!!")

    
    # input_path = "Test_Images\\Image_13175864.png"
    # input_path = r"Test_Images\Image_253177.png"
    # input_path = r"Test_Images\Image_13182103.png"

    # img = Image.open(input_path)
    # inferenceOnFrame(img)