from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import warnings
warnings.filterwarnings("ignore")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# Binary dataset
label_map = {
    "Brightness": 0,
    "Dark": 1,
    "HQ": 2, # Modify
    "Motion": 3 # Modify
}

map_tracking = {}
# map_cheking = {}


class IQADataset(Dataset):

    def __init__(self, root, status = "Patch_training", transform = None):
        root_status = os.path.join(root, status)
        labels = os.listdir(root_status)
        labels.sort()

        sample = []
        for label in labels:
            listname = os.listdir(os.path.join(root_status, label))
            for name in listname:
                full_path = os.path.join(root_status, label, name)

                if status == "Patch_training":
                    map_tracking[full_path] = 0

                    ### For Brightness label ###
                    if label_map[label] == 0: 
                        for _ in range(12):
                            sample.append((full_path, label_map[label]))
                    
                    ### For Dark label ###
                    elif label_map[label] == 1:
                        for _ in range(6):
                            sample.append((full_path, label_map[label]))

                    ### For HQ ###
                    elif label_map[label] == 2: # Modify
                        for _ in range(2):
                            sample.append((full_path, label_map[label]))

                    ### For Motion ###
                    elif label_map[label] == 3: # Modify
                        for _ in range(2):
                            sample.append((full_path, label_map[label]))
                
                else:
                    sample.append((full_path, label_map[label]))

        # for i in sample:
        #     map_cheking[i[0]] = []

        self.sample = sample
        self.status = status
        self.transform = transform

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        img_path, label = self.sample[index]
        
        img = Image.open(img_path)
        img = np.array(img)

        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)

        # Resize image to 224x224 size for ViT and Swin
        img = self.transform["resize"](image = img)["image"]

        if label not in [0, 1, 2, 3]:
            img = self.transform["val"](image = img)["image"]
            return img, np.array(label)

        if self.status == "Patch_training":
            if map_tracking[img_path] == 0: # Original Image
                if self.transform:
                    img = self.transform["val"](image = img)["image"]
                    # pass # For visualize

            elif map_tracking[img_path] == 1: # Horizontal Flip
                if self.transform:
                    if label in [0, 1, 3]: # For Brightness and Dark
                        img = self.transform["train_horizontal_flip"](image = img)["image"]
                    elif label == 2: # For HQ (# Modify)
                        img = self.transform["HQ_transforms"](image = img)["image"]

            elif map_tracking[img_path] == 2: # Vertical Flip
                if self.transform:
                    img = self.transform["train_vertical_flip"](image = img)["image"]

            elif map_tracking[img_path] == 3: # Rotate 90
                if self.transform:
                    img = self.transform["train_rotate_1"](image = img)["image"]

            elif map_tracking[img_path] == 4: # Rotate 180
                if self.transform:
                    img = self.transform["train_rotate_2"](image = img)["image"]
            
            elif map_tracking[img_path] == 5: # Rotate 270
                if self.transform:
                    img = self.transform["train_rotate_3"](image = img)["image"]

            elif map_tracking[img_path] == 6: # Increase or Decrease the brightness
                if self.transform:
                    if label == 0: # If brighness
                        img = self.transform["train_increase_light"](image = img)["image"]
                    else: # If dark
                        img = self.transform["train_decrease_light"](img)

            elif map_tracking[img_path] == 7: # Add sunflare
                if self.transform:
                    if label == 0:
                        img = self.transform["train_add_sunflare"](image = img)["image"]
                    else: # If dark
                        img = self.transform["train_shearing"](image = img)["image"]

            elif map_tracking[img_path] == 8: # Shearing (For Brightness and HQ)
                if self.transform:
                    if label == 0:
                        img = self.transform["train_shearing"](image = img)["image"]
                    else: # If dark
                        pass

            elif map_tracking[img_path] == 9:
                if self.transform:
                    if label == 0:
                        img = self.transform["train_gaussian_noise"](image = img)["image"]
                    else: # If dark
                        pass

            elif map_tracking[img_path] == 10:
                if self.transform:
                    if label == 0:
                        img = self.transform["train_optical_distortion"](image = img)["image"]
                    else: # If dark
                        pass

            elif map_tracking[img_path] == 11:
                if self.transform:
                    if label == 0:
                        img = self.transform["train_contrast"](image = img)["image"]
                    else: # If dark
                        pass
            
            # elif map_tracking[img_path] == 12:
            #     if self.transform:
            #         if label == 0:
            #             img = self.transform["train_equalize"](image = img)["image"]
            #         else:
            #             pass

            ### Command if map_checking ###
            if map_tracking[img_path] != 0:
                img = self.transform["val"](image = img)["image"]
                
            map_tracking[img_path] += 1
        
        else:
            if self.transform:
                img = self.transform["val"](image = img)["image"]

        # map_cheking[img_path].append(img)

        label = np.array(label)
        return img, label

if __name__ == "__main__":

    from torch.utils.data import DataLoader

    from utils import input_path, data_transforms

    status = "Patch_training"

    train_dataset = IQADataset(input_path, status, transform = data_transforms)


    # num_img = train_dataset.__len__()
    # for index in range(0, num_img):
    #     train_dataset.__getitem__(index)


    # count = 0
    # for img_path, list_img in map_cheking.items():
    #     print(img_path)
        
    #     for img in list_img:
    #         plt.imshow(img)
    #         plt.show()

    #     count += 1
    #     if count == 1:
    #         break


    print(train_dataset.__len__())
    # train_dataset.__getitem__(0)
    # for key, value in map_tracking.items():
    #     print(key, value)
    #     break

    dataloader = DataLoader(
        train_dataset,
        batch_size = 8,
        shuffle = True
    )

    for i, (input, label) in enumerate(dataloader):
        print(input.shape) # torch.Size([8, 3, 128, 128])
        print(label.shape) # torch.Size([8])
        print(label)

        break

