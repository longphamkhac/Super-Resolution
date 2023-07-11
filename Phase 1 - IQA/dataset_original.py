from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

label_map = {
    "Brightness": 0,
    "Dark": 1,
    "HQ": 2,
    "Motion": 3
}

class IQADataset(Dataset):
    
    def __init__(self, root, status = "Patch_training", transform = None):
        root_status = os.path.join(root, status)
        labels = os.listdir(root_status)
        labels.sort()

        sample = []
        for label in labels:
            listname = os.listdir(os.path.join(root_status, label))
            for name in listname:
                sample.append((os.path.join(root_status, label, name), label_map[label]))
        
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        img_path, label = self.sample[index]
        
        img = Image.open(img_path)
        img = np.array(img)

        if self.transform:
            img = self.transform(image = img)["image"]

        label = np.array(label)
        return img, label, img_path

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from utils import input_path, data_transforms
    status = "Patch_validation"

    train_dataset = IQADataset(input_path, status, transform = data_transforms["train"])

    dataloader = DataLoader(
        train_dataset,
        batch_size = 8,
        shuffle = True
    )

    for i, (inputs, labels, imgs_path) in enumerate(dataloader):
        print(inputs.shape)
        print(labels.shape)

        print(len(imgs_path))
        print(type(imgs_path))

        break
