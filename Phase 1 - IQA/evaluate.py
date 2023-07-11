import os 
import torch 
import cv2 
import torch.nn as nn

from torchvision import transforms
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix
from PIL import Image

from utils import *
from resnet.resnet import *
# from swinT import build_model, get_config

def get_model(type_model='resnet18', path_trained=''):
    if type_model == 'resnet18':
        PATH_PRETRAINED = 'pretrained/resnet18-5c106cde.pth'
        model    = resnet18(pretrained=False, path_pretrained=PATH_PRETRAINED)
        num_ftrs = model.fc.in_features
        print('Number features were extracted by backbone ResNet18: ', num_ftrs)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, len(class_names)))
        model.to(device)
    elif type_model == 'resnet50':
        PATH_PRETRAINED = 'pretrained/resnet50-19c8e357.pth'
        model    = resnet50(pretrained=False, path_pretrained=PATH_PRETRAINED)
        num_ftrs = model.fc.in_features
        print('Number features were extracted by backbone ResNet50: ', num_ftrs)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(class_names)))
        model.to(device)
    elif type_model == "swin":
        from timm import models

        model = models.create_model(
            model_name = "swin_tiny_patch4_window7_224",
            pretrained = True,
            num_classes = 4
        )
        nfeats = model.head.in_features
        print("Number features were extracted by backbone Swin: ", nfeats)
        model.head = nn.Sequential(
            nn.Linear(nfeats, 96),
            nn.ReLU(inplace = True),
            nn.Linear(96, len(class_names))
        )
        model.to(device)
    elif type_model == 'viT':
        from timm import models
        model = models.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=4)
        nfeats= model.head.in_features
        print('Number features were extracted by backbone ViT: ', nfeats)
        model.head = nn.Sequential(
            nn.Linear(nfeats, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, len(class_names))
        )
        model.to(device)

    model.load_state_dict(torch.load(path_trained, map_location = device))
    print('[1] Loading model successful from ', path_trained)
    return model

def print_cfm(matrix, classes):
    print('Confusion matrix:')
    print(matrix)
    tp = 0
    for i in range(0, len(classes)):
        r = matrix[i, i] / np.sum(matrix[i, :])
        p = matrix[i, i] / np.sum(matrix[:, i])
        f = 2*p*r/(p + r + 0.0000001)
        print(f'Precision: {p:4f}. Recall: {r:4f}. Fscore: {f:4f}. Class: {classes[i]}')
        tp += matrix[i, i]
    print('Accuracy: ', tp/np.sum(matrix))

def evaluate_dataset(type_model, path_model):
    val_loader = dataloaders["val"]
    print('Number test iteration: ', len(val_loader), '. Batch size: ', BATCH_SIZE)

    model = get_model(type_model=type_model, path_trained=path_model)
    model.eval()
    matrix= np.zeros((len(class_names), len(class_names)))
    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)
        outputs= model(inputs)
        _,preds= torch.max(outputs, 1)

        cfm    = confusion_matrix(labels.numpy(), preds.cpu().numpy(), labels=range(0, len(class_names)))
        matrix+= cfm
    print_cfm(matrix, class_names)

def evaluate_patch(type_model, path_model, src_pt):
    names = class_names
    model = get_model(type_model=type_model, path_trained=path_model)
    trasf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mmean, sstd)
    ])
    if type_model == 'swin' or type_model == 'viT':
        trasf = transforms.Compose([
            transforms.Pad(padding=48), # for swinT. Input size 224x224
            transforms.ToTensor(),
            transforms.Normalize(mmean, sstd)
        ])
    categs= os.listdir(src_pt)
    files = []
    for cat in categs:
        fs = os.listdir(os.path.join(src_pt, cat))
        for f in fs:
            files.append([os.path.join(src_pt, cat, f), cat])
    print('Number test: ', len(files))
    predicts, groundtruthes = [], []
    model.eval()
    for file in tqdm(files):
        img = Image.open(file[0])
        img = trasf(img).float().unsqueeze_(0).to(device)
        out = nn.Softmax(dim=1)(model(img))
        v,c = torch.max(out, 1)
        c   = c.cpu().numpy()[0]
        v   = v.detach().cpu().numpy()[0]
        predicts.append(class_names[c])
        groundtruthes.append(file[1])
    
    matrix = confusion_matrix(groundtruthes, predicts, labels=names)
    print_cfm(matrix, names)

def evaluate_frame(type_model, path_model, data_pt):
    names = class_names
    model = get_model(type_model=type_model, path_trained=path_model)
    trasf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mmean, sstd)
    ])
    categs= os.listdir(data_pt)
    files = []
    for cat in categs:
        fs = os.listdir(os.path.join(data_pt, cat))
        for f in fs:
            files.append([os.path.join(data_pt, cat, f), cat])
    print('Number test: ', len(files))

    image = Image.open(files[0][0])
    fw, fh= image.size
    psize = 112
    lstPts= []
    sy    = 0
    while (sy + psize) < fh:
        sx = 0
        while (sx + psize) < fw:
            lstPts.append([sx, sy])
            sx += (psize // 2)
        sy += (psize // 2)       

    predicts, groundtruthes = [], []
    model.eval()
    for file in tqdm(files):
        img = Image.open(file[0])
        img = trasf(img)
        img = torch.stack([img[:, lstPts[i][1]: lstPts[i][1]+psize, lstPts[i][0]:lstPts[i][0]+psize] \
            for i in range(0, len(lstPts))])
        img = img.to(device)
        
        out = model(img)
        _,pd= torch.max(out, 1)
        nLq = torch.sum(pd)
        
        if (nLq.cpu().numpy() / img.shape[0]) > 0.1:
            predicts.append('LQ')
        else:
            predicts.append('HQ')
        groundtruthes.append(file[1])
    
    matrix = confusion_matrix(groundtruthes, predicts, labels=names)
    print_cfm(matrix, names)

if __name__ == "__main__":
    evaluate_dataset(type_model='viT', path_model="viT/net-epoch-12-0.799.pth.tar")
    # evaluate_dataset(type_model='resnet18', path_model="resnet18_4ce/net-epoch-28.pth.tar")
    print('\n[3] Starting on new validated dataset\n')
    evaluate_patch(type_model='viT', path_model='viT/net-epoch-12-0.799.pth.tar', 
        src_pt='evalds/Patch/')
    # evaluate_patch(type_model='resnet18', path_model='resnet18_4ce/net-epoch-28.pth.tar', src_pt='evalds/Patch/')

