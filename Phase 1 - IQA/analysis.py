from operator import index
import numpy as np
from PIL import Image
import torch
from evaluate import get_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from utils import dataloaders, input_path, data_transforms
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

num_classes = 4

label_map = {
    0: "Brightness",
    1: "Dark",
    2: "HQ",
    3: "Motion"
}

def plot_img(path):
    img = plt.imread(path)
    plt.imshow(img)
    plt.show()

def get_false_cases(model, device):
    from dataset_original import IQADataset
    from torch.utils.data import DataLoader
    
    map_tracking = {
        "0": [],
        "1": [],
        "2": [],
        "3": []
    } #  key: False Predictions, value: True Img Path

    # Validation Dataset
    val_dataset = IQADataset(input_path, status = "Patch_validation", transform = data_transforms["val"])
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False, num_workers = 8)

    # Train Dataset
    # train_dataset = IQADataset(input_path, status = "Patch_training", transform = data_transforms["val"])
    # train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = False, num_workers = 8)

    for inputs, labels, inputs_path in tqdm(val_loader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        
        _, preds = torch.max(outputs, dim = 1)

        preds, labels = torch.Tensor.tolist(preds.cpu().detach()), torch.Tensor.tolist(labels.data)
        preds = np.array(preds)
        labels = np.array(labels)

        ids = np.where(preds != labels)[0]
        
        false_preds = preds[ids] # False Predictions [2 2 2 2 1 1 1]

        true_img_path = [inputs_path[idx] for idx in ids]
        true_label = [] # ['Brightness || Im_good4343.png', 'Brightness || Im_good4350.png', ...]
        for img_path in true_img_path:
            splits = img_path.split("\\")
            true_label.append(splits[-2] + " || " + splits[-1])

        for i in range(len(false_preds)):
            map_tracking[str(false_preds[i])].append(true_label[i])

    with open("false_predictions_val.txt", "w") as f:
        for key in map_tracking.keys():
            f.write(key + "\n")
            for img_path in map_tracking[key]:
                f.write(img_path + "\n")
                

def get_results(model, device):
    val_loader = dataloaders["val"]

    y_pred = []
    y_true = []
    
    y_score = []
    max_probs = []

    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        out_proba = F.softmax(outputs, dim = 1) # (BATCH_SIZE, NUM_CLASSES) (torch.Tensor)

        max_prob, preds = torch.max(outputs, dim = 1)
        # preds_proba, _ = torch.max(out_proba, dim = 1) => pred_proba is the same shape and type with preds

        length = preds.shape[0]
        preds_list, labels_list, max_problist = torch.Tensor.tolist(preds.cpu().detach()), torch.Tensor.tolist(labels.data), torch.Tensor.tolist(max_prob.cpu().detach())

        y_score.append(out_proba.cpu().detach())
        for i in range(0, length):
            y_pred.append(preds_list[i])
            y_true.append(labels_list[i])
            max_probs.append(max_problist[i])

    y_score = torch.cat(y_score, dim = 0)

    return np.array(y_pred), np.array(y_true), np.array(y_score), np.array(max_probs)


def get_results_multi_head(model, device):
    motion_threshold = 0.7
    val_loader = dataloaders["val"]

    y_pred = []
    y_true = []

    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)

        outputs = model(inputs)

        out1 = outputs[:, :3]
        out2 = outputs[:, 3:]

        out1 = F.softmax(out1)
        out2 = F.sigmoid(out2)

        outputs = torch.cat([out1, out2], dim = 1)

        preds = []
        for output in outputs:
            if output[-1] > motion_threshold:
                preds.append(3)
            else:
                _, index = torch.max(output[:3], dim = 0)
                preds.append(index.item())

        preds = torch.from_numpy(np.array(preds))

        preds_list, labels_list = torch.Tensor.tolist(preds.cpu().detach()), torch.Tensor.tolist(labels.data)
        
        length = preds.shape[0]
        for i in range(0, length):
            y_pred.append(preds_list[i])
            y_true.append(labels_list[i])

    return np.array(y_pred), np.array(y_true)


def infer_test(model, device, path):
    val_transform = data_transforms["val"]

    img = Image.open(path)
    img = np.array(img)
    
    # splits = path.split("/")
    # label = splits[-2] + " || " + splits[-1]
    
    img = val_transform(image = img)["image"]
    input = torch.unsqueeze(img, dim = 0).to(device)

    output = model(input) # torch.Size([1, 4])
    output_proba = F.softmax(output, dim = 1).data

    # print(label + " => " + str(output_proba))
    print(str(output_proba))
    return output_proba

def infer_test_multi_head(model, device, path):
    val_transform = data_transforms["val"]

    img = Image.open(path)
    img = np.array(img)

    img = val_transform(image = img)["image"]
    input = torch.unsqueeze(img, dim = 0).to(device)

    output = model(input)

    out1 = output[:, :3]
    out2 = output[:, 3:]

    out1 = F.softmax(out1)
    out2 = F.sigmoid(out2)

    splits = path.split("\\")
    label = splits[-2] + " || " + splits[-1]

    output = torch.cat([out1, out2], dim = 1).squeeze(-1)
    output = output.data

    print(label + " => " + str(output))

    return output

def plot_cfm(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)

    from mlxtend.plotting import plot_confusion_matrix

    fig, ax = plot_confusion_matrix(conf_mat = conf_matrix, figsize = (6, 6))
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

def plot_PR_curve(y_true, y_score):
    from sklearn.preprocessing import label_binarize
    from yellowbrick.classifier import PrecisionRecallCurve

    y_true = label_binarize(y_true, classes = [0, 1, 2]) # (1064, 4)

    precision_dict = dict()
    recall_dict = dict()

    for i in range(num_classes):
        
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
    
    for i in range(num_classes):
        plt.plot(recall_dict[i], precision_dict[i], label = "{}".format(label_map[i]))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc = "best")
    plt.title("Precision-Recall Curve")
    plt.show()


def logging_images(root_path, files_path, predict_label, loss = "_Focal"):
    save_path = os.path.join("analysis_HQ", predict_label + loss)

    with open(files_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            label, img_name = line.split("||")
            label, img_name = label.strip(), img_name.strip()

            full_path = os.path.join(root_path, label, img_name)
            img = Image.open(full_path)

            img.save(os.path.join(save_path, img_name))

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = r"swin\net-epoch-3-0.95.pth.tar"

    model = get_model(type_model = "swin", path_trained = weight_path)
    model.eval()

    # from evaluate import evaluate_dataset
    # evaluate_dataset(type_model = "resnet18", path_model = weight_path)

    # input_path = r"D:\IQA_Data\Patch_validation\HQ\Im_good4345.png"
    
    # out_proba = infer_test(model, device, input_path)

    # out_proba = infer_test_multi_head(model, device, input_path)

    # print(out_proba)

    # y_pred, y_true, y_score, _ = get_results(model, device)

    ### Multi-head ###
    y_pred, y_true = get_results_multi_head(model, device)

    # print(y_pred.shape) # (1064, )
    # print(y_true.shape) # (1064, )
    # print(y_score.shape) # (1064, 4)

    acc = accuracy_score(y_true, y_pred)
    print("Acc:", acc)
    print(classification_report(y_true, y_pred))

    plot_cfm(y_true, y_pred) # CFM

    # plot_PR_curve(y_true, y_score) #PR Curve

    # get_false_cases(model, device)

    # root_path = r"D:\IQA_Data\Patch_validation\HQ"
    
    # checking_path = "checking.txt"
    # logging_path = "logging_results.txt"

    #### Logging prob of false cases ####
    # logging_path = "log_false_case.txt"
    # with open(logging_path, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         label, name = line.split(" || ")
    #         label, name = label.strip(), name.strip()

    #         input_path = os.path.join(root_path, name)

    #         out_proba = infer_test_multi_head(model, device, input_path)
        
