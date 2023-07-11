import os
import time

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torchgeometry.losses import FocalLoss

from resnet.resnet import *
# from swinT import build_model, get_config
from utils import *


def plot_progress(train_loss, valid_loss, save_path = ""):
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)

    index_train = np.arange(len(train_loss))
    index_val = np.arange(len(valid_loss))

    num_iter_train = len(index_train)
    num_iter_val = len(index_val)

    plt.figure()
    plt.subplot(121)
    (l1,) = plt.plot(
        index_train[0: num_iter_train], train_loss[0: num_iter_train], label = "Train iter loss"
    )

    plt.legend(handles = [l1])
    plt.gcf().gca().set_xlim(left=0)
    plt.title("Train iter loss")

    plt.subplot(122)
    (l2,) = plt.plot(
        index_val[0: num_iter_val], train_loss[0: num_iter_val], label = "Valid iter loss", color = "orange"
    )

    plt.legend(handles = [l2])
    plt.gcf().gca().set_xlim(left=0)
    plt.title("Val iter loss")

    plt.savefig(os.path.join(save_path, "loss_iter.png"))


def get_model(gamma, type_model="resnet18"):
    if type_model == "resnet18":
        PATH_PRETRAINED = 'pretrained/resnet18-5c106cde.pth'
        model = resnet18(pretrained=True, path_pretrained=PATH_PRETRAINED)
        num_ftrs = model.fc.in_features
        print("Number features were extracted by backbone ResNet18: ", num_ftrs) # 512
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, len(class_names)),
        )
        model.to(device)
    elif type_model == "resnet50":
        PATH_PRETRAINED = "pretrained/resnet50-19c8e357.pth"
        model = resnet50(pretrained=True, path_pretrained=PATH_PRETRAINED) # 2048
        num_ftrs = model.fc.in_features
        print("Number features were extracted by backbone ResNet50: ", num_ftrs) # 2048
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(class_names)),
        )
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

    elif type_model == "viT":
        from timm import models

        model = models.create_model(
            "vit_tiny_patch16_224", pretrained=True, num_classes=4
        )
        nfeats = model.head.in_features
        print("Number features were extracted by backbone ViT: ", nfeats)
        model.head = nn.Sequential(
            nn.Linear(nfeats, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, len(class_names)),
        )
        model.to(device)
    elif type_model == "fpn":
        # from resnet.fpn import ResNetFPN
        from resnet.fpn_custom import ResNetFPN

        model = ResNetFPN(nclass=4, backbone_name="resnet18")
        print(
            "Number features were extracted by backbone ResNet18-FPN: ",
            model.num_fts,
        )
        model.to(device)
    else:
        raise ValueError("Type model is not supported!")

    # criterion = weight_mse  # nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing = 0.8)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = FocalLoss(alpha = 1, gamma = gamma, reduction = "mean") # Focal Loss

    # optimizer_ft = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENT)
    optimizer_ft = optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 0.0005)
    # scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_ft,
        T_0 = 2000,
        T_mult = 1,
        eta_min = 1e-6,
        last_epoch = -1
    )
    return model, criterion, optimizer_ft, scheduler


def train_model(model, criterion, optimizer, scheduler, save_path, num_epochs=25):

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    best_acc = 0.0
    best_loss = 100

    iteration_train_loss = []
    iteration_valid_loss = []
    lr_iteration = []

    t = np.linspace(1, num_epochs, num_epochs)
    epoch_train_loss = 0 * t
    epoch_valid_loss = 0 * t
    epoch_train_acc = 0 * t
    epoch_valid_acc = 0 * t

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        since = time.time()
        for phase in ["train", "val"]:
            # if phase == 'train':
            #    scheduler.step()

            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss, running_correct = 0.0, 0

            # iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.long().to(device)

                # For BCE lOSS
                # labels_bce = one_hot(labels, num_classes = 3)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    # Adding L2 Regularization
                    # l1_norm = sum(p.abs().sum() for p in model.parameters())
                    l2_norm = sum(p.pow(2).sum() for p in model.parameters())

                    if phase == "train":
                        loss = criterion(outputs.unsqueeze(-1).unsqueeze(-1), labels.unsqueeze(-1).unsqueeze(-1)) + l2_lambda * l2_norm
                    else:
                        loss = criterion(outputs.unsqueeze(-1).unsqueeze(-1), labels.unsqueeze(-1).unsqueeze(-1))

                    # loss = criterion(outputs.unsqueeze(-1).unsqueeze(-1), labels.unsqueeze(-1).unsqueeze(-1)) + l2_lambda * l2_norm
                    

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)

                # Iteration Loss
                if phase == "train":
                    iteration_train_loss.append(loss.item())
                    # lr_iteration.append(optimizer.param_groups[0]["lr"])
                elif phase == "val":
                    iteration_valid_loss.append(loss.item())

                # print(lr_iteration)

            # if phase == 'train':
            #    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct.double() / dataset_sizes[phase]
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "train":
                epoch_train_acc[epoch] = epoch_acc
                epoch_train_loss[epoch] = epoch_loss
            else:
                epoch_valid_acc[epoch] = epoch_acc
                epoch_valid_loss[epoch] = epoch_loss

            if phase == "val" and (epoch_acc > best_acc):
                best_acc = epoch_acc
                best_loss = epoch_loss
                save_str = os.path.join(
                    save_path,
                    "net-epoch-"
                    + str(epoch + 1)
                    + "-"
                    + str(round(best_acc.item(), 3))
                    + ".pth.tar",
                )
                torch.save(model.state_dict(), save_str)
                print("Model saved!")

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.1f}m {:.1f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val accuracy: {:4f} and loss: {:4f}".format(best_acc, best_loss))

        # Plot Iteration Progress
        plot_progress(iteration_train_loss, iteration_valid_loss, save_path = save_path)

        if (epoch + 1) % 5 == 0:
            plt.figure()
            plt.subplot(121)
            (l1,) = plt.plot(
                t[: epoch + 1], epoch_train_loss[: epoch + 1], label="Train loss"
            )
            (l2,) = plt.plot(
                t[: epoch + 1], epoch_valid_loss[: epoch + 1], label="Valid loss"
            )
            plt.legend(handles=[l1, l2])
            plt.gcf().gca().set_xlim(left=0)
            plt.title("Loss")

            plt.subplot(122)
            (l1,) = plt.plot(
                t[: epoch + 1], epoch_train_acc[: epoch + 1], label="Train accuracy"
            )
            (l2,) = plt.plot(
                t[: epoch + 1], epoch_valid_acc[: epoch + 1], label="Valid accuracy"
            )
            plt.legend(handles=[l1, l2])
            plt.gcf().gca().set_ylim(0, 1)
            plt.title("Accuracy")

            plt.savefig(os.path.join(save_path, "train_epoch.png"))

    return model


def train_multi_head(model, criterion1, optimizer, scheduler, save_path, num_epochs=25):
    criterion2 = nn.BCELoss()
    motion_threshold = 0.5 # Modify
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    best_acc = 0.0
    best_loss = 100

    iteration_train_loss = []
    iteration_valid_loss = []

    t = np.linspace(1, num_epochs, num_epochs)
    epoch_train_loss = 0 * t
    epoch_valid_loss = 0 * t
    epoch_train_acc = 0 * t
    epoch_valid_acc = 0 * t

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        since = time.time()
        for phase in ["train", "val"]:
            # if phase == 'train':
            #    scheduler.step()

            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss, running_correct = 0.0, 0

            # iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.long().to(device)

                # For BCE lOSS
                # labels_bce = one_hot(labels, num_classes = 3)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    ### Modify ###
                    out1 = outputs[:, :3]
                    out2 = outputs[:, 3:]

                    out2 = F.sigmoid(out2)

                    outputs = torch.cat([out1, out2], dim = 1)

                    # Calculate loss for Brightness, Dark, and HQ
                    index1 = np.where(labels.cpu() != 3)[0]
                    
                    out1 = out1[index1]
                    labels1 = labels[index1]

                    loss1 = criterion1(
                        out1.unsqueeze(-1).unsqueeze(-1), 
                        labels1.unsqueeze(-1).unsqueeze(-1)
                    )


                    # Calculate loss for Motion
                    index2 = np.where(labels.cpu() == 3)[0]

                    labels_bce = labels.clone().detach()
                    labels_bce[index1] = 0
                    labels_bce[index2] = 1

                    loss2 = criterion2(out2.squeeze(1).float(), labels_bce.float())

                    if len(index1) == 0:
                        loss1 = 0
                    if len(index2) == 0:
                        loss2 = 0

                    loss = loss1 + loss2

                    # Adding L2 Regularization
                    # l1_norm = sum(p.abs().sum() for p in model.parameters())
                    l2_norm = sum(p.pow(2).sum() for p in model.parameters())

                    if phase == "train":
                        loss = loss + l2_lambda * l2_norm
                    else:
                        loss = loss              
                    
                    # Predictions
                    preds = []
                    for output in outputs:
                        if output[-1] > motion_threshold:
                            preds.append(3)
                        else:
                            _, index = torch.max(output[:3], dim = 0)
                            preds.append(index.item())

                    preds = torch.from_numpy(np.array(preds))

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds.cpu().detach() == labels.cpu().data)

                # Iteration Loss
                if phase == "train":
                    iteration_train_loss.append(loss.item())
                    # lr_iteration.append(optimizer.param_groups[0]["lr"])
                elif phase == "val":
                    iteration_valid_loss.append(loss.item())

                # print(lr_iteration)

            # if phase == 'train':
            #    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct.double() / dataset_sizes[phase]
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "train":
                epoch_train_acc[epoch] = epoch_acc
                epoch_train_loss[epoch] = epoch_loss
            else:
                epoch_valid_acc[epoch] = epoch_acc
                epoch_valid_loss[epoch] = epoch_loss

            if phase == "val" and (epoch_acc > best_acc):
                best_acc = epoch_acc
                best_loss = epoch_loss
                save_str = os.path.join(
                    save_path,
                    "net-epoch-"
                    + str(epoch + 1)
                    + "-"
                    + str(round(best_acc.item(), 3))
                    + ".pth.tar",
                )
                torch.save(model.state_dict(), save_str)
                print("Model saved!")

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.1f}m {:.1f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val accuracy: {:4f} and loss: {:4f}".format(best_acc, best_loss))

        # Plot Iteration Progress
        plot_progress(iteration_train_loss, iteration_valid_loss, save_path = save_path)

        if (epoch + 1) % 5 == 0:
            plt.figure()
            plt.subplot(121)
            (l1,) = plt.plot(
                t[: epoch + 1], epoch_train_loss[: epoch + 1], label="Train loss"
            )
            (l2,) = plt.plot(
                t[: epoch + 1], epoch_valid_loss[: epoch + 1], label="Valid loss"
            )
            plt.legend(handles=[l1, l2])
            plt.gcf().gca().set_xlim(left=0)
            plt.title("Loss")

            plt.subplot(122)
            (l1,) = plt.plot(
                t[: epoch + 1], epoch_train_acc[: epoch + 1], label="Train accuracy"
            )
            (l2,) = plt.plot(
                t[: epoch + 1], epoch_valid_acc[: epoch + 1], label="Valid accuracy"
            )
            plt.legend(handles=[l1, l2])
            plt.gcf().gca().set_ylim(0, 1)
            plt.title("Accuracy")

            plt.savefig(os.path.join(save_path, "train_epoch.png"))

    return model


if __name__ == "__main__":
    
    type_model = "swin"
    model, criterion, optimizer, scheduler = get_model(gamma = 2, type_model = type_model)


    # weight = {}
    # for key, param in state_dict.items():
    #     if key in ["fc.0.weight", "fc.0.bias"]:
    #         continue
    #     weight[key] = param

    # torch.save(weight, "model_weights.pth")

    ##### Resnet50 ####
    # x = torch.rand(2, 3, 64, 64) # torch.Size([2, 2048, 2, 2])
    # x = torch.rand(2, 3, 128, 128) # torch.Size([2, 2048, 4, 4])
    # x = torch.rand(2, 3, 32, 32) # torch.Size([2, 2048, 1, 1])

    ##### Resnet18 ####
    # x = torch.rand(2, 3, 64, 64) # torch.Size([2, 512, 2, 2])
    # x = torch.rand(2, 3, 128, 128) # torch.Size([2, 512, 4, 4])
    # x = torch.rand(2, 3, 32, 32) # torch.Size([2, 512, 1, 1])


    # gamma_tuning = [2, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8]
    model = train_multi_head(model, criterion, optimizer, scheduler, save_path=type_model, num_epochs=100)
