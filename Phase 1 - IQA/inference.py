import cv2 
import torch.nn as nn
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from resnet.resnet import resnet18, resnet50
from glob import glob
# from resnetModel import *

backbone = 'ResNet18'
sizeSub = 112
mmean = [0.485, 0.456, 0.406]
sstd  = [0.229, 0.224, 0.225]
classes = ['HQ', 'LQ']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mmean, sstd)
])

if backbone == 'ResNet50':
    PATH_MODEL = 'ModelResNet50/net-best_epoch-39.pth.tar'
else:
    PATH_MODEL = 'ModelResNet18/net-best_epoch-9.pth.tar'

def imshow(im):
    plt.imshow(im)
    plt.show()

def getModelResNet50():
    PATH_PRETRAINED = 'Pretrained/resnet50-19c8e357.pth'

    model_ft = resnet50(pretrained=True, path_pretrained=PATH_PRETRAINED)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2))
    model_ft = model_ft.to(device)

    model_ft.load_state_dict(torch.load(PATH_MODEL, map_location=device))
    print('Model is loaded! Backbone: ResNet50')
    return model_ft

def getModelResNet18_o():
    PATH_PRETRAINED = 'Pretrained/resnet18-5c106cde.pth'
    model_ft = resnet18(pretrained=True, path_pretrained=PATH_PRETRAINED)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 96),
        nn.ReLU(inplace=True),
        nn.Linear(96, 2))
    model_ft = model_ft.to(device)

    model_ft.load_state_dict(torch.load(PATH_MODEL, map_location=device))
    print('Model is loaded! Backbone ResNet18')
    
    return model_ft

def getModelResNet18(path_trained = 'saved_weights/resnet18_4wmse_tiny_lrm_moredata/net-epoch-19-0.816.pth.tar'):
    from resnet.resnet import resnet18
    PATH_PRETRAINED = 'saved_weights/resnet18_4wmse_tiny_lrm_moredata/net-epoch-19-0.816.pth.tar'
    model    = resnet18(pretrained=False, path_pretrained=PATH_PRETRAINED)
    num_ftrs = model.fc.in_features
    print('Number features were extracted by backbone ResNet18: ', num_ftrs)
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 96),
        nn.ReLU(inplace=True),
        nn.Linear(96, 4))
    model.to(device)

    model.load_state_dict(torch.load(path_trained, map_location=device))
    print('[1] Loading model successful from ', path_trained)
    return model

def inferenceOnPatch(model, image):
    model.eval()
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    input = Variable(image_tensor).to(device)
    output = model(input)
    _, pred = torch.max(output, 1)

    return pred

def threshold_res(prob_hq):
    types = [
        'Bad',
        'Poor',
        'Fair',
        'Good',
        'Exellent'
    ]
    if prob_hq < 3:
        return types[0]
    elif prob_hq < 10:
        return types[1]
    elif prob_hq < 30:
        return types[2]
    elif prob_hq < 50:
        return types[3]
    else:
        return types[4]

def get_most_likely_type(most_likely_type):
    if most_likely_type == 0:
        return 'Over Exposed'
    if most_likely_type == 1:
        return 'Under Exposed'
    if most_likely_type == 2:
        return 'Good'
    if most_likely_type == 3:
        return 'Motion blurred '

def show_label(image, prob_hq, most_likely_type, save_path = ''):
    image_arr = np.array(image)
    show_image = np.ones((image_arr.shape[0] + 30, image_arr.shape[1], image_arr.shape[2]), dtype=np.uint8)*255
    show_image[30:, :] = image_arr
    color = 'red'
    image_type = threshold_res(prob_hq*100)
    from matplotlib import patches
    ax = plt.gca()
    # plt.imshow(show_image)
    most_likely_type_str = get_most_likely_type(most_likely_type)
    # plt.title(f'{image_type} ({prob_hq*100}%) ({most_likely_type_str})')

    if image_type == 'Bad':
        A, B = (20, 5),(135, 28) #Bad
    elif image_type == 'Poor':
        A, B = (135, 5),(235, 28) #Poor
    elif image_type == 'Fair':
        A, B = (230, 5),(330, 28) #Fair
    elif image_type == 'Good':
        A, B = (330, 5),(430, 28) #Good
    else:
        A, B = (430, 5),(568, 28) #Good

    # p = patches.Rectangle((420, 10), 200, 110, linewidth=1, alpha=0.7,
    #                         edgecolor=color, facecolor='none')

    # ax.add_patch(p)
    # ax.text( 50, 25,  'Bad', color = color)
    # ax.text(150, 25,  'Poor', color = color)
    # ax.text(250, 25,  'Fair', color = color)
    # ax.text(350, 25,  'Good', color = color)
    # ax.text(450, 25, 'Exellent', color = color)
    # plt.savefig(save_path, format='png', dpi=1000)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(show_image, 'Bad', (50, 25), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(show_image, 'Poor', (150, 25), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(show_image, 'Fair', (250, 25), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(show_image, 'Good', (350, 25), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(show_image, 'Exellent', (450, 25), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
 
    cv2.rectangle(show_image, A, B, (0,255,0), 2)
    # imshow(show_image)
    return show_image

def show_res(image, results):
    from matplotlib import patches
    plt.imshow(image)
    ax = plt.gca()
    for res in results:
        y1, x1, y2, x2, type = res[0][0], res[0][1], res[0][2], res[0][3], res[0][4]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, alpha=0.7, linestyle="dashed",
                            edgecolor='red', facecolor='none')
        ax.add_patch(p)
        ax.text(x1, y1, type, color = 'red')
    plt.show()

def inferenceOnFrame_origin(original, save_path= ''):
    start = time.time()
    original = original.crop((56, 24, 664, 550))#crop((46, 14, 674, 560)) # Crop đi những viền đen của ảnh

    width, height = original.size

    NperRow = height // sizeSub
    NperCol = width // sizeSub

    coordPatches = [[0, 0]]
    nLQ = 0
    nHQ = 0
    res = []
    draw_res = []
    for i in range(0, NperRow):
        for j in range(0, NperCol):
            nr, nc = sizeSub, sizeSub
            if j == (NperCol - 1):
                nc = width - coordPatches[-1][1]
            if i == (NperRow - 1):
                nr = height - coordPatches[-1][0]
            br, bc = coordPatches[-1][0], coordPatches[-1][1]
            patch = original.crop((bc, br, (bc+nc), (br+nr)))

            predict = inferenceOnPatch(model, patch)
            # print(predict)
            nLQ += predict
            res.append(predict.numpy()[0])
            draw_res.append([
                (bc, br, bc+nc, br+nr, predict.numpy()[0])
            ])
            if predict.numpy()[0] == 2:
                nHQ += 1

            if j != (NperCol - 1):
                coordPatches.append([coordPatches[-1][0], coordPatches[-1][1] + sizeSub])
        if i != (NperRow - 1):
            coordPatches.append([coordPatches[-1][0] + sizeSub, 0])

    prob = nLQ.cpu().numpy()[0] / (NperCol*NperRow)
    prob_hq = nHQ / (NperCol*NperRow)
    print(f'High quality prob: {prob_hq*100} %')
    uniqes_res = np.unique(res, return_counts = True)
    most_likely_type = uniqes_res[0][np.argmax(uniqes_res[1])]
    # show_res(original, draw_res)
    show_label(original, prob_hq, most_likely_type, save_path)
    print(f'Most likely: {most_likely_type}')
    # print(f'Number LQ patches: {nLQ.numpy()[0]} / {NperCol*NperRow} = {prob}')
    # print('Time processing: ', time.time()-start)

    # plt.imshow(original)
    # plt.title('LQ: ' + str(prob))
    # plt.show()
    return original, most_likely_type

def inferenceOnFrame(original, save_path= ''):
    start = time.time()
    original = original.crop((56, 24, 664, 550))#crop((46, 14, 674, 560)) # Crop đi những viền đen của ảnh

    width, height = original.size

    NperRow = height // sizeSub
    NperCol = width // sizeSub

    coordPatches = [[0, 0]]
    nLQ = 0
    nHQ = 0
    res = []
    draw_res = []
    for i in range(0, NperRow):
        for j in range(0, NperCol):
            nr, nc = sizeSub, sizeSub
            if j == (NperCol - 1):
                nc = width - coordPatches[-1][1]
            if i == (NperRow - 1):
                nr = height - coordPatches[-1][0]
            br, bc = coordPatches[-1][0], coordPatches[-1][1]
            patch = original.crop((bc, br, (bc+nc), (br+nr)))

            predict = inferenceOnPatch(model, patch)
            # print(predict)
            nLQ += predict
            res.append(predict.numpy()[0])
            draw_res.append([
                (bc, br, bc+nc, br+nr, predict.numpy()[0])
            ])
            if predict.numpy()[0] == 2:
                nHQ += 1
            
            if predict.numpy()[0] == 1:
                nHQ += 0.2

            if j != (NperCol - 1):
                coordPatches.append([coordPatches[-1][0], coordPatches[-1][1] + sizeSub])
        if i != (NperRow - 1):
            coordPatches.append([coordPatches[-1][0] + sizeSub, 0])

    prob = nLQ.cpu().numpy()[0] / (NperCol*NperRow)
    prob_hq = nHQ / (NperCol*NperRow)
    print(f'High quality prob: {prob_hq*100} %')
    uniqes_res = np.unique(res, return_counts = True)
    most_likely_type = uniqes_res[0][np.argmax(uniqes_res[1])]
    # show_res(original, draw_res)
    # show_label(original, prob_hq, most_likely_type, save_path)
    print(f'Most likely: {most_likely_type}')
    # print(f'Number LQ patches: {nLQ.numpy()[0]} / {NperCol*NperRow} = {prob}')
    # print('Time processing: ', time.time()-start)

    # plt.imshow(original)
    # plt.title('LQ: ' + str(prob))
    # plt.show()
    return original, prob_hq, most_likely_type

def save_images_sequence_to_video(images, file_name = 'project.mp4'):
    # size = images[0].shape[:2]
    # out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
    # for i in range(len(images)):
    #     out.write(images[i])
    # out.release()
    skvideo.io.vwrite(file_name, images)

def inferenceOnFrameAsBatch(frame, coords, sPatch):
    batches = []
    frame = frame.crop((100, 20, 590, 560))
    frame_copy = frame
    w, h = frame.size
    frame = test_transforms(frame)
    # with torch.no_grad():
    # for ind in range(0, len(coords)):
    #     er, ec = min(h, coords[ind][0] + sPatch), min(w, coords[ind][1] + sPatch)
    #     batch  = test_transforms(frame.crop((coords[ind][1], coords[ind][0], ec, er)))
    #     batches.append(batch)
    # batches = torch.stack([b for b in batches])
    batches = torch.stack([frame[:, coords[ind][0]:min(h, coords[ind][0]+sPatch), coords[ind][1]:min(w, coords[ind][1]+sPatch)] for ind in range(0, len(coords))])
    # print('Batches shape: ', batches.shape)
    inputs = batches.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    # print('Batches predict: ', preds)
    nLQ = torch.sum(preds)
    # print('Sum predicts: ', nLQ.numpy(), '\n---------------')
    return frame_copy, nLQ.cpu().numpy() / batches.shape[0]


def main_check_label():
    """
    Bật hàm này về main, để kiểm tra label đầu ra của model và label thực tế
    Brighness = 0
    Dark = 1
    HQ = 2
    Motion = 3
    """
    image_paths = glob('dataset3/Patch_training/Motion/*.png')
    for path_test in image_paths:
        input_image = Image.open(path_test)
        
        model = getModelResNet18()
        # print('==> Method 1')
        t = time.time()
        output, probLQ = inferenceOnFrame(input_image)
        # print(probLQ, time.time() - t)

# if __name__ == '__main__':
def test_image_draft_on():
    path_test = '/home/duongnh/Codes/Y_sinh/Metaplasie intestinale/Image_102314.png'
    paths_test = glob('/home/duongnh/Codes/Y_sinh/Metaplasie intestinale/*.png')
    model = getModelResNet18()

    for path_test in paths_test:
        if 'res' in path_test:
            continue
        input_image = Image.open(path_test)
        

        t = time.time()
        save_path = input_image.filename.split('.png')[0] + '_res.png'
        print(save_path)
        output, probLQ = inferenceOnFrame(input_image, save_path)
        print('Done !')
        # plt.subplot(121)
        # plt.imshow(output)
        # plt.title(probLQ)

        # plt.show()

if __name__ == "__main__":
# def test_image_draft_on_video():
    cap = cv2.VideoCapture('/home/duongnh/Codes/Y_sinh/14-2-2018Sequence_15-14-3-228 original.avi')
    model = getModelResNet18()

    
    a = 0

    nFrame = 0
    nBest  = 0
    since = time.time()
    sequence_of_images = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        print(f'{nFrame}/{frame_count}')
        ret, frame = cap.read()
        if not ret:
            break
        nFrame += 1
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        original, prob_hq, most_likely_type = inferenceOnFrame(pil_image)
        out_image = show_label(original, prob_hq, most_likely_type, '')
        if prob_hq > 25:
            cv2.imwrite(f'captured/{nFrame}_g.png', cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(f'captured/{nFrame}.png', cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))
        # sequence_of_images.append(out_image)

    # save_images_sequence_to_video(sequence_of_images)
    print(f'Number good: {nBest} / {nFrame}')
    print(f'Total time : {time.time() - since}')

    print('Done !')

def main():
# if __name__ == '__main__':
    """
    Đây là hàm main ban đầu của a Thành
    """
    path_test = '/home/duongnh/Codes/Y_sinh/Metaplasie intestinale/Image_102314.png'
    # path_test = 'Gastro_image_quality_classification/NBI_data_for_image_quality_classification/High/Image_2710.png'
    # path_test = 'Gastro_image_quality_classification/NBI_data_for_image_quality_classification/Low_Quality/MotionArtifact/Image_6168.png'
    # path_test = 'Gastro_image_quality_classification/NBI_data_for_image_quality_classification/Low_Quality/Fifty-fifty/Image_12607.png'
    # path_test = 'Gastro_image_quality_classification/NBI_data_for_image_quality_classification/Low_Quality/OtherArtifacts/Image_22888.png'
    input_image = Image.open(path_test)

    model = getModelResNet18()

    print('Method 1')
    t = time.time()
    output, probLQ = inferenceOnFrame(input_image)
    plt.subplot(121)
    plt.imshow(output)
    plt.title(probLQ)
    # print(probLQ, time.time() - t)

    # print('Method 2')
    # startPoints = [[0, 0], [0, 112], [0, 224], [0, 336], [112, 0], [112, 112], [112, 224], [112, 336], [224, 0], [224, 112], [224, 224], [224, 336], [336, 0], [336, 112], [336, 224], [336, 336]]
    # t = time.time()
    # output, probLQ = inferenceOnFrameAsBatch(input_image, startPoints, sizeSub)
    # plt.subplot(122)
    # plt.imshow(output)
    # plt.title(probLQ)
    # print(probLQ, time.time() - t)

    plt.show()

    # threshProb = 0.1

    # cap = cv2.VideoCapture('Video-test/2-5-2018Sequence_10-14-48-210.avi')
    # nFrame = 0
    # nBest  = 0
    # since = time.time()

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     nFrame += 1
    #     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     output, probLQ = inferenceOnFrame(pil_image)
    #     if probLQ < threshProb:
    #         nBest += 1
    #         output.save('Video-test/2-5-2018Sequence_10-14-48-210/' + str(nBest) + '_' + str(probLQ) + '.png')

    # print(f'Number good: {nBest} / {nFrame}')
    # print(f'Total time : {time.time() - since}')
    a = 1

def main2():
    """
    Đây là hàm main ban đầu của a Thành
    """
    path_test = 'dataset3/Patch_validation/HQ/Im_good4336.png'
    # path_test = 'Gastro_image_quality_classification/NBI_data_for_image_quality_classification/High/Image_2710.png'
    # path_test = 'Gastro_image_quality_classification/NBI_data_for_image_quality_classification/Low_Quality/MotionArtifact/Image_6168.png'
    # path_test = 'Gastro_image_quality_classification/NBI_data_for_image_quality_classification/Low_Quality/Fifty-fifty/Image_12607.png'
    # path_test = 'Gastro_image_quality_classification/NBI_data_for_image_quality_classification/Low_Quality/OtherArtifacts/Image_22888.png'
    input_image = Image.open(path_test)

    model = getModelResNet18()

    print('Method 1')
    t = time.time()
    output, probLQ = inferenceOnFrame(input_image)
    plt.subplot(121)
    plt.imshow(output)
    plt.title(probLQ)
    print(probLQ, time.time() - t)

    print('Method 2')
    startPoints = [[0, 0], [0, 112], [0, 224], [0, 336], [112, 0], [112, 112], [112, 224], [112, 336], [224, 0], [224, 112], [224, 224], [224, 336], [336, 0], [336, 112], [336, 224], [336, 336]]
    t = time.time()
    output, probLQ = inferenceOnFrameAsBatch(input_image, startPoints, sizeSub)
    plt.subplot(122)
    plt.imshow(output)
    plt.title(probLQ)
    print(probLQ, time.time() - t)

    plt.show()
    # threshProb = 0.1

    # cap = cv2.VideoCapture('Video-test/2-5-2018Sequence_10-14-48-210.avi')
    # nFrame = 0
    # nBest  = 0
    # since = time.time()

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     nFrame += 1
    #     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     output, probLQ = inferenceOnFrame(pil_image)
    #     if probLQ < threshProb:
    #         nBest += 1
    #         output.save('Video-test/2-5-2018Sequence_10-14-48-210/' + str(nBest) + '_' + str(probLQ) + '.png')

    # print(f'Number good: {nBest} / {nFrame}')
    # print(f'Total time : {time.time() - since}')
    a = 1