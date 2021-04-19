import base64

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn


deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False,
                                                 progress=True,
                                                 num_classes=2)


# we use model module from torchvision to get the deeplabv3_resnet50 model.
# We specify the number of classes usingnum_classes as two because we will generate two grayscale images,
# one for predicting region with hands and another with no hands.

class HandSegModel(nn.Module):
    def __init__(self):
        super(HandSegModel, self).__init__()
        self.dl = deeplab

    def forward(self, x):
        y = self.dl(x)['out']
        return y


# can pass np array or path to image file
def SegmentHands(pathtest):
    if isinstance(pathtest, np.ndarray):
        img = Image.fromarray(pathtest)
    else:
        img = Image.open(pathtest)

    preprocess = transforms.Compose([transforms.Resize((288, 384), 2),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    Xtest = preprocess(img)

    checkpoint = torch.load('checkpointhandseg1.pt', map_location=torch.device('cpu'))
    model = HandSegModel()
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        model.to(device)
        Xtest = Xtest.to(device).float()
        ytest = model(Xtest.unsqueeze(0).float())
        ypos = ytest[0, 1, :, :].clone().detach().cpu().numpy()
        yneg = ytest[0, 0, :, :].clone().detach().cpu().numpy()
        ytest = ypos >= yneg

    mask = ytest.astype('float32')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def getColoredMask(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] += mask.astype('uint8') * 250
    masked = cv2.addWeighted(image, 1.0, color_mask, 1.0, 0.0)
    return masked


# decode from base64 and save image
def process_image_segmentation(img_url):
    with open('seg_pic.png', 'wb') as output:
        output.write(base64.b64decode(img_url))


def getMaskedImage(image, mask):
    black_mask = np.zeros_like(image)

    log = mask.astype('uint8') != 0.
    black_mask[:, :, 0] = image[:, :, 0] * log
    black_mask[:, :, 1] = image[:, :, 1] * log
    black_mask[:, :, 2] = image[:, :, 2] * log

    # replace black with white
    return np.where(black_mask == 0, black_mask, 255)


def readImage():
    im = cv2.imread('seg_pic.png')
    im = cv2.resize(im, (384, 288))
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    mask = SegmentHands(rgb)

    return getMaskedImage(im, mask)
