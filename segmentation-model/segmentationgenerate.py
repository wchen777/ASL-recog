import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from .handsegmentation import HandSegModel


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

    checkpoint = torch.load('checkpoints/checkpointhandseg7.pt')
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


cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (384, 288))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if i % 5 == 0:
        i = 0
        mask = SegmentHands(rgb)
        colmask = getColoredMask(frame, mask)

    cv2.imshow('color', np.hstack((frame, colmask)))
    key = cv2.waitKey(24)
    if key & 0xFF == ord('q'):
        break
    i += 1
cap.release()
cv2.destroyAllWindows()

