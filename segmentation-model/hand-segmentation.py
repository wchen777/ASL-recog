import os
import torch
import torch.optim as optim
import numpy as np
import glob
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# https://towardsdatascience.com/semantic-hand-segmentation-using-pytorch-3e7a0a0386fa

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


class SegDataset(Dataset):

    def __init__(self, parentDir, imageDir, maskDir):
        self.imageList = glob.glob(parentDir + '/' + imageDir + '/*')
        self.imageList.sort()
        self.maskList = glob.glob(parentDir + '/' + maskDir + '/*')
        self.maskList.sort()

    def __getitem__(self, index):
        preprocess = transforms.Compose([
            transforms.Resize((384, 288), 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        X = Image.open(self.imageList[index]).convert('RGB')
        X = preprocess(X)

        trfresize = transforms.Resize((384, 288), 2)
        trftensor = transforms.ToTensor()

        yimg = Image.open(self.maskList[index]).convert('L')
        y1 = trftensor(trfresize(yimg))
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0)

        return X, y

    def __len__(self):
        return len(self.imageList)


# EGOdataset = SegDataset(os.path.join('data', 'egodata'), 'images', 'masks')

HOFdataset = SegDataset(os.path.join('data', 'hand_over_face'), 'images_resized', 'masks')
GTEAdataset = SegDataset(os.path.join('data', 'gtea'), 'Images', 'Masks')

# combine dataset
megaDataset = ConcatDataset([HOFdataset, GTEAdataset])


# TTR is Train Test Ratio
def trainTestSplit(dataset, TTR):
    trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valDataset = torch.utils.data.Subset(dataset, range(int(TTR * len(dataset)), len(dataset)))
    return trainDataset, valDataset


batchSize = 2
trainDataset, valDataset = trainTestSplit(megaDataset, 0.9)
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True)
valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True, drop_last=True)


# performance metrics, mean IOU (intersectino over union)
def meanIOU(target, predicted):
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return

    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return

    iousum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)

        intersection = np.logical_and(target_arr, predicted_arr).sum()
        union = np.logical_or(target_arr, predicted_arr).sum()
        if union == 0:
            iou_score = 0
        else:
            iou_score = intersection / union
        iousum += iou_score

    miou = iousum / target.shape[0]
    return miou


# pixel accuracy for performance metric
def pixelAcc(target, predicted):
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return

    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return

    accsum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)

        same = (target_arr == predicted_arr).sum()
        a, b = target_arr.shape
        total = a * b
        accsum += same / total

    pixelAccuracy = accsum / target.shape[0]
    return pixelAccuracy


model = HandSegModel()
optimizer = optim.Adam(model.parameters(), lr=0.00005)
loss_fn = nn.BCEWithLogitsLoss()
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)


# train our model with loop
def training_loop(n_epochs, optimizer, lr_scheduler, model, loss_fn, train_loader, val_loader, lastCkptPath=None):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    tr_loss_arr = []
    val_loss_arr = []
    meanioutrain = []
    pixelacctrain = []
    meanioutest = []
    pixelacctest = []
    prevEpoch = 0

    if lastCkptPath != None:
        checkpoint = torch.load(lastCkptPath)
        prevEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    tr_loss_arr = checkpoint['Training Loss']
        val_loss_arr = checkpoint['Validation Loss']
        meanioutrain = checkpoint['MeanIOU train']
        pixelacctrain = checkpoint['PixelAcc train']
        meanioutest = checkpoint['MeanIOU test']
        pixelacctest = checkpoint['PixelAcc test']
        print("loaded model, ", checkpoint['description'], "at epoch", prevEpoch)
        model.to(device)

    for epoch in range(0, n_epochs):
        train_loss = 0.0
        pixelacc = 0
        meaniou = 0

        pbar = tqdm(train_loader, total=len(train_loader))
        for X, y in pbar:
            torch.cuda.empty_cache()
            model.train()
            X = X.to(device).float()
            y = y.to(device).float()
            ypred = model(X)
            loss = loss_fn(ypred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss_arr.append(loss.item())
            meanioutrain.append(meanIOU(y, ypred))
            pixelacctrain.append(pixelAcc(y, ypred))
            pbar.set_postfix({'Epoch': epoch + 1 + prevEpoch,
                              'Training Loss': np.mean(tr_loss_arr),
                              'Mean IOU': np.mean(meanioutrain),
                              'Pixel Acc': np.mean(pixelacctrain)
                              })

        with torch.no_grad():

            val_loss = 0
            pbar = tqdm(val_loader, total=len(val_loader))
            for X, y in pbar:
                torch.cuda.empty_cache()
                X = X.to(device).float()
                y = y.to(device).float()
                model.eval()
                ypred = model(X)

                val_loss_arr.append(loss_fn(ypred, y).item())
                pixelacctest.append(pixelAcc(y, ypred))
                meanioutest.append(meanIOU(y, ypred))

                pbar.set_postfix({'Epoch': epoch + 1 + prevEpoch,
                                  'Validation Loss': np.mean(val_loss_arr),
                                  'Mean IOU': np.mean(meanioutest),
                                  'Pixel Acc': np.mean(pixelacctest)
                                  })

        checkpoint = {
            'epoch': epoch + 1 + prevEpoch,
            'description': "add your description",
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Training Loss': tr_loss_arr,
            'Validation Loss': val_loss_arr,
            'MeanIOU train': meanioutrain,
            'PixelAcc train': pixelacctrain,
            'MeanIOU test': meanioutest,
            'PixelAcc test': pixelacctest
        }
        torch.save(checkpoint, 'checkpoints/checkpointhandseg' + str(epoch + 1 + prevEpoch) + '.pt')
        lr_scheduler.step()

    return tr_loss_arr, val_loss_arr, meanioutrain, pixelacctrain, meanioutest, pixelacctest


# call the training loop,
# make sure to pass correct checkpoint path, or none if starting with the training

retval = training_loop(3,
                       optimizer,
                       lr_scheduler,
                       model,
                       loss_fn,
                       trainLoader,
                       valLoader,
                       'checkpoints/checkpointhandseg.pt')


# after the training loop returns, we can plot the data
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
N = 1000
ax[0][0].plot(running_mean(retval[0], N), 'r.', label='training loss')
ax[1][0].plot(running_mean(retval[1], N), 'b.', label='validation loss')
ax[0][1].plot(running_mean(retval[2], N), 'g.', label='meanIOU training')
ax[1][1].plot(running_mean(retval[4], N), 'r.', label='meanIOU validation')
ax[0][2].plot(running_mean(retval[3], N), 'b.', label='pixelAcc  training')
ax[1][2].plot(running_mean(retval[5], N), 'b.', label='pixelAcc validation')
for i in ax:
    for j in i:
        j.legend()
        j.grid(True)
plt.show()

