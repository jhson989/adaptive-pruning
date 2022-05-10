import numpy as np
import os
import torch
from PIL import Image

class Logger():
    def __init__(self, path, retrain):
        self.logFile = None
        if os.path.isfile(path+"log.txt") and retrain == True:
            self.logFile = open(path+"log.txt", "a")
            self.logFile.write("\n\n [[[Retrain]]] \n")
        else :
            self.logFile = open(path+"log.txt", "w")
        
    def __del__(self):
        self.logFile.close()

    def log(self, logStr):
        print(logStr)
        self.logFile.write(logStr+"\n")
        self.logFile.flush()


def weightsInit(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def printModelSize(model, logger):
    ## total number of parameters
    numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    numNonzeros = sum(torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)
    logger.log("Pruning ratio : %d/%d = (%.3f/%.3f)GB =  %.3f " % 
                (numNonzeros, numParams, float(numNonzeros)*8/pow(2,30), float(numParams)*8/pow(2,30), float(numNonzeros)/numParams*100)
              )

    return float(numNonzeros)/numParams


def listAllImg(dataPath):
    filePaths = []
    for root, dir, files in os.walk(dataPath):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                filePath = os.path.join(root, file)
                filePaths.append(filePath)
            
    return filePaths

def saveImage(args, epoch, i, imgs, interval=150):
    
    if i % interval != 0:
        return
        
    imgs = [img for img in imgs if img != None]

    for idx in range(len(imgs)):
        imgs[idx] = np.transpose(np.float32(imgs[idx].to("cpu").detach().numpy()[0])*255, (1,2,0))

    img = combine(imgs)
    img.save(args.savePath+"img_%d_%d.png"%(epoch,i))

def combine(imgs):

    for i in range(len(imgs)):
        imgs[i] = Image.fromarray(np.uint8(imgs[i]))

    widths, heights = zip(*(i.size for i in imgs))
    totalWidth = sum(widths)
    totalHeight = max(heights)

    new_img = Image.new("RGB", (totalWidth, totalHeight))
    offset = 0
    for img in imgs:
        new_img.paste(img, (offset, 0))
        offset += img.size[0]

    return new_img

"""**굵은 텍스트**# 새 섹션

# Dataloader
"""

import torch
from torchvision import datasets, transforms

def getDataLoader(train, args, logger):
    # Define data
    mnistTransform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0,))
    ])

    if train == True:

        trainDataset = datasets.MNIST(
            "./data/train",
            train=True,
            download=True,
            transform=mnistTransform
        )

        return torch.utils.data.DataLoader(
            dataset=trainDataset,
            batch_size = args.batchSize,
            shuffle=True,
            num_workers=args.numWorkers,
            drop_last=True
        )

    else :
        testDataset = datasets.MNIST(
            "./data/eval",
            train=False,
            download=True,
            transform=mnistTransform
        )

        return torch.utils.data.DataLoader(
            dataset=testDataset,
            batch_size = args.batchSize,
            shuffle=False,
            num_workers=args.numWorkers
        )

"""# Model"""

import torch
from torch import nn
import torch.nn.functional as F


#### Over-parameterized LeNet to solve the mnist problem
class LeNet(nn.Module):
    def __init__(self, overFactor=5):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, overFactor*6, 3)
        self.conv2 = nn.Conv2d(overFactor*6, overFactor*16, 3)
        self.fc1 = nn.Linear(overFactor * 16 * 5 * 5, overFactor * 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(overFactor * 120, overFactor * 84)
        self.fc3 = nn.Linear(overFactor * 84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""# PruningTrainer"""

# Commented out IPython magic to ensure Python compatibility.
import torch
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch import nn

class PruningTranier:

    def __init__(self, args, logger):

        ### Arguments
        self.args = args
        self.logger = logger

        ### Train Policy
        # criterion
        self.crit = nn.CrossEntropyLoss().to(self.args.device)
        self.bestLoss = 2.0
        self.bestPrunedLoss = 2.0

        ### Pruning policy
        self.recentLosses = [2.0]
        self.pruned = False
        self.startPrune = False


    def train(self, model, dataLoader, evalDataLoader=None):

        ### Data
        self.dataLoader = dataLoader
        self.evalDataLoader = evalDataLoader

        ### Model
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        ### Re-training
        startEpoch = 0
        if self.args.retrain == True:
            startEpoch = self.load(self.args.loadPath)

        ### Training iteration
        for epoch in range(startEpoch, self.args.numEpoch):

            ### Train
            avgLoss = 0.0
            self.model.train()
            for idx, (img, gt) in enumerate(self.dataLoader):

                ### learning
                img, gt = img.to(self.args.device), gt.to(self.args.device)
                self.optimizer.zero_grad()
                pred = self.model(img)
                loss = self.crit(pred, gt)
                loss.backward()
                self.optimizer.step()

                ### Logging
                avgLoss = avgLoss + loss.item()
                if idx % self.args.logFreq == 0 and idx != 0: 
                    self.logger.log("[[%4d/%4d] [%4d/%4d]] loss CE(%.3f)"  % (epoch, self.args.numEpoch, idx, len(self.dataLoader), avgLoss/self.args.logFreq))
                    avgLoss = 0.0

            ### Eval
            if self.evalDataLoader is not None :
                avgLoss = self.eval(self.evalDataLoader, epoch)
            
            ### Prune
            self.adaptivePrune(epoch, avgLoss)



    def eval(self, evalDataLoader, epoch):
        
        ### Eval
        self.model.eval()
        with torch.no_grad():
            avgLoss = 0.0
            for idx, (img, gt) in enumerate(evalDataLoader):

                ### Forward
                img, gt = img.to(self.args.device), gt.to(self.args.device)
                pred = self.model(img)
                loss = self.crit(pred, gt)

                avgLoss = avgLoss + loss.item()

            ### Logging
            avgLoss = avgLoss/len(evalDataLoader)
            self.logger.log("Eval loss : CE(%.3f)" % (avgLoss))

            
            if avgLoss < self.bestLoss :
                self.logger.log("Best model at %d" % epoch)
                self.bestLoss = avgLoss
                self.save("best.pth", epoch)

            if avgLoss < self.bestPrunedLoss and self.pruned == True :
                self.logger.log("Best pruned model at %d" % epoch)
                self.bestPrunedLoss = avgLoss
                self.save("pruned_best.pth", epoch)

            self.save("last.pth", epoch)

            return avgLoss



    def adaptivePrune(self, epoch, loss, searchLen = 10):

        if self.args.prune == False:
            return

        self.recentLosses.append(loss)
        self.recentLosses = self.recentLosses[1:] if len(self.recentLosses) > searchLen else self.recentLosses

        self.remove()
        printModelSize(self.model, logger)

        if self.startPrune == False :
            # Pruning
            if sum( int(p<loss) for p in self.recentLosses ) >= (searchLen-2) :
                self.recentLosses = [2.0]
                amount = self.args.pruneAmount + self.args.amountDecay
                self.args.pruneAmount =  amount if (0.1 <= amount and amount <= 0.99) else self.args.pruneAmount
                self.startPrune = True
            # Training
            else :
                return
                
        elif self.startPrune == True :
            #Training
            if sum( int(p<loss) for p in self.recentLosses ) >= (searchLen-2) :
                self.recentLosses = [2.0]
                self.startPrune = False
                return


        # Adaptive pruning
        self.logger.log( "next : prune (%.1f) pct\n" % ((1-self.args.pruneAmount)*100) )
        self.prune()
        #self.args.amountDecay = self.args.amountDecay + 0.001 if self.args.amountDecay < self.args.maxDecay else self.args.amountDecay
        #self.args.amountDecay = self.args.amountDecay - 0.001 if self.args.amountDecay > self.args.minDecay else self.args.amountDecay
        


    def prune(self):
        
        if self.args.prune == False:
            return

        if self.pruned == True :
            self.remove()

        self.pruned = True
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.args.pruneAmount)
                prune.l1_unstructured(module, name='bias', amount=self.args.pruneAmount)



    def remove(self):

        if self.args.prune == False:
            return

        if self.pruned == False:
            return
            
        self.pruned = False
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')
                prune.remove(module, 'bias')



    def save(self, filename, numEpoch):

        filename = self.args.savePath + filename
        
        self.remove()
        self.model.eval()
        torch.save({
            "epoch" : numEpoch,
            "modelStateDict" : self.model.state_dict(),
            "optimizerStateDict" : self.optimizer.state_dict(),
            "pruned" : self.pruned,
            "pruneAmount" : self.args.pruneAmount,
            "amountDecay" : self.args.amountDecay
            }, filename)



    def load(self, filename):

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["modelStateDict"])
        printModelSize(self.model, logger)

        self.optimizer.load_state_dict(checkpoint["optimizerStateDict"])
        self.args.pruneAmount = checkpoint["pruneAmount"]
        self.args.amountDecay = checkpoint["amountDecay"]

        if checkpoint["pruned"]:
            self.prune()

        return checkpoint["epoch"]

"""# main"""

import argparse, os
import random
import torch

#############################################################
# Hyper-parameters
#############################################################
import easydict
args = easydict.EasyDict({ 
    "train" : True, 
    
    # Train policy
    "numEpoch" : 1000,
    "batchSize" : 1024,
    "lr" : 1e-4,
    "manualSeed" : 1,

    # Prune
    "prune" : True,
    "pruneAmount" : 0.2,
    "amountDecay" : 0.01,
    "maxDecay" : 0.05,
    "minDecay" : 0.01,
    "overFactor" : 2,

    # Record
    "savePath" : "./result/pruned/",
    "retrain" : True, 
    "loadPath" : "./result/unpruned/best.pth",
    "logFreq" : 20,   

    # Hardware
    "ngpu" : 1,
    "numWorkers" : 5,    
})

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

try:
    if not os.path.exists(args.savePath):
        os.makedirs(args.savePath)
except OSError:
    print("Error: Creating save folder. [" + args.savePath + "]")

if torch.cuda.is_available() == False:
    args.ngpu = 0

if args.ngpu == 1:
    args.device = torch.device("cuda")
else :
    args.device = torch.device("cpu")


logger = Logger(args.savePath, args.retrain)
logger.log(str(args))

model = LeNet(args.overFactor).to(device=args.device)

logger.log("[[[Train]]] Train started..")

# Define data
trainDataLoader = getDataLoader(True, args, logger)
testDataLoader = getDataLoader(False, args, logger)

# Define trainer
trainer = PruningTranier(args=args, logger=logger)

# Start training
trainer.train(model, trainDataLoader, testDataLoader)
