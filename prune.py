
import argparse, os
import random
import secrets
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import easydict


#####################################################################################
# Configuration
#####################################################################################
args = easydict.EasyDict({ 
    "train" : True, 
    
    # Train policy
    "numEpoch" : 30,
    "batchSize" : 1024*4,
    "lr" : 1e-3,
    "manualSeed" : 1,

    # Record
    "retrain" : False, 
    "savePath" : "./result/pruned/",
    "loadPath" : "./result/unpruned/best.pth",
    "logFreq" : 10,

    # Hardware
    "ngpu" : 1,
    "numWorkers" : 5,    

    # Genetic 
    "numMember" : 20,
    "numGeneration" : 10,

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

#####################################################################################
# Logger
#####################################################################################
class Logger():
    def __init__(self, path, retrain):
        self.logFile = None
        if os.path.isfile(path+"log.txt") and retrain == True:
            self.logFile = open(path+"log.txt", "a")
            self.log("\n\n----------------------[[[Retrain]]]----------------------\n")
        else :
            self.logFile = open(path+"log.txt", "w")
        
    def __del__(self):
        self.logFile.close()

    def log(self, logStr):
        print(logStr)
        self.logFile.write(logStr+"\n")
        self.logFile.flush()


logger = Logger(args.savePath, args.retrain)
logger.log(str(args))



#####################################################################################
# Dataloader
#####################################################################################
def getDataLoader(train):
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

trainDataLoader = getDataLoader(True)
testDataLoader = getDataLoader(False)
crit = nn.CrossEntropyLoss().to(args.device)



#####################################################################################
# Model
#####################################################################################
class EvolvingSparseConnectedModel(nn.Module):

    def __init__(self, ID):
        self.ID = ID
        self.accuracy = 1000000.0
        super(EvolvingSparseConnectedModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 100, bias=False)
        self.fc2 = nn.Linear(100, 100, bias=False)
        self.fc3 = nn.Linear(100, 100, bias=False)
        self.fc4 = nn.Linear(100, 100, bias=False)
        self.fc5 = nn.Linear(100, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)

    def learn(self):
        logger.log("\n\n------------ AI[%d] start ------------\n" % self.ID)
        self.to(args.device)
        optimizer = optim.Adam(self.parameters(), lr=args.lr)

        #########################################################################
        ### Pruning
        amount = float(90+(random.randrange(4,9)))/100.0
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight') # For logging
                prune.l1_unstructured(module, name='weight', amount=amount)
        self.printSize()

        for epoch in range(args.numEpoch):

            #########################################################################
            ### Train
            self.train()
            for idx, (img, gt) in enumerate(trainDataLoader):
                ### learning
                img, gt = img.to(args.device), gt.to(args.device)
                optimizer.zero_grad()
                pred = self.forward(img)
                loss = crit(pred, gt)
                loss.backward()
                optimizer.step()

                ### Logging
                #avgLoss = avgLoss + loss.item()
                #if idx % args.logFreq == 0 and idx != 0: 
                #    logger.log(
                #        "[%d] : [[%4d/%4d] [%4d/%4d]] loss CE(%.3f)" % 
                #        (self.ID, epoch, args.numEpoch, idx, len(trainDataLoader), avgLoss/args.logFreq)
                #    )
                #    avgLoss = 0.0
                    
            #########################################################################
            ### Eval
            total = 0.0
            correct = 0
            self.eval()
            with torch.no_grad():
                for idx, (img, gt) in enumerate(testDataLoader):
                    img, gt = img.to(args.device), gt.to(args.device)
                    pred = self.forward(img)

                    _, predIdx = torch.max(pred, 1)
                    total += gt.size(0)
                    correct += (predIdx == gt).sum().float()


                ### Logging
                accuracy = 100.0*correct/total
                logger.log("Idx(%d) : eval(%.3f)%%" % (epoch, accuracy))
                self.accuracy = accuracy if accuracy < self.accuracy else self.accuracy

        #########################################################################
        ### Remove
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')
        self.cpu()

    def evolve(self, parent1, parent2):
        logger.log(" --- evolved : %d <- %d %d" % (self.ID, parent1.ID, parent2.ID))
        for m, p1, p2 in zip(self.modules(), parent1.modules(), parent2.modules()):
            if isinstance(m, torch.nn.Linear):
                with torch.no_grad():
                    m.weight.copy_((p1.weight + p2.weight)/2)

    def printSize(self):
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        numNonzeros = sum(torch.count_nonzero(p) for p in self.parameters() if p.requires_grad)
        logger.log(" --- pruned ratio : %d/%d = (%.3f/%.3f)GB =  %.3f %%" % 
                (numNonzeros, numParams, float(numNonzeros)*8/pow(2,30), float(numParams)*8/pow(2,30), float(numNonzeros)/numParams*100)
              )




#####################################################################################
# Model
#####################################################################################

for gen in range(args.numGeneration):
    logger.log("\n\n==============================================")
    logger.log("Generation %d start" % (gen))
    logger.log("==============================================\n")
    AIs = [EvolvingSparseConnectedModel(m) for m in range(args.numMember)]

    #### Train
    for AI in AIs:
        AI.learn()
    
    #### Sort
    AIs.sort(key=lambda x: x.accuracy)
    lottery = []
    for idx, AI in enumerate(AIs):
        for _ in range(idx, args.numMember):
            lottery.append(AI.ID) 
    logger.log(str(lottery))

    #### Pick
    AIs.sort(key=lambda x: x.ID)
    children = [EvolvingSparseConnectedModel(m) for m in range(args.numMember)]
    
    for child in children:
        parent1 = random.choice(lottery)
        parent2 = random.choice(lottery)
        child.evolve(AIs[parent1], AIs[parent2])

    AIs = children
    children = []