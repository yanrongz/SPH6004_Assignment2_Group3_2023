# Author: Liang Jingyu
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from tqdm import trange
from model import attention, RMSELoss, r_2, MAE
from ICUData import ICUData

def train_test_split(data, label, rate):
    if len(data) != len(label):
        return -1
    l = len(data)
    train_data = data[:round(l*rate)]
    test_data = data[round(l*rate)+1:]
    train_label = label[:round(l*rate)]
    test_label = label[round(l*rate)+1:]
    return train_data, train_label, test_data, test_label

def train(model,optimizer,epoch,dataloader):
    res = []
    size = len(dataloader.dataset)
    output = open("./output.txt","w")
    for i in trange(epoch):
        for batch,(X,y) in enumerate(dataloader):
            X=X.cuda()
            y=y.cuda()
            pred=model.forward(X)
            loss=model.loss(pred,y)
            output.write(f"pred=\n{pred}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                res.append(loss)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    output.close()
    return res
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    yhat = []
    ytruth = []
    with torch.no_grad():
        for X, y in dataloader:
            X=X.cuda()
            y=y.cuda()
            pred = model(X)
            yhat.extend(torch.unsqueeze(pred,0))
            ytruth.extend(torch.unsqueeze(y,0))
            test_loss += loss_fn(pred, y).item()

    test_loss /= size
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    ytruth = torch.cat(ytruth,dim=0)
    yhat = torch.cat(yhat,dim=0)
    r2 = r_2(yhat,ytruth)
    print(f"r_2 score = {r2}")
    mae = MAE(ytruth, yhat)
    print('Mean Absolute Error: %.2f' % mae)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    learning_rate = 1e-3
    batch_size = 1
    epochs = 1
    model = attention(13)#Based on number of Features 
    model.cuda()
    input_data = torch.load("2Step_Transformer.pt")
    input_label = torch.load("2Step_Transformer_label.pt")
    train_data, train_label, test_data, test_label = train_test_split(input_data, input_label, 0.8)
    trainSet = ICUData(train_data,train_label)
    testSet = ICUData(test_data,test_label)
    trainLoader = DataLoader(trainSet, batch_size=batch_size)
    testLoader = DataLoader(testSet, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    result = train(model,optimizer,epochs,trainLoader)
    test_loss = RMSELoss    
    test_loop(testLoader,model,RMSELoss)
    plt.plot(list(x for x in range(len(result))),result)
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.show()