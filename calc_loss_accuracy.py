from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def model_training(model, device, train_loader, optimizer, scheduler, criterion):
    scheduler  = scheduler
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} LR ={scheduler.get_last_lr()[0]} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
    return(train_acc,train_losses)


def model_testing_old(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        miss_classified_data = [[],[],[]]
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            for i in range(0,len(pred.tolist())):
                #print (pred[i][0],test[i])
                if (pred.tolist()[i][0]!=target.tolist()[i]):
                    #print (data[i])
                    #imshow(data[i])
                    miss_classified_data[0].append(pred.tolist()[i][0])
                    miss_classified_data[1].append(target.tolist()[i])
                    miss_classified_data[2].append(data[i])
            
            

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    return(test_acc,test_losses,miss_classified_data)


def model_testing(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    misclassified_data = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)

        output = model(data)  # Get the model's prediction (ignore conv_features)
        loss = criterion(output, target)

        test_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Store misclassified images along with their information
        misclassified_mask = ~pred.eq(target.view_as(pred)).squeeze()
        for i in range(len(data)):
            if misclassified_mask[i]:
                info = {
                    'image': data[i].cpu(),
                    'predicted': pred[i].cpu(),
                    'actual': target[i].cpu()
                }
                misclassified_data.append(info)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    print (type(misclassified_data))
    return (test_acc,test_losses,misclassified_data)


