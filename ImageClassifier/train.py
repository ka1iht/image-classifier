import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
from collections import OrderedDict
from PIL import Image

def validation():
    print("Parameter Validation....")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--GPU option is Enabled...but no GPU is detected!")
    if(not os.path.isdir(args.data_directory)):
        raise Exception('Directory does not exist!')
    data_dir = os.listdir(args.data_directory)
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception('Missing: Test, Train or Validation sub-directories')
    if args.arch not in ('vgg','densenet',None):
        raise Exception('Please choose either: vgg or densenet')

def data_process(data_dir, train_dir, test_dir, valid_dir):
    print("Processing Data into Iterators....")
    data_transforms = transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ]) 


    train_transforms = transforms.Compose([transforms.Resize(225), 
                                       transforms.RandomCrop(224), 
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ]) 

    valid_transforms = transforms.Compose([transforms.Resize(225), 
                                       transforms.RandomCrop(224), 
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ]) 


    test_transforms = transforms.Compose([transforms.Resize(225), 
                                       transforms.RandomCrop(224), 
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ]) 


    data = datasets.ImageFolder(data_dir, transform = data_transforms)
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    dataloader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = True)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    loaders = {'train':trainloader,'valid':validloader,'test':testloader,'labels':cat_to_name}
    return loaders

def get_data():
    print("Retrieving Data....")
    data_dir = args.data_directory
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    return data_process(data_dir, train_dir, test_dir, valid_dir)

def build_model(data):
    print("Building Network Model")
    if (args.arch is None):
        arch_type = 'vgg'
    else:
        arch_type = args.arch
    if (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        input_node=25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_node=1024
    if (args.hidden_units is None):
        hidden_units = 4096
    else:
        hidden_units = args.hidden_units
    for param in model.parameters():
        param.requires_grad = False
    hidden_units = int(hidden_units)
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_node, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    return model

def testing(model, loader, device='cpu'):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train(model, data):
    print("Training Model....")
    
    print_every=40
    
    if (args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = args.learning_rate
    if (args.epochs is None):
        epochs = 3
    else:
        epochs = args.epochs
    if (args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'
    
    learn_rate = float(learn_rate)
    epochs = int(epochs)
    
    trainloader = data['train']
    validloader = data['valid']
    testloader = data['test']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    running_loss = 0
    steps = 0
    model.to(device)
    
    for i in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logprobs = model.forward(inputs)
            loss = criterion(logprobs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logprobs = model.forward(inputs)
                        batch_loss = criterion(logprobs, labels)
                    
                        valid_loss += batch_loss.item()
                    
                        prob = torch.exp(logprobs)
                        top_prob, top_class = prob.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {i+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("\nTraining Complete!")            
    
    test_result = testing(model, testloader, device)
    print('Final Accuracy: {}'.format(test_result))
    return model

def save_model(model):
    print("Saving Trained Model....")
    if (args.save_dir is None):
        save_dir = 'CheckPoint.pth'
    else:
        save_dir = args.save_dir
    checkpoint = {
                'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    return 0

def create_model():
    validation()
    data = get_data()
    model = build_model(data)
    model = train(model,data)
    save_model(model)
    return None

def parse():
    parser = argparse.ArgumentParser(description='Train a neural network with the following criteria!')
    parser.add_argument('data_directory', help='Data Directory (Required)')
    parser.add_argument('--save_dir', help='Directory to save the Neural Network.')
    parser.add_argument('--arch', help='Pretrained Models to use [vgg,densenet]')
    parser.add_argument('--learning_rate', help='Learning Rate')
    parser.add_argument('--hidden_units', help='Number of hidden units')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu',action='store_true', help='gpu')
    args = parser.parse_args()
    return args

def main():
    print("Creating a Neural Network Model....")
    global args
    args = parse()
    create_model()
    print("Model finished!")
    return None

main()