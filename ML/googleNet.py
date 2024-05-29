# %%
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split, SubsetRandomSampler, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pickle
import random

# %%
#randomSeed=42
randomSeed=random.randint(0, 2**32 - 1)
modelName="GoogleNet"


# %%
# Define a transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your data
train_datasets=[]
val_datasets=[]
test_datasets=[]
test_loader_image=None
test_loader_tactile=None
for path in ['Data_tactile','Data']:
    #dataset = datasets.ImageFolder('Data', transform=transform)
    dataset = datasets.ImageFolder(path, transform=transform)

    # Force the data to be balanced
    targets = [t[1] for t in dataset]
    class_indices = [np.where(np.array(targets) == i)[0] for i in range(10)]

    # Split the data into training, validation and testing sets
    train_indices = []
    val_indices = []
    test_indices = []
    test_mixed_indices = []

    np.random.seed(randomSeed)  # Ensure reproducibility
    for class_index in class_indices:
        np.random.shuffle(class_index)
        split_train = int(np.floor(0.7 * len(class_index)))
        split_val = int(np.floor(0.8 * len(class_index)))
        split_test1 = int(np.floor(0.9 * len(class_index)))
        
        train_indices.extend(class_index[:split_train])
        val_indices.extend(class_index[split_train:split_val])
        test_indices.extend(class_index[split_val:split_test1])
        test_mixed_indices.extend(class_index[split_test1:])
    
    # Create the datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    testMix_dataset = Subset(dataset, test_mixed_indices)
    
    if path=='Data':
        test_loader_image = DataLoader(test_dataset, batch_size=64, shuffle=True)
    else:
        test_loader_tactile = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Append the datasets 
    train_datasets.append(train_dataset)
    val_datasets.append(val_dataset)
    test_datasets.append(testMix_dataset)

# Merge the data
train_dataset = ConcatDataset(train_datasets)
val_dataset = ConcatDataset(val_datasets)
test_dataset = ConcatDataset(test_datasets)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Save DataLoader for later testing
with open(modelName+'_dataloader_mixed.pkl', 'wb') as f:
    pickle.dump(test_loader, f)
with open(modelName+'_dataloader_image.pkl', 'wb') as f:
    pickle.dump(test_loader_image, f)
with open(modelName+'_dataloader_tactile.pkl', 'wb') as f:
    pickle.dump(test_loader_tactile, f)

# %%

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
lr_list=[0.1,0.01,0.001,0.0001]
momentum_list=[0.3,0.6,0.9,0.99]
epochs=100000

# %%
with open('googleNet_log.txt', 'w', newline='') as recordFile:
    for lr_i in lr_list:
        for momentum_i in momentum_list:
            # Load the pretrained alexnet model
            model = models.googlenet(pretrained=True)

            # Move the model to the GPU if available
            model = model.to(device)

            # Assuming your images are labeled from 0 to 9
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
            optimizer = optim.SGD(model.parameters(), lr=lr_i, momentum=momentum_i)

            # Train the model
            prev_loss_f1 = -1
            print(f"lr: {lr_i}, momentum: {momentum_i}")
            recordFile.write(f"lr: {lr_i}, momentum: {momentum_i} : ")
            # CSV file to save the scores
            with open(f'Scores/googleNet_lr{lr_i}_momentum{momentum_i}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Train F1', 'Test F1', 'Saved'])

            with open(f'Scores/googleNet_lr{lr_i}_momentum{momentum_i}.csv', 'a', newline='') as f:
                writer = csv.writer(f)

                repeat_checker=0
                repeat_val_f1=0
                sw=True
                for epoch in tqdm(range(epochs)):  # Maximum number of epochs
                    all_preds = []
                    all_labels = []

                    model.train()  # Set the model to training mode
                    for inputs, labels in train_loader:
                        inputs=inputs.to(device)
                        labels=labels.to(device)
                        model.to(device)
                        optimizer.zero_grad()  # Zero the parameter gradients
                        outputs = model(inputs)  # Forward pass
                        loss = criterion(outputs, labels)  # Compute loss
                        loss.backward()  # Backward pass
                        optimizer.step()  # Optimize
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                    train_f1 = f1_score(all_labels, all_preds, average='macro')
                    #print(f"Epoch {epoch+1}, Train F1: {f1}")

                    # Validation phase
                    all_preds = []
                    all_labels = []
                    model.eval()  # Set the model to evaluation mode
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                    test_f1 = f1_score(all_labels, all_preds, average='macro')
                    #print(f"Epoch {epoch+1}, Test F1: {f1}")

                    

                    # Save the model if the test loss is lower than the previous one
                    saved="N"
                    if test_f1 > prev_loss_f1:
                        torch.save(model.state_dict(), f'Model/googleNet_lr{lr_i}_momentum{momentum_i}.pth')
                        prev_loss_f1 = test_f1
                        saved="Y"
                    # Save the scores
                    writer.writerow([train_f1, test_f1, saved])

                    # If F1 score is not improving, stop the training
                    if repeat_val_f1==test_f1:
                        repeat_checker+=1
                        if repeat_checker>20:
                            print('F1 score is not improving. Stopping training.')
                            recordFile.write(f"F1 score is not improving. Stopping training. epoch : {epoch} \n")
                            sw=False
                            break
                    else:
                        repeat_checker=0
                        repeat_val_f1=test_f1

                    # If the test loss is low enough, stop the training
                    if test_f1 > 0.995:  # Set your threshold here
                        print('F1 score is high enough. Stopping training.')
                        recordFile.write(f"F1 score is high enough. Stopping training. epoch : {epoch} \n")
                        sw=False
                        break
                    
                    # If the train loss is low enough, stop the training
                    if train_f1>0.999:
                        print('Train F1 score is high enough. Stopping training.')
                        recordFile.write(f"Train F1 score is high enough. Stopping training. epoch : {epoch} \n")
                        sw=False
                        break
                    
                if sw:
                    recordFile.write(f"model run finished without no problem epoch : {epochs}\n")
                # Test the model of mixed
                num_samples = 0
                num_correct = 0
                for inputs, labels in test_loader:
                    inputs=inputs.to(device)
                    labels=labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    num_samples += labels.size(0)
                    num_correct += (preds == labels).sum().item()
                test_f1 = f1_score(all_labels, all_preds, average='macro')
                acc = num_correct / num_samples
                recordFile.write(f"Mixed = Test F1: {test_f1}, Test Accuracy: {acc}\n")

                # Test the model of image
                num_samples = 0
                num_correct = 0
                for inputs, labels in test_loader_image:
                    inputs=inputs.to(device)
                    labels=labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    num_samples += labels.size(0)
                    num_correct += (preds == labels).sum().item()
                test_f1 = f1_score(all_labels, all_preds, average='macro')
                acc = num_correct / num_samples
                recordFile.write(f"Image = Test F1: {test_f1}, Test Accuracy: {acc}\n")

                # Test the model of tactile
                num_samples = 0
                num_correct = 0
                for inputs, labels in test_loader_tactile:
                    inputs=inputs.to(device)
                    labels=labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    num_samples += labels.size(0)
                    num_correct += (preds == labels).sum().item()
                test_f1 = f1_score(all_labels, all_preds, average='macro')
                acc = num_correct / num_samples
                recordFile.write(f"Tactile = Test F1: {test_f1}, Test Accuracy: {acc}\n")


