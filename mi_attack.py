import torch
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
from torchvision import transforms, datasets
from models import resnet_cifar10, vgg_fmnist
from pathlib import Path
import os
from glob import glob
import torchplot as plt
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x

def get_loss_n_accuracy(model, data_loader, device='cuda:0', num_classes=10):
    """ Returns loss/acc, and per-class loss/accuracy on supplied data loader """
    
    with torch.inference_mode():
        # disable BN stats during inference
        model.eval()
        total_loss, correctly_labeled_samples = 0, 0
        confusion_matrix = torch.zeros(num_classes, num_classes)
        per_class_loss = torch.zeros(num_classes, device=device)
        per_class_ctr = torch.zeros(num_classes, device=device)
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device=device, non_blocking=True),\
                    labels.to(device=device, non_blocking=True)
                                                        
            outputs = model(inputs)
            losses = criterion(outputs, labels)
            # keep track of total loss
            total_loss += losses.sum()
            # get num of correctly predicted inputs in the current batch
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
            
            # per-class acc (filling confusion matrix)
            for t, p in zip(labels.view(-1), pred_labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            # per-class loss
            for i in range(num_classes):
                filt = labels == i
                per_class_loss[i] += losses[filt].sum()
                per_class_ctr[i] += filt.sum()
            
        loss = total_loss/len(data_loader.dataset)
        accuracy = correctly_labeled_samples / len(data_loader.dataset)
        per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
        per_class_loss = per_class_loss/per_class_ctr
        
        return (loss, accuracy), (per_class_accuracy, per_class_loss)
    

def mia_by_threshold(model, tr_loader, te_loader, threshold, device='cuda:0', n_classes=10):
    with torch.inference_mode():
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        model.eval()
        tp, fp = torch.zeros(n_classes, device=device), torch.zeros(n_classes, device=device)
        tn, fn = torch.zeros(n_classes, device=device), torch.zeros(n_classes, device=device)
        
        # on training loader (members, i.e., positive class)
        for _, (inputs, labels) in enumerate(tr_loader):
            inputs, labels = inputs.to(device=device, non_blocking=True), labels.to(device=device, non_blocking=True)

            outputs = model(inputs)
            losses = criterion(outputs, labels)
            # with global threshold
            predictions = losses < threshold
            # class-wise confusion matrix values
            for i in range(n_classes):
                preds = predictions[labels == i]
                n_member_pred = preds.sum()
                tp[i] += n_member_pred
                fn[i] += len(preds) - n_member_pred
        
        # on test loader (non-members, i.e., negative class)
        for _, (inputs, labels) in enumerate(te_loader):
            inputs, labels = inputs.to(device=device, non_blocking=True), labels.to(device=device, non_blocking=True)
            outputs = model(inputs)
            losses = criterion(outputs, labels)
            # with global threshold
            predictions = losses < threshold
            # class-wise confusion matrix values
            for i in range(n_classes):
                preds = predictions[labels == i]
                n_member_pred = preds.sum()
                fp[i] += n_member_pred
                tn[i] += len(preds) - n_member_pred
        
        # class-wise bacc, tpr, fpr computations
        class_tpr, class_fpr = torch.zeros(n_classes, device=device), torch.zeros(n_classes, device=device)
        class_bacc = torch.zeros(n_classes, device=device)
        membership_advantage = torch.zeros(n_classes, device=device)  # Initialize membership advantage tensor
        for i in range(n_classes):
            class_i_tpr, class_i_tnr = tp[i] / (tp[i] + fn[i]), tn[i] / (tn[i] + fp[i])
            class_tpr[i], class_fpr[i] = class_i_tpr, 1 - class_i_tnr
            class_bacc[i] = (class_i_tpr + class_i_tnr) / 2
            
            # Compute membership advantage
            membership_advantage[i] = class_i_tpr - class_i_tnr
        
        # dataset-wise bacc, tpr, fpr computations
        ds_tp, ds_fp = tp.sum(), fp.sum()
        ds_tn, ds_fn = tn.sum(), fn.sum()
        ds_tpr, ds_tnr = ds_tp / (ds_tp + ds_fn), ds_tn / (ds_tn + ds_fp)
        ds_bacc, ds_fpr = (ds_tpr + ds_tnr) / 2, 1 - ds_tnr

        # Compute membership advantage for the entire dataset
        dataset_membership_advantage = membership_advantage.mean()
        
        return dataset_membership_advantage

def load_resnet_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(num_classes=10).to(device)  # Replace DEVICE with your target device
    saved_state_dict = torch.load(model_path)

    # Check for missing and unexpected keys
    missing_keys = []
    unexpected_keys = []
    for key in saved_state_dict.keys():
        if key not in model.state_dict():
            unexpected_keys.append(key)
    for key in model.state_dict().keys():
        if key not in saved_state_dict:
            missing_keys.append(key)

    # Fix the missing and unexpected keys
    for missing_key in missing_keys:
        saved_state_dict[missing_key] = model.state_dict()[missing_key]
    for unexpected_key in unexpected_keys:
        del saved_state_dict[unexpected_key]

    # Load the modified state dictionary
    model.load_state_dict(saved_state_dict)

    return model

def load_vgg_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)  # Replace DEVICE with your target device
    saved_state_dict = torch.load(model_path)

    # Check for missing and unexpected keys
    missing_keys = []
    unexpected_keys = []
    for key in saved_state_dict.keys():
        if key not in model.state_dict():
            unexpected_keys.append(key)
    for key in model.state_dict().keys():
        if key not in saved_state_dict:
            missing_keys.append(key)

    # Fix the missing and unexpected keys
    for missing_key in missing_keys:
        saved_state_dict[missing_key] = model.state_dict()[missing_key]
    for unexpected_key in unexpected_keys:
        del saved_state_dict[unexpected_key]

    # Load the modified state dictionary
    model.load_state_dict(saved_state_dict)

    return model


"""
# membership advantage computation for ResNet28 with CIFAR-10 dataset
# define the data transforms
norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor(), norm,])
test_transforms = transforms.Compose([transforms.ToTensor(), norm,])

# download the CIFAR_10 dataset from torchvision API to the local directory
train_val_dataset = datasets.CIFAR10(root="./datasets/", train=True, download=False, transform=train_transforms)
test_dataset = datasets.CIFAR10(root="./datasets/", train=False, download=False, transform=test_transforms)

# split the dataset into training, validation and testing sets
train_size = 5000
val_size = len(train_val_dataset) - train_size
test_size = 5000
train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])
test_dataset, _ = torch.utils.data.random_split(dataset=test_dataset, lengths=[test_size, len(test_dataset)-test_size])

# create the dataloader for training, validation and testing sets
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# loop through the directory and pick each model
models_directory = Path("saved_dp_models")
membership_advantages = []
epochs_list = [10, 20, 30, 40, 50, 60, 70, 80]
eps = [1, 2, 4, 8, 10]
for model_path in glob(os.path.join(models_directory, "*.pth")):
    # resnet_model = resnet_cifar10.ResNet18().to(DEVICE) # uncomment for the non-dp trained models

    resnet_model = load_resnet_model(model_path) # uncomment for the dp-trained models
    print("Loaded model:", model_path)  

    with torch.inference_mode():
        # metrics after training
        (tr_loss, tr_acc), (tr_per_class_acc, tr_per_class_loss) = get_loss_n_accuracy(resnet_model, val_dataloader, device=DEVICE)

    # get the membership advantage for the model
    membership_advantage = mia_by_threshold(resnet_model, train_dataloader, test_dataloader, threshold=tr_loss, device=DEVICE)
    membership_advantages.append(membership_advantage)

# plot the # epsilon vs. membership advantage for all models
dest = Path("figures")
dest.mkdir(parents=True, exist_ok=True)
dest = dest / "dp_resnet_memberships_plot.png"
plt.plot(eps, membership_advantages, label='Membership Advantage')
plt.xlabel('# epsilson')
plt.ylabel('Membership Advantage')
plt.title('Membership Advantage vs. # epsilon for ResNet-18 models')
plt.legend()
plt.savefig(dest)
plt.show()
"""

# membership advantage computation for VGG16 with FashionMNIST dataset
# define the data transforms
transform = transforms.Compose([transforms.Resize((224,224)), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))])

# download the CIFAR_10 dataset from torchvision API to the local directory
train_val_dataset = datasets.FashionMNIST(root="./datasets/", train=True, download=False, transform=transform)
test_dataset = datasets.FashionMNIST(root="./datasets/", train=False, download=False, transform=transform)

# split the dataset into training, validation and testing sets
train_size = 5000
val_size = len(train_val_dataset) - train_size
test_size = 5000
train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])
test_dataset, _ = torch.utils.data.random_split(dataset=test_dataset, lengths=[test_size, len(test_dataset)-test_size])

# create the dataloader for training, validation and testing sets
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# loop through the directory and pick each model
models_directory = Path("saved_dp_models_2")
membership_advantages = []
eps = [1, 2, 4, 8, 10]
epochs_list = [10, 20, 30, 40, 50, 60, 70, 80]
for model_path in glob(os.path.join(models_directory, "*.pth")):
    # vggmodel = vgg_fmnist.VGG16().to(DEVICE) # uncomment for the non-dp trained models
    vggmodel = load_vgg_model(model_path) # uncomment for the dp-trained models
    print("Loaded model:", model_path)

    with torch.inference_mode():
        # metrics after training
        (tr_loss, tr_acc), (tr_per_class_acc, tr_per_class_loss) = get_loss_n_accuracy(vggmodel, val_dataloader, device=DEVICE)

    # get the membership advantage for the model
    membership_advantage = mia_by_threshold(vggmodel, train_dataloader, test_dataloader, threshold=tr_loss, device=DEVICE)
    membership_advantages.append(membership_advantage)


# plot the # epsilon vs. membership advantage for all models
dest = Path("figures")
dest.mkdir(parents=True, exist_ok=True)
dest = dest / "dp_vgg16_memberships_plot.png"
plt.plot(eps, membership_advantages, label='Membership Advantage')
plt.xlabel('# epislon')
plt.ylabel('Membership Advantage')
plt.title('Membership Advantage vs. # epsilon for VGG16 models')
plt.legend()
plt.savefig(dest)
plt.show()
