import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import transforms, datasets
from torchvision.utils import save_image
import torch.optim as optim
from pathlib import Path
from tqdm.notebook import tqdm
from torchmetrics import Accuracy



"""""" # define the VGG16 model
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


# define the training and validation data loaders
transform = transforms.Compose([transforms.Resize((224,224)), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.FashionMNIST('./datasets/', download=True, train=True, transform=transform)

val_size = 10000
train_size = len(train_set) - val_size
train_set, val_set = random_split(train_set, [train_size, val_size])
test_set = datasets.FashionMNIST('./datasets/', download=True, train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

# define the optimizer, loss function and learning rate
epoch = 80
learning_rate = 1e-4
load_model = True
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
model = VGG16().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

# train the model
train_losses = []
val_losses = []
epochs_list = []

for epochs in tqdm(range(epoch)): #I decided to train the model for 50 epochs
    loss_ep = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()
    
    print(f"Epoch {epochs+1} ] Loss : {loss_ep/len(train_loader):.6f}")
    
    if (epochs + 1) in [10, 20, 30, 40, 50, 60, 70, 80]:
        train_losses.append(loss_ep / len(train_loader))
        
        with torch.no_grad():
            val_loss_ep = 0
            for batch_idx, (data, targets) in enumerate(val_loader):
                data = data.to(DEVICE)
                targets = targets.to(DEVICE)
                scores = model(data)
                val_loss = criterion(scores, targets)
                val_loss_ep += val_loss.item()
            val_losses.append(val_loss_ep / len(val_loader))
        
        epochs_list.append(epochs + 1)
        
        # Save the model at the desired epoch
        torch.save(model.state_dict(), f"model_epoch_{epochs+1}.pth")
    
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data, targets) in enumerate(val_loader):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            ## Forward Pass
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        print(f"Epoch {epochs+1} ] Accuracy : {float(num_correct) / float(num_samples) * 100:.2f}%")
        

# plot the training and validation losses
dest = Path("figures")
dest.mkdir(parents=True, exist_ok=True)
dest1 = dest / "vgg16_fmnist_wt_diff_batches.png"
plt.plot(epochs_list, train_losses, label='Train Loss')
plt.plot(epochs_list, val_losses, label='Val Loss')
plt.xlabel('# Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(dest1)
plt.show()


""""""# define the resnet-cifar10 model
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        DROPOUT = 0.1

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# define the transforms to be applied to the images
norm = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor(), norm,])
test_transforms = transforms.Compose([transforms.ToTensor(), norm,])

# download the CIFAR_10 dataset from torchvision API to the local directory
train_val_dataset = datasets.CIFAR10(root="./datasets/", train=True, download=False, transform=train_transforms)
test_dataset = datasets.CIFAR10(root="./datasets/", train=False, download=False, transform=test_transforms)

# split the dataset into training, validation and testing sets
train_size = int(0.9 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])

# create the dataloader for training, validation and testing sets
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# initialize the model, loss function, optimizer and evaluation metrics
model_resnet = ResNet18()
# optimizer = torch.optim.SGD(model_resnet.parameters(), momentum=0.9, lr=0.01)
optimizer = torch.optim.Adam(model_resnet.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
accuracy = Accuracy(task="multiclass", num_classes=10)

# define the optimizer, loss function and learning rate
epoch = 90
device = 'cuda' if torch.cuda.is_available() else 'cpu' # device-agnostic setup
accuracy = accuracy.to(device)
model_resnet = model_resnet.to(device)

# train the model
train_losses = []
val_losses = []
epochs_list = []

for epochs in range(epoch):
    loss_ep = 0
    
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data = data.to(device)
        targets = targets.to(device)
        ## Forward Pass
        optimizer.zero_grad()
        scores = model_resnet(data)
        loss = loss_fn(scores, targets)
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()
    
    print(f"Epoch {epochs+1} ] Loss : {loss_ep/len(train_dataloader):.6f}")
    
    if (epochs + 1) in [10, 20, 30, 40, 50, 60, 70, 80]:
        train_losses.append(loss_ep / len(train_dataloader))
        
        with torch.no_grad():
            val_loss_ep = 0
            for batch_idx, (data, targets) in enumerate(val_dataloader):
                data = data.to(device)
                targets = targets.to(device)
                scores = model_resnet(data)
                val_loss = loss_fn(scores, targets)
                val_loss_ep += val_loss.item()
            val_losses.append(val_loss_ep / len(val_dataloader))
        
        epochs_list.append(epochs + 1)
        
        # Save the model at the desired epoch
        torch.save(model_resnet.state_dict(), f"model_epoch_{epochs+1}.pth")
    
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data, targets) in enumerate(val_dataloader):
            data = data.to(device)
            targets = targets.to(device)
            ## Forward Pass
            scores = model_resnet(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        print(f"Epoch {epochs+1} ] Accuracy : {float(num_correct) / float(num_samples) * 100:.2f}%")


# plot the training and validation losses
dest = Path("figures")
dest.mkdir(parents=True, exist_ok=True)
dest1 = dest / "resnet_cifar10_wt_diff_batches.png"
plt.plot(epochs_list, train_losses, label='Train Loss')
plt.plot(epochs_list, val_losses, label='Val Loss')
plt.xlabel('# Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(dest1)
plt.show()


