# import the necessary packages
from poison_craft import craft_random_lflip, craft_clabel_poisons
from tqdm.notebook import tqdm
from torchvision import datasets, transforms, models
from torchmetrics import Accuracy
import argparse
import torch
from torch.autograd import Variable
import torchplot as plt

import os
import numpy as np
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")


"""

# implement task 1 here
parser = argparse.ArgumentParser()
parser.add_argument('--poisonpct', default=0.1, type=float, help="percentage of poisoned samples")
args = parser.parse_args()


def generate_poisoned_dataset(batch_size = 32, n_iters = 10000, inputpctratio = 0.1):
    # get the MNIST dataset
    mnist_dataset_full = datasets.MNIST("./datasets/", train = True, download = True, transform = transforms.ToTensor())
    mnist_test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True, transform = transforms.ToTensor())

    # Selecting classes 1 and 7 for the training and testing sets
    idx = (mnist_dataset_full.targets==1) | (mnist_dataset_full.targets==7)
    mnist_dataset_full.targets = mnist_dataset_full.targets[idx]
    mnist_dataset_full.data = mnist_dataset_full.data[idx]

    idx2 = (mnist_test_dataset.targets==1) | (mnist_test_dataset.targets==7)
    mnist_test_dataset.targets = mnist_test_dataset.targets[idx2]
    mnist_test_dataset.data = mnist_test_dataset.data[idx2]

    # Setting the label for class 1 to 0 and class 7 to 1 for the training and testing sets
    mnist_dataset_full.targets[mnist_dataset_full.targets == 1] = 0
    mnist_dataset_full.targets[mnist_dataset_full.targets == 7] = 1
    mnist_test_dataset.targets[mnist_test_dataset.targets == 1] = 0
    mnist_test_dataset.targets[mnist_test_dataset.targets == 7] = 1

    # get the poisoned mnist dataset
    ratio = float(args.poisonpct) or inputpctratio
    poisoned_mnist_dataset = craft_random_lflip(mnist_dataset_full, ratio)

    # split the dataset into training, validation and testing sets
    train_size = int(0.9 * len(poisoned_mnist_dataset))
    val_size = len(poisoned_mnist_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset=poisoned_mnist_dataset, lengths=[train_size, val_size])

    epochs = n_iters / (len(train_dataset) / batch_size)

    # create the dataloader for training, validation and testing sets
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, epochs


# create the model class
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


# implement the training loop for logistic regression on the poisoned mnist dataset
def train(input_dim = 784, output_dim = 10, lr_rate = 0.001, poison_ratio=0.1):

    # initialize the model, the loss function and the optimizer
    model = LogisticRegression(input_dim, output_dim)
    criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

    # get the training, validation, testing dataloaders and epochs
    train_dataloader, val_dataloader, test_dataloader, epochs = generate_poisoned_dataset(inputpctratio = poison_ratio)

    iter = 0
    for epoch in tqdm(range(int(epochs))):
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter+=1
            if iter%500==0:
                # calculate Accuracy
                correct = 0
                total = 0
                for images, labels in val_dataloader:
                    images = Variable(images.view(-1, 28*28))
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total+= labels.size(0)
                    # for gpu, bring the predicted and labels back to cpu for python operations to work
                    correct+= (predicted == labels).sum()
                accuracy = 100 * correct/total
                print("Iteration: {}. Train Loss: {}. Val. Accuracy: {}.".format(iter, loss.item(), accuracy))
    return model, test_dataloader
        

# implement the testing loop for logistic regression on the poisoned mnist dataset
def test(poison_ratio=0.1):
    # get the trained model and the testing dataloader
    model, test_dataloader = train(poison_ratio=poison_ratio)

    # calculate Accuracy
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images = Variable(images.view(-1, 28*28))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total+= labels.size(0)
        # for gpu, bring the predicted and labels back to cpu for python operations to work
        correct+= (predicted == labels).sum()
    accuracy = 100 * correct/total
    print("Test Accuracy: {} for poisoned ratio: {}.".format(accuracy, poison_ratio))
    return accuracy

"""


# implement task 2 here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
batch_size = 16 # batch size for training

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])

transforms_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = './cifar10-raw-images'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transforms_val)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transforms_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class_names = train_dataset.classes

# define the function to plot the images
def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()

# Step 1: Choose 5 frog images (targets) from the CIFAR-10's test-set.
classes = test_dataset.classes
frog_indices = [idx for idx in range(len(test_dataset)) if test_dataset.targets[idx] == classes.index('frog')]
frog_targets = torch.stack([test_dataset[idx][0] for idx in frog_indices[:5]])

# Step 2: Choose 100 dog images (base images) from the CIFAR-10's test-set. (You will use them to craft poisons).
dog_indices = [idx for idx in range(len(test_dataset)) if test_dataset.targets[idx] == classes.index('dog')]
np.random.seed(0)
dog_base_indices = np.random.choice(dog_indices, size=100, replace=False)
dog_bases = torch.stack([test_dataset[idx][0] for idx in dog_base_indices])


# define the ResNet18 model for the feature extractor
class StudentNetwork(nn.Module):
    def __init__(self):
        super(StudentNetwork, self).__init__()

        # load a pre-trained model for the feature extractor
        self.feature_extractor = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1]).eval()
        self.fc = nn.Linear(512, 2) # binary classification (num_of_class == 2)

        # fix the pre-trained network
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.feature_extractor(images)
        x = torch.flatten(features, 1)
        outputs = self.fc(x)
        return features, outputs

# initialize the model, loss function and optimizer
model = StudentNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.01)


# generate the base and target instances
base_instance = None
target_instance = None

for inputs, labels in test_dataloader:
    for i in range(inputs.shape[0]):
        if labels[i].item() == 0: # if it's a dog
            base_instance = inputs[i].unsqueeze(0).to(device)
        elif labels[i].item() == 1: # if it's a frog
            target_instance = inputs[i].unsqueeze(0).to(device)


# Generate 100 poisons for each target image using the 100 dog images as the base images.
poisoned_base_instances_1 = []
target_instance_1 = frog_targets[0].unsqueeze(0).to(device) # change the index for each target image
for item in tqdm(dog_bases):
    base_instance = item.unsqueeze(0).to(device)
    poison = craft_clabel_poisons(base_instance, target_instance_1, model)
    poisoned_base_instances_1.append(poison)


# copy poisoned_base_instances_1 to another variable
poisoned_base_instances = poisoned_base_instances_1.copy()

# the training and validation loop
num_epochs = 10
start_time = time.time()
poison_amt = 30

for epoch in range(num_epochs):
    """ Training Phase """
    running_loss = 0.
    running_corrects = 0

    # load a batch data of images
    for i, (inputs, labels) in enumerate(train_dataloader):
        
        if poison_amt > 0:
            if i == 0 and batch_size >= poison_amt:
                # change the first batch of images to the poisoned instances
                for index in range(len(poisoned_base_instances[:poison_amt])):
                    inputs[index] = poisoned_base_instances[index]
                    labels[index] = torch.tensor([0])
                    
                poison_amt -= len(poisoned_base_instances[:poison_amt])
            
            elif i > 0 and poison_amt > 0:
                # change the next batches of images to the poisoned instances
                for index in range(min(10, poison_amt)):
                    inputs[index] = poisoned_base_instances[index]
                    labels[index] = torch.tensor([0])
                    
                poison_amt -= 10

        labels = labels.type(torch.LongTensor)

        inputs = inputs.to(device)
        labels = labels.to(device)
            
        # forward inputs and get output
        optimizer.zero_grad()

        features, outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # get loss value and update the network weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

    """ Validation Phase """
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0

        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            features, outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects / len(val_dataset) * 100.
        print('[Validation #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

    if (epoch == 0) or epoch % 5 == 0:
        """ Poisoning Attack Test Phase """
        with torch.no_grad():
            _, outputs = model(target_instance)
            _, preds = torch.max(outputs, 1)

            imshow(target_instance[0].cpu(), f'Target Instance (predicted class name: {class_names[preds.item()]})')
            percentages = nn.Softmax(dim=1)(outputs)[0]
            print(f'[Predicted Confidence] {class_names[0]}: {percentages[0]} | {class_names[1]}: {percentages[1]}')

