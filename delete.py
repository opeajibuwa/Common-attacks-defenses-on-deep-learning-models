# import the necessary packages
from poison_craft import craft_random_lflip, craft_clabel_poisons
from tqdm.notebook import tqdm
from torchvision import datasets, transforms, models
from torchmetrics import Accuracy
import argparse
from torch.autograd import Variable
import torchplot as plt
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
warnings.filterwarnings("ignore")


""" implement task 1 here """
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


if __name__ == "__main__":
    # test the model for different poison ratios
    poison_ratios = [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    test_accuracies = []
    for poison_ratio in poison_ratios:
        test_accuracies.append(test(poison_ratio))
    plt.plot(poison_ratios, test_accuracies)
    plt.xlabel("Poison Ratio")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Poison Ratio")
    plt.show()
    plt.savefig("figures/test_accuracy_vs_poison_ratio.png")

