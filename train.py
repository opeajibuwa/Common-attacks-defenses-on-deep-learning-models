"""This script trains and validates different models"""

# import the necessary packages
from models import lenet_mnist, lenet_cifar10, resnet_cifar10, resnet_mnist, vgg_cifar10, vgg_mnist, resnet_dropout
from datasets import load_mnist_dataset, load_cifar10_dataset
from tqdm.notebook import tqdm
import argparse
from pathlib import Path
from torchmetrics import Accuracy
import torch.nn as nn
import torch
import torchplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='lenet_mnist',
                    help="model to train and validate. One of [lenet_mnist, lenet_cifar10, ...]")
parser.add_argument('--batchsz', default=32,
                    help="batch size for training and validation")
parser.add_argument('--lr', default=0.01,
                    help="the learning rate")
parser.add_argument('--epochs', default=50,
                    help="no of training iterations")
parser.add_argument('--optim', default="sgd",
                    help="the optimizer to use for training")
parser.add_argument('--wd', default=0.0, help="weight decay")



def train_and_validate(model, model_name, train_dataloader, val_dataloader, loss_fn, optimizer, epochs, device, accuracy):
    """Train and validate a model on the training data and evaluate it on the test data.

    Args:
        model (nn.Module): The model to train and validate.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        loss_fn (Loss): The loss function.
        optimizer (Optimizer): The optimizer.
        epochs (int): The number of epochs to train the model.
        device (torch.device): The device to use for training and validation.

    Returns:
        list of the training loss, validation loss, training accuracy, and validation accuracy.
    """

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in tqdm(range(epochs)):
        # Training loop
        train_loss, train_acc = 0.0, 0.0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            
            model.train()
            
            if model_name == 'lenet_cifar10' or model_name == 'resnet_mnist':
                y_pred, probas = model(X)
            else: y_pred = model(X)
            
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            
            acc = accuracy(y_pred, y)
            train_acc += acc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        
        if epoch % 5 == 0:
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
        
        # Validation loop
        val_loss, val_acc = 0.0, 0.0
        model.eval()
        with torch.inference_mode():
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                
                if model_name == 'lenet_cifar10' or model_name == 'resnet_mnist':
                    y_pred, probas = model(X)
                else: y_pred = model(X)
                
                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                
                acc = accuracy(y_pred, y)
                val_acc += acc
                
            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)

            if epoch % 5 == 0:
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)

        print(f" Model: {model_name} | Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}")

    # save the trained model
    MODEL_PATH = Path("saved_models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = f"{model_name}.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    print(f"Saving the model: {MODEL_SAVE_PATH}") # saving the model
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list
    

def plot_loss_and_accuracy(train_loss_list, val_loss_list, train_acc_list, val_acc_list, model_name):
    """Plot the training, validation loss and accuracy on the same plot.

    Args:
        train_loss_list (list): The training loss.
        val_loss_list (list): The validation loss.
        train_acc_list (list): The training accuracy.
        val_acc_list (list): The validation accuracy.
    """

    epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    dest = Path("figures")
    dest.mkdir(parents=True, exist_ok=True)
    dest1 = dest / f"{model_name}_loss_and_accuracy.png"
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train loss")
    plt.plot(epochs, val_loss_list, label="Val loss")
    plt.title("training loss and validation loss vs epochs")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label="Train acc")
    plt.plot(epochs, val_acc_list, label="Val acc")
    plt.title("training accuracy and validation accuracy vs epochs")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(dest1)
    return plt.show()


if __name__ == '__main__':

    # general initializations and setup
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # device-agnostic setup
    accuracy = Accuracy(task="multiclass", num_classes=10)
    accuracy = accuracy.to(device)
    loss_fn = nn.CrossEntropyLoss()

    if args.model == "lenet_mnist":

        # load the training and validation data
        train_dataloader, val_dataloader, _ = load_mnist_dataset("lenet_mnist", int(args.batchsz))

        # initialize the model, loss function, optimizer and evaluation metrics
        model = lenet_mnist.LeNet5()
        optimizer = torch.optim.SGD(model.parameters(), float(args.lr))
       
        # train the model
        model_lenet5v1 = model.to(device)

        # call the train and validate function
        train_llt, val_llt, train_acl, val_acl = train_and_validate(model_lenet5v1, args.model, train_dataloader, 
                                                                    val_dataloader, loss_fn, optimizer, int(args.epochs), device, accuracy)

        # plot the training, validation loss and accuracy on the same plot
        plot_loss_and_accuracy(train_llt, val_llt, train_acl, val_acl, args.model)

    
    if args.model == "lenet_cifar10":

        # load the training and validation data
        train_dataloader, val_dataloader, _ = load_cifar10_dataset("lenet_cifar10", int(args.batchsz))

        # initialize the model, loss function, optimizer and evaluation metrics
        model = lenet_cifar10.LeNet5(num_classes=10)
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), float(args.lr), momentum=0.9)
        if args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), float(args.lr))

        # train the model
        model_lenet5v1 = model.to(device)

        # call the train and validate function 
        train_llt, val_llt, train_acl, val_acl = train_and_validate(model_lenet5v1, args.model, train_dataloader,
                                                                    val_dataloader, loss_fn, optimizer, int(args.epochs), device, accuracy)
        
        # plot the training, validation loss and accuracy on the same plot
        plot_loss_and_accuracy(train_llt, val_llt, train_acl, val_acl, args.model)

    if args.model == "resnet_cifar10":

        # load the training and validation data
        train_dataloader, val_dataloader, _ = load_cifar10_dataset("resnet_cifar10", int(args.batchsz))

        # initialize the model, loss function, optimizer and evaluation metrics
        model = resnet_cifar10.ResNet18()
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), float(args.lr), momentum=0.9)
        if args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), float(args.lr))

        # train the model
        model_resnet18v1 = model.to(device)

        # call the train and validate function
        train_llt, val_llt, train_acl, val_acl = train_and_validate(model_resnet18v1, args.model, train_dataloader,
                                                                    val_dataloader, loss_fn, optimizer, int(args.epochs), device, accuracy)
        
        # plot the training, validation loss and accuracy on the same plot
        plot_loss_and_accuracy(train_llt, val_llt, train_acl, val_acl, args.model)

    if args.model == "resnet_mnist":

        # load the training and validation data
        train_dataloader, val_dataloader, _ = load_mnist_dataset("resnet_mnist", int(args.batchsz))

        # initialize the model, loss function, optimizer and evaluation metrics
        model = resnet_mnist.ResNet18(num_classes=10)
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), float(args.lr), momentum=0.9)
        if args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), float(args.lr))

        # train the model
        model_resnet18v1 = model.to(device)

        # call the train and validate function
        train_llt, val_llt, train_acl, val_acl = train_and_validate(model_resnet18v1, args.model, train_dataloader,
                                                                    val_dataloader, loss_fn, optimizer, int(args.epochs), device, accuracy)
        
        # plot the training, validation loss and accuracy on the same plot
        plot_loss_and_accuracy(train_llt, val_llt, train_acl, val_acl, args.model)

    if args.model == "vgg_cifar10":

        # load the training and validation data
        train_dataloader, val_dataloader, _ = load_cifar10_dataset("vgg_cifar10", int(args.batchsz))

        # initialize the model, loss function, optimizer and evaluation metrics
        model = vgg_cifar10.VGG16(num_classes=10)
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), float(args.lr), momentum=0.9)
        if args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), float(args.lr))

        # train the model
        model_vggv1 = model.to(device)

        # call the train and validate function
        train_llt, val_llt, train_acl, val_acl = train_and_validate(model_vggv1, args.model, train_dataloader,
                                                                    val_dataloader, loss_fn, optimizer, int(args.epochs), device, accuracy)
        
        # plot the training, validation loss and accuracy on the same plot
        plot_loss_and_accuracy(train_llt, val_llt, train_acl, val_acl, args.model)

    if args.model == "vgg_mnist":

        # load the training and validation data
        train_dataloader, val_dataloader, _ = load_mnist_dataset("vgg_mnist", int(args.batchsz))

        # initialize the model, loss function, optimizer and evaluation metrics
        model = vgg_mnist.VGG16(num_classes=10)
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), float(args.lr), momentum=0.9)
        if args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), float(args.lr))
        
        # train the model
        model_vggv1 = model.to(device)
        
        # call the train and validate function
        train_llt, val_llt, train_acl, val_acl = train_and_validate(model_vggv1, args.model, train_dataloader,
                                                                    val_dataloader, loss_fn, optimizer, int(args.epochs), device, accuracy)
        
        # plot the training, validation loss and accuracy on the same plot
        plot_loss_and_accuracy(train_llt, val_llt, train_acl, val_acl, args.model)

    if args.model == "resnet_dropout":

        # load the training and validation data
        train_dataloader, val_dataloader, _ = load_cifar10_dataset("resnet_dropout", int(args.batchsz))

        # initialize the model, loss function, optimizer and evaluation metrics
        model = resnet_dropout.ResNet18()
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), float(args.lr), momentum=0.9)
        if args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), float(args.lr))

        # train the model
        model_resnet18v1 = model.to(device)

        # call the train and validate function
        train_llt, val_llt, train_acl, val_acl = train_and_validate(model_resnet18v1, args.model, train_dataloader,
                                                                    val_dataloader, loss_fn, optimizer, int(args.epochs), device, accuracy)
        
        # plot the training, validation loss and accuracy on the same plot
        plot_loss_and_accuracy(train_llt, val_llt, train_acl, val_acl, args.model)

    if args.model == "resnet_cifar10_wd":

        # load the training and validation data
        train_dataloader, val_dataloader, _ = load_cifar10_dataset("resnet_cifar10", int(args.batchsz))

        # initialize the model, loss function, optimizer and evaluation metrics
        model = resnet_cifar10.ResNet18()
        optimizer = torch.optim.Adam(model.parameters(), float(args.lr), weight_decay=float(args.wd))

        # train the model
        model_resnet18v1 = model.to(device)

        # call the train and validate function
        train_llt, val_llt, train_acl, val_acl = train_and_validate(model_resnet18v1, args.model, train_dataloader,
                                                                    val_dataloader, loss_fn, optimizer, int(args.epochs), device, accuracy)
        
        # plot the training, validation loss and accuracy on the same plot
        plot_loss_and_accuracy(train_llt, val_llt, train_acl, val_acl, args.model)




        


