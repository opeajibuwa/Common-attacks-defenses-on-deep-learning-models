""" this file contains functions to load the datasets for training/testing the models"""

# import the libraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def compose_tranforms(model: str):
    """ this function composes the transforms for the datasets"""

    # define the transforms/augmentation for the MNIST dataset-based models
    if model == "lenet_mnist":

        train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=False, transform=transforms.ToTensor())
        imgs = torch.stack([img_t for img_t, _ in train_val_dataset], dim=0)
        mean = imgs.mean()
        std = imgs.std()
        mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return mnist_transform, mnist_transform

    if model == "resnet_mnist":
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        return train_transform, test_transform
    
    if model == "vgg_mnist":
        train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
                                               transforms.Normalize(mean=(0.5), std=(0.5))])

        test_transforms = transforms.Compose([transforms.Resize((224, 224)), 
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.5), std=(0.5))])
        return train_transforms, test_transforms
    
    # define the transforms augmentation for the CIFAR10 dataset-based models
    if model == "lenet_cifar10":
        train_transform = transforms.Compose([transforms.RandomGrayscale(0.2),
                                            transforms.RandomVerticalFlip(0.2),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomAdjustSharpness(0.4),
                                            transforms.RandomRotation(30),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ])
        
        test_transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        return train_transform, test_transform

    if model == "resnet_cifar10":
        norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                               transforms.ToTensor(), norm,])
        test_transforms = transforms.Compose([transforms.ToTensor(), norm,])
        return train_transforms, test_transforms

    if model == "resnet_dropout":
        norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                               transforms.ToTensor(), norm,])
        test_transforms = transforms.Compose([transforms.ToTensor(), norm,])
        return train_transforms, test_transforms

    if model == "vgg_cifar10":
        train_transforms = transforms.Compose([transforms.Resize((70, 70)), 
                                               transforms.RandomCrop((64, 64)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        test_transforms = transforms.Compose([transforms.Resize((70, 70)), 
                                              transforms.CenterCrop((64, 64)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return train_transforms, test_transforms


def load_mnist_dataset(model: str, BATCH_SIZE: int):
    """ this function loads the MNIST dataset and returns the train, validation and test dataloaders"""

    # define the transforms
    if model == "lenet_mnist":
        train_transform, test_transform = compose_tranforms(model="lenet_mnist")
    if model == "resnet_mnist":
        train_transform, test_transform = compose_tranforms(model="resnet_mnist")
    if model == "vgg_mnist":
        train_transform, test_transform = compose_tranforms(model="vgg_mnist")
    

    # load the dataset
    train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True, transform=test_transform)

    # split the dataset into training, validation and testing sets
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])

    # create the dataloader for training, validation and testing sets
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def load_cifar10_dataset(model: str, BATCH_SIZE: int):
    """ this function loads the CIFAR10 dataset and returns the train, validation and test dataloaders"""

    # define the transforms
    if model == "lenet_cifar10":
        train_transform, test_transform = compose_tranforms(model="lenet_cifar10")
    if model == "resnet_cifar10":
        train_transform, test_transform = compose_tranforms(model="resnet_cifar10")
    if model == "vgg_cifar10":
        train_transform, test_transform = compose_tranforms(model="vgg_cifar10")
    if model == "resnet_dropout":
        train_transform, test_transform = compose_tranforms(model="resnet_dropout")

    
    # load the dataset
    train_val_dataset = datasets.CIFAR10(root="./datasets/", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./datasets/", train=False, download=True, transform=test_transform)

    # split the dataset into training, validation and testing sets
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])

    # create the dataloader for training, validation and testing sets
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader