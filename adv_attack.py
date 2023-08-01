# import the necessary packages
from models import lenet_mnist, resnet_cifar10, resnet_dropout
from tqdm.notebook import tqdm
import argparse
from pathlib import Path
from torchmetrics import Accuracy
import torch.nn as nn
import torch
import torchplot as plt
from torchvision import datasets, transforms
import time

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./datasets/', train=False, download=True, 
                                          transform=transforms.Compose([transforms.ToTensor(),])), 
                                          batch_size=1, shuffle=True)

# CIFAR10 Test dataset and dataloader declaration
norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
test_transforms = transforms.Compose([transforms.ToTensor(), norm, transforms.RandomRotation(30),])
test_loader_cifar10 = torch.utils.data.DataLoader(datasets.CIFAR10('./datasets/', train=False, download=True,
                                            transform=test_transforms), batch_size=1, shuffle=True) 

# initialize the network and loss function for the LeNet MNIST model
device = 'cuda' if torch.cuda.is_available() else 'cpu' # device-agnostic setup
lenet_model = lenet_mnist.LeNet5().to(device)
MODEL_SAVE_PATH = Path("saved_models/lenet_mnist.pth")
lenet_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
loss_fn = nn.CrossEntropyLoss()

# initialize the network for the ResNet CIFAR10 model
resnet_model = resnet_cifar10.ResNet18().to(device)
MODEL_SAVE_PATH = Path("saved_models/resnet_cifar10.pth")
resnet_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# initialize the network for the ResNet dropout CIFAR10 model
resnet_dropout_model = resnet_dropout.ResNet18().to(device)
MODEL_SAVE_PATH = Path("saved_models/resnet_dropout.pth")
resnet_dropout_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# initialize the network for the ResNet CIFAR10 model with weight decays
resnet_cifar10_wd_model = resnet_cifar10.ResNet18().to(device)
MODEL_SAVE_PATH = Path("saved_models/resnet_cifar10_wd.pth")
resnet_cifar10_wd_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# initialize the network for the adversarially trained LesNet MNIST model
lenet_model_adv = lenet_mnist.LeNet5().to(device)
MODEL_SAVE_PATH = Path("adv_saved_models/lenet_mnist.pth")
lenet_model_adv.load_state_dict(torch.load(MODEL_SAVE_PATH))

# initialize the network for the adversarially trained ResNet CIFAR10 model
resnet_model_adv = resnet_cifar10.ResNet18().to(device)
MODEL_SAVE_PATH = Path("adv_saved_models/resnet_cifar10.pth")
resnet_model_adv.load_state_dict(torch.load(MODEL_SAVE_PATH))

# set the model to evaluation mode
lenet_model.eval()
lenet_model_adv.eval()
resnet_model.train()
resnet_dropout_model.train()
resnet_cifar10_wd_model.train()
resnet_model_adv.train()

# implement the PGD attack
def PGD(model, x, y, niter, step_size, eps, y_target=None, random_start=False):
    """Performs the projected gradient descent attack on a batch of images."""
    if random_start:
        x_adv = x.clone().detach().requires_grad_(True) + torch.zeros_like(x).uniform_(-eps, eps)
        x_adv = x_adv.to(device)
    else:
        x_adv = x.clone().detach().requires_grad_(True).to(device)

    num_channels = x.shape[1]

    for i in range(niter):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        pred = model(_x_adv)
        loss = loss_fn(pred, y)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            gradients = _x_adv.grad.sign() * step_size
            x_adv += gradients # Untargeted PGD
            
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps) # Project shift to L_inf norm ball
        x_adv = torch.clamp(x_adv,0,1) #set output to correct range

    return x_adv.detach()



# evaluate the model on the pertubed samples
def test(model_name, model, test_loader,step_size, eps):
    correct = 0
    adv_examples = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)  # Forward pass 
        init_pred = output.max(1, keepdim=True)[1] # get the initial prediction
        
        if init_pred.item() != label.item():  # Only bother to attck if initial prediction is correct
            continue

        perturbed_data = PGD(model,data,label,7,step_size,eps)
        
        output = model(perturbed_data) # Re-classify the perturbed image
        final_pred = output.max(1, keepdim=True)[1] # get the new prediction

        if final_pred.item() == label.item(): # Check for success
            correct += 1

            if (eps == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().cpu().detach().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().cpu().detach().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Model: {model_name} | Epsilon: {eps} | Test Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial examples
    return final_acc, adv_examples


if __name__ == "__main__":
    epsilons = [[0.3,], [0.03]]
    # epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

    step_size = 2/255
    accuracies_lenet = []
    accuracies_resnet = []
    examples_lenet = []
    examples_resnet = []

    # Run test for each epsilon
    start = time.time()
    for index in tqdm(range(len(epsilons))):
        eps = epsilons[index] # uncomment this line to run the test for multiple epsilons on the two DNNs
        # if index == 0:
        #     model_name = "LeNet_MNIST"
        #     eps = epsilons[index][0]
        #     acc, ex = test(model_name, lenet_model_adv, test_loader, step_size, eps)
        #     accuracies_lenet.append(acc)
        #     examples_lenet.append(ex)

        if index == 1:
            model_name = "ResNet_CIFAR10"
            eps = epsilons[index][0]
            acc, ex = test(model_name, resnet_model_adv, test_loader_cifar10, step_size, eps)
            accuracies_resnet.append(acc)
            examples_resnet.append(ex)

        

    # print the classification accuracy for each epsilon for the two DNNs
    """
    print("LeNet MNIST")
    for i in range(len(epsilons)):
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilons[i], int(accuracies_lenet[i]), len(test_loader), accuracies_lenet[i]))
    print("ResNet CIFAR10")
    for i in range(len(epsilons)):
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilons[i], int(accuracies_resnet[i]), len(test_loader_cifar10), accuracies_resnet[i]))
    """

    print(f"Time taken: {time.time() - start:.0f} seconds")
