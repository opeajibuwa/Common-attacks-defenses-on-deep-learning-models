""" script for evaluating pre-trained models """

# import packages
from models import lenet_mnist, lenet_cifar10, resnet_cifar10, resnet_mnist, vgg_cifar10, vgg_mnist
from datasets import load_mnist_dataset, load_cifar10_dataset
import argparse
from tqdm.notebook import tqdm
from pathlib import Path
from torchmetrics import Accuracy
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batchsz', default=32,
                    help="batch size for training")
parser.add_argument('--model', default='lenet_mnist',
                    help="model to train")

    
def load_test_saved_model(model_instance, MODEL_SAVE_PATH, test_dataloader, device, loss_fn, accuracy, model_name):
    """Test a saved model on the test data."""
    model_instance.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # test the model
    test_loss, test_acc = 0, 0
    model_instance.to(device)

    model_instance.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            if args.model == "lenet_cifar10" or args.model == 'resnet_mnist':
                y_pred, probas = model_instance(X)
            else: y_pred = model_instance(X)

            test_loss += loss_fn(y_pred, y)
            test_acc += accuracy(y_pred, y)

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f" Model: {model_name} |Test loss: {test_loss: .5f}| Test acc: {test_acc: .5f}")


if __name__ == '__main__':

    # general initializations
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # device-agnostic setup
    accuracy = Accuracy(task="multiclass", num_classes=10)
    accuracy = accuracy.to(device)
    loss_fn = nn.CrossEntropyLoss()

    if args.model == "lenet_cifar10":
        # load the test data
        _, _, test_dataloader = load_cifar10_dataset("lenet_cifar10", int(args.batchsz))
        model = lenet_cifar10.LeNet5(num_classes=10, grayscale=False)
        model_save_path = Path("saved_models/lenet_cifar10.pth")
        load_test_saved_model(model, model_save_path, test_dataloader, device, loss_fn, accuracy, model_name="lenet_cifar10")

    if args.model == "lenet_mnist":
        # load the test data
        _, _, test_dataloader = load_mnist_dataset("lenet_mnist", int(args.batchsz))
        model = lenet_mnist.LeNet5()
        model_save_path = Path("saved_models/lenet_mnist.pth")
        load_test_saved_model(model, model_save_path, test_dataloader, device, loss_fn, accuracy, model_name="lenet_mnist")

    if args.model == "resnet_cifar10":
        # load the test data
        _, _, test_dataloader = load_cifar10_dataset("resnet_cifar10", int(args.batchsz))
        model = resnet_cifar10.ResNet18()
        model_save_path = Path("saved_models/resnet_cifar10.pth")
        load_test_saved_model(model, model_save_path, test_dataloader, device, loss_fn, accuracy, model_name="resnet_cifar10")

    if args.model == "resnet_mnist":
        # load the test data
        _, _, test_dataloader = load_mnist_dataset("resnet_mnist", int(args.batchsz))
        model = resnet_mnist.ResNet18(num_classes=10)
        model_save_path = Path("saved_models/resnet_mnist.pth")
        load_test_saved_model(model, model_save_path, test_dataloader, device, loss_fn, accuracy, model_name="resnet_mnist")

    if args.model == "vgg_cifar10":
        # load the test data
        _, _, test_dataloader = load_cifar10_dataset("vgg_cifar10", int(args.batchsz))
        model = vgg_cifar10.VGG16(num_classes=10)
        model_save_path = Path("saved_models/vgg_cifar10.pth")
        load_test_saved_model(model, model_save_path, test_dataloader, device, loss_fn, accuracy, model_name="vgg_cifar10")

    if args.model == "vgg_mnist":
        # load the test data
        _, _, test_dataloader = load_mnist_dataset("vgg_mnist", int(args.batchsz))
        model = vgg_mnist.VGG16(num_classes=10)
        model_save_path = Path("saved_models/vgg_mnist.pth")
        load_test_saved_model(model, model_save_path, test_dataloader, device, loss_fn, accuracy, model_name="vgg_mnist")

    if args.model == "resnet_wd":
        # load the test data
        _, _, test_dataloader = load_cifar10_dataset("resnet_cifar10", int(args.batchsz))
        model = resnet_cifar10.ResNet18()
        model_save_path = Path("saved_models/resnet_cifar10_wd.pth")
        load_test_saved_model(model, model_save_path, test_dataloader, device, loss_fn, accuracy, model_name="resnet_wd")

    if args.model == "lenet_mnist_adv":
        # load the test data
        _, _, test_dataloader = load_mnist_dataset("lenet_mnist", int(args.batchsz))
        model = lenet_mnist.LeNet5()
        model_save_path = Path("adv_saved_models/lenet_mnist.pth")
        load_test_saved_model(model, model_save_path, test_dataloader, device, loss_fn, accuracy, model_name="lenet_mnist_adv")

    if args.model == "resnet_cifar10_adv":
        # load the test data
        _, _, test_dataloader = load_cifar10_dataset("resnet_cifar10", int(args.batchsz))
        model = resnet_cifar10.ResNet18()
        model_save_path = Path("adv_saved_models/resnet_cifar10.pth")
        load_test_saved_model(model, model_save_path, test_dataloader, device, loss_fn, accuracy, model_name="resnet_cifar10_adv")