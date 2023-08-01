# import the necessary packages
from torchvision import datasets, transforms
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim


# implementing the poison function for the mnist dataset
def craft_random_lflip(train_set, percent_poison):
    """
    # train_set: an instance for the training dataset
    # ratio    : the percentage of samples whose labels will be flipped
    # You can add more arguments if required
    """
    
    # If ratio is zero, return the input dataset unchanged
    if percent_poison == 0:
        return train_set

    # Create a copy of the dataset to avoid modifying the original one
    flipped_set = datasets.MNIST("./datasets/", train=True, download=True, transform=transforms.ToTensor())
    flipped_set.data = train_set.data.clone()
    flipped_set.targets = train_set.targets.clone()

    # Determine the number of samples whose labels should be flipped
    num_flips = int(len(flipped_set) * percent_poison)

    # Generate a list of indices for the samples to be flipped
    flip_indices = random.sample(range(len(flipped_set)), num_flips)

    # Flip the labels of the selected samples
    for i in flip_indices:
        target = flipped_set.targets[i]
        flipped_set.targets[i] = 1 - target

    return flipped_set

# # get the poisoned mnist dataset
# mnist_dataset_poisoned = craft_random_lflip(mnist_dataset_full, 0.05)


# implementing the poison function for the cifar10 dataset
# Step 3: Craft 100 poisons using the 100 dog images and the target frog image.
def craft_clabel_poisons(base_instance, target_instance, model, num_iterations=100):
    mean_tensor = torch.from_numpy(np.array([0.485, 0.456, 0.406]))
    std_tensor = torch.from_numpy(np.array([0.229, 0.224, 0.225]))

    unnormalized_base_instance = base_instance.clone()
    unnormalized_base_instance[:, 0, :, :] *= std_tensor[0]
    unnormalized_base_instance[:, 0, :, :] += mean_tensor[0]
    unnormalized_base_instance[:, 1, :, :] *= std_tensor[1]
    unnormalized_base_instance[:, 1, :, :] += mean_tensor[1]
    unnormalized_base_instance[:, 2, :, :] *= std_tensor[2]
    unnormalized_base_instance[:, 2, :, :] += mean_tensor[2]

    perturbed_instance = unnormalized_base_instance.clone()
    target_features, outputs = model(target_instance)

    transforms_normalization = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    epsilon = 16 / 255
    alpha = 0.05 / 255

    start_time = time.time()
    for i in range(num_iterations):
        perturbed_instance.requires_grad = True

        poison_instance = transforms_normalization(perturbed_instance)
        poison_features, _ = model(poison_instance)

        feature_loss = nn.MSELoss()(poison_features, target_features)
        image_loss = nn.MSELoss()(poison_instance, base_instance)
        loss = feature_loss + image_loss / 1e2
        loss.backward()

        signed_gradient = perturbed_instance.grad.sign()

        perturbed_instance = perturbed_instance - alpha * signed_gradient
        eta = torch.clamp(perturbed_instance - unnormalized_base_instance, -epsilon, epsilon)
        perturbed_instance = torch.clamp(unnormalized_base_instance + eta, 0, 1).detach()

        # if i == 0 or (i + 1) % 500 == 0:
        #     print(f'Feature loss: {feature_loss}, Image loss: {image_loss}, Time: {time.time() - start_time}')

    poison_instance = transforms_normalization(perturbed_instance)
    # imshow(poison_instance[0].cpu(), 'Poison Instance')
    
    return poison_instance

