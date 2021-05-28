import argparse

import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import sys
from tqdm import tqdm, trange
from pathlib import Path
import wandb


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# --------------------------------
# Device configuration
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# --------------------------------
# Hyper-parameters
# --------------------------------
parser = argparse.ArgumentParser(description='ex3 convnet param options')
parser.add_argument('-e', '--epoch', type=int, default=30, help='Number of epochs')
parser.add_argument('-s', '--e_stop', type=bool, default=True, help='Apply early stop')
parser.add_argument('-c', '--comment', type=str, default="q4a", help='Run comment')

args = parser.parse_args()

# get hyperparameters from cl for experiments
print(f'CL-Arguments: {args}')

input_size = 32 * 32 * 3
layer_config = [512, 256]
num_classes = 10
num_epochs = args.epoch
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.99
reg = 0  # 0.001
num_training = 49000
num_validation = 1000
fine_tune = True
pretrained = True

print(f'layer_config: {layer_config}')
# update hyperparameters wandb is tracking
# set up wandb for hyperparameters logging and tuning
wandb.init(project="HLCV_CNN_3", name=args.comment)

wandb.config.epochs = args.epoch
wandb.config.dropout = 0
wandb.config.jitter = 0
wandb.config.norm_layer = "BatchNorm"
wandb.config.data_augment = 1
wandb.config.early_stop = "early stopping" if args.e_stop else "w/o early stopping"


data_aug_transforms = [transforms.RandomHorizontalFlip(p=0.5)]  # , transforms.RandomGrayscale(p=0.05)]

# -------------------------------------------------
# Load the CIFAR-10 dataset
# -------------------------------------------------
# Q1,
norm_transform = transforms.Compose(data_aug_transforms + [transforms.ToTensor(),
                                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                                [0.229, 0.224, 0.225])
                                                           ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                             train=True,
                                             transform=norm_transform,
                                             download=False)

# no augmentation for testing
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                                                [0.229, 0.224, 0.225])])

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                            train=False,
                                            transform=test_transform
                                            )
# -------------------------------------------------
# Prepare the training and validation splits
# -------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

# -------------------------------------------------
# Data loader
# -------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class VggModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(VggModel, self).__init__()
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the pretrained flag. You can enable and  #
        # disable training the feature extraction layers based on the fine_tune flag.   #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.features = torchvision.models.vgg11_bn(pretrained=pretrained).features
        if fine_tune:
            set_parameter_requires_grad(self.features, fine_tune)

        # 2-layer classifier with BatchNorm(use 1d because BatchNorm2d required 4d input size)
        layers = []
        layers.append(torch.nn.Linear(in_features=layer_config[0], out_features=layer_config[1]))
        layers.append(torch.nn.BatchNorm1d(num_features=layer_config[1]))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(in_features=layer_config[1], out_features=num_classes))

        self.classifier = torch.nn.Sequential(*layers)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        out = self.classifier(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out


# Initialize the model for this run
model = VggModel(num_classes, fine_tune, pretrained)

# Print the model we just instantiated
print(model)

#################################################################################
# TODO: Only select the required parameters to pass to the optimizer. No need to#
# update parameters which should be held fixed (conv layers).                   #
#################################################################################
print("Params to learn:")
if fine_tune:
    params_to_update = []
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print(f"\t{name}")
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
else:
    params_to_update = model.parameters()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)

model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update, lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)
best_validation_acc = 0.0  # this is added for Q2.B early stopping

for epoch in trange(num_epochs, desc="training epoch"):
    for i, (images, labels) in enumerate(tqdm(train_loader, desc="training batch")):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        print(f"image size: {images.size()}")
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(val_loader, desc="validating"):
            images = images.to(device)
            labels = labels.to(device)
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        #################################################################################
        # TODO: Q2.b Use the early stopping mechanism from previous questions to save   #
        # the model which has acheieved the best validation accuracy so-far.            #
        #################################################################################
        best_model = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if args.e_stop:
            current_epoch_val_acc = correct / total
            if current_epoch_val_acc > best_validation_acc:
                best_validation_acc = current_epoch_val_acc
                best_model = model
                torch.save(best_model.state_dict(), f'{num_epochs}_early_stopping_model.pt')
                print(f'Saving model with best validation accuracy so-far...\n')
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        print('Validataion accuracy is: {} %'.format(100 * correct / total))

    # Log the loss and accuracy values at the end of each epoch
    val_accuracy = 100 * correct / total
    wandb.log({
        "Epoch": epoch,
        "Train Loss": loss.item(),
        "Valid Acc": val_accuracy})

#################################################################################
# TODO: Use the early stopping mechanism from previous question to load the     #
# weights from the best model so far and perform testing with this model.       #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
if args.e_stop:
    torch.save(model.state_dict(), f'after_training_{num_epochs}epochs_model.pt')  # saving the model state dict w/o early stopping
    print(f'Best Validataion accuracy is: {100 * best_validation_acc}')
    model.load_state_dict(torch.load(f'{num_epochs}_early_stopping_model.pt'))
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc="testing"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
    test_accuracy = 100 * correct / total
    wandb.run.summary["best_accuracy"] = test_accuracy

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
