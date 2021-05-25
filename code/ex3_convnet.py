import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


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
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512, 512]
num_epochs = 20
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg = 0.001
num_training = 49000
num_validation = 1000
norm_layer = None
print(hidden_size)

# -------------------------------------------------
# Load the CIFAR-10 dataset
# -------------------------------------------------
#################################################################################
# TODO: Q3.a Chose the right data augmentation transforms with the right        #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms + [transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                           ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                             train=True,
                                             transform=norm_transform,
                                             download=False)

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


# -------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
# -------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # input channels for the first conv block == input_size == 3 channels
        self.input_size = input_size

        # first 5 values of hidden_layers are number of filters or out_channels
        for h_size in hidden_layers[:5]:
            # each convolution block consists of: conv, maxpool, relu
            # need to preserve spatial dimensionality
            layers.append(nn.Conv2d(in_channels=self.input_size,
                                    out_channels=h_size,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1))

            if norm_layer is not None:
                layers.append(nn.BatchNorm2d(h_size))

            # after aech maxpool, filter size dimension is halved
            layers.append(nn.MaxPool2d(kernel_size=2,
                                       stride=2))
            layers.append(nn.ReLU())

            # update the input_size for the next conv block to be == to the out_channels of the previous conv block
            self.input_size = h_size

        # flatten after 5 conv layers, nn.Flatten() default start_dim = 1 already, so no need to specify,
        # assuming dim = 0 is the batch dimension
        # the flattened dimension should be == 512*1*1 == 512 or the last value in hidden_layers list
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=hidden_layers[-1], out_features=num_classes))

        self.conv_net = nn.Sequential(*layers)

        # self.conv_net follows the schematic below per model's specification
        """
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.fc_classifier = nn.Linear(in_features=512*1*1, out_features=num_classes)
        """

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = self.conv_net(x)

        # the above function using nn.Sequential is the same as below if implemented with nn.Sequential
        """
        # 5 blocks of convolution layers
        x = nn.ReLU(self.max_pool(self.conv1(x)))
        x = nn.ReLU(self.max_pool(self.conv2(x)))
        x = nn.ReLU(self.max_pool(self.conv3(x)))
        x = nn.ReLU(self.max_pool(self.conv4(x)))
        x = nn.ReLU(self.max_pool(self.conv5(x)))

        # flatten x
        x = torch.flatten(x, start_dim=1)

        # classification layer
        out = self.fc_classifier(x)
        """
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out


# -------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
# -------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    if disp:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'{name}, {param.shape}, # params: {param.numel()}, {param}\n')

    model_sz = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total # params: {model_sz}\n')
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz


# -------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
# -------------------------------------------------
def VisualizeFilter(model, before=True):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image fo stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # conv_net[0] is the first convolution layer
    # shape: [128,3,3,3]  representing BxCxHxW
    # where B = batch, number of filters in this case, C = channels, HxW kernel or filter size
    conv1_weights = model.conv_net[0].weight.detach().clone()
    print(f"conv_net[0].weight shape: {conv1_weights.size()}")

    # make_grid takes BxCxHxW, normalize=True to adjust values of Tensor to be between (0,1)
    # ncol x nrow grid of images where each image tensor is of shape: CxHxW
    visualized_filters = torchvision.utils.make_grid(conv1_weights, nrow=16, normalize=True)

    # permute visualized_filters to be: HxWxC for matplotlib
    plt.imshow(visualized_filters.permute(1, 2, 0))
    filename = "plt_visualized_filters.png"
    if before:
        filename = f"before_{filename}"
    plt.savefig(filename, dpi=90)
    plt.show()

    # if no need to show grid, can simply save visulized vilters directly with
    # torchvision.utils.save_image as below:
    # torchvision.utils.save_image(conv1_weights, fp="visualized_filters.png", nrow=16, normalize=True)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


# ======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
# ======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
# --------------------------------------------------------------------------------------
# for Q2.a, uncomment the line 243 below
# norm_layer = True
model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer).to(device)
# Q2.a - Initialize the model with correct batch norm layer
# for Q2.a -- each conv layer batch norm will take the previous conv layer out_channels as argument
# which is turned-on if norm_layer is not None


model.apply(weights_init)
# Print the model
print(model)
# Print model size
# ======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
# ======================================================================================
PrintModelSize(model)
# ======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
# ======================================================================================
VisualizeFilter(model)
sys.exit(0)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
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
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Validataion accuracy is: {} %'.format(100 * correct / total))
        #################################################################################
        # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
        # acheieved the best validation accuracy so-far.                                #
        #################################################################################
        best_model = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model.train()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()
#################################################################################
# TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
# best model so far and perform testing with this model.                        #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Q1.c: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training <-- isn't the call below for AFTER training at this point?
VisualizeFilter(model, False)
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
