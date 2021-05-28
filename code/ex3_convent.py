
"""## *Adding Wandb login*

"""

# Flexible integration for any Python script
!pip install wandb -qqq
import wandb
wandb.login()

# 1. Start a W&B run
wandb.init(project='HLCV_CNN_exercise3', entity='priyanka09')

## wandb.init(project="my-test-project") when you run into local machine

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

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512, 512]
num_epochs = 20
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
norm_layer = None

#norm_layer = None
#dropout = [None]

norm_layer = True
dropout = [i/10 for i in range(1,10)]
early_stopping = False
visualize_filter = True
data_aug = False

print(hidden_size)






#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
#################################################################################
# TODO: Q3.a Chose the right data augmentation transforms with the right        #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

if data_aug == True:

    transformsList = [
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(.3,.7)),
        transforms.RandomGrayscale(p=0.1)   
                    ]

    data_aug_transforms = transformsList


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
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
#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

"""# CNN Definition and Implementation"""

#-------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
#-------------------------------------------------
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

        ## For Q1.a make use of conv2d and relu layers from the torch.nn module. 
        ## norm_layer == None, Dropout == None, Filter_num = kernal = 3, padding = (Filter_num - 1)/2
        if norm_layer == None and Dropout == None:

          self.conv = nn.Sequential(

                      nn.Conv2d(in_channels= input_size, out_channels= hidden_size[0], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),


                      nn.Conv2d(in_channels= hidden_size[0], out_channels= hidden_size[1], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),



                      nn.Conv2d(in_channels= hidden_size[1], out_channels= hidden_size[2], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),



                      nn.Conv2d(in_channels= hidden_size[2], out_channels= hidden_size[3], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),


                      nn.Conv2d(in_channels= hidden_size[3], out_channels= hidden_size[4], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU()

          )


          self.fc = nn.Sequential(
                    
                    nn.Linear(hidden_layers[4], hidden_layers[5]),
                    nn.ReLU(),
                    nn.Linear(hidden_layers[5], num_classes)
          )

        ## For Q2.a make use of BatchNorm2d layer from the torch.nn module.
        ## Batch normalization take out_channels as input
        ## norm_layer != None, Dropout == None

        elif norm_layer != None and Dropout == None:
            self.conv = nn.Sequential(

                      nn.Conv2d(in_channels= input_size, out_channels= hidden_size[0], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.BatchNorm2d(hidden_size[0]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),


                      nn.Conv2d(in_channels= hidden_size[0], out_channels= hidden_size[1], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.BatchNorm2d(hidden_size[1]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),



                      nn.Conv2d(in_channels= hidden_size[1], out_channels= hidden_size[2], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.BatchNorm2d(hidden_size[2]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),



                      nn.Conv2d(in_channels= hidden_size[2], out_channels= hidden_size[3], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.BatchNorm2d(hidden_size[3]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),


                      nn.Conv2d(in_channels= hidden_size[3], out_channels= hidden_size[4], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.BatchNorm2d(hidden_size[4]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU()

          )


            self.fc = nn.Sequential(
                    
                    nn.Linear(hidden_layers[4], hidden_layers[5]),
                    nn.BatchNorm2d(hidden_size[4]),
                    nn.ReLU(),
                    nn.Linear(hidden_layers[5], num_classes)
          )


        ## For Q3.b Use Dropout layer from the torch.nn module.
        ## norm_layer == None, Dropout != None and norm_layer != None, Dropout != None 
        elif norm_layer == None and Dropout != None:
            self.conv = nn.Sequential(

                      nn.Conv2d(in_channels= input_size, out_channels= hidden_size[0], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      #nn.BatchNorm2d(hidden_size[0]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Dropout2d(Dropout),


                      nn.Conv2d(in_channels= hidden_size[0], out_channels= hidden_size[1], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      #nn.BatchNorm2d(hidden_size[1]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Dropout2d(Dropout),




                      nn.Conv2d(in_channels= hidden_size[1], out_channels= hidden_size[2], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      #nn.BatchNorm2d(hidden_size[2]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Dropout2d(Dropout),




                      nn.Conv2d(in_channels= hidden_size[2], out_channels= hidden_size[3], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      #nn.BatchNorm2d(hidden_size[3]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Dropout2d(Dropout),



                      nn.Conv2d(in_channels= hidden_size[3], out_channels= hidden_size[4], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      #nn.BatchNorm2d(hidden_size[4]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Dropout2d(Dropout)
            )


            self.fc = nn.Sequential(
                    
                    nn.Linear(hidden_layers[4], hidden_layers[5]),
                    #nn.BatchNorm2d(hidden_size[5]),
                    nn.ReLU(),
                    nn.Linear(hidden_layers[5], num_classes)
          )




        elif norm_layer != None and Dropout != None:
            self.conv = nn.Sequential(

                      nn.Conv2d(in_channels= input_size, out_channels= hidden_size[0], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.BatchNorm2d(hidden_size[0]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Dropout2d(Dropout),


                      nn.Conv2d(in_channels= hidden_size[0], out_channels= hidden_size[1], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.BatchNorm2d(hidden_size[1]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Dropout2d(Dropout),




                      nn.Conv2d(in_channels= hidden_size[1], out_channels= hidden_size[2], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.BatchNorm2d(hidden_size[2]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Dropout2d(Dropout),




                      nn.Conv2d(in_channels= hidden_size[2], out_channels= hidden_size[3], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.BatchNorm2d(hidden_size[3]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Dropout2d(Dropout),



                      nn.Conv2d(in_channels= hidden_size[3], out_channels= hidden_size[4], kernel_size = 3, stride= 1, padding= 1, padding_mode= 'zeros'),
                      nn.BatchNorm2d(hidden_size[4]),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Dropout2d(Dropout)
          )


            self.fc = nn.Sequential(
                    
                    nn.Linear(hidden_layers[4], hidden_layers[5]),
                    nn.BatchNorm2d(hidden_size[5]),
                    nn.ReLU(),
                    nn.Linear(hidden_layers[5], num_classes)
          )




        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_layers[-1], num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = self.features(x)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out


#-------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
#-------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model_sz = 0
    for i, param in enumerate(model.parameters()):
        if param.requires_grad:
            model_sz += param.numel()
    print("Total number of params: ", model_sz)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz

#-------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
#-------------------------------------------------
def VisualizeFilter(model):
    #################################################################################
    # TODO: Implement the function to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image for stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    visualize_filter = model.layers[0][0].weight.data.cpu().numpy()
    fig = plt.figure()
    num_row = 8
    num_col=16
    for index in range(1, num_row*num_col + 1):
      sub_fig = fig.add_subplot(num_row, num_col, index)
      sub_fig.axes.set_xticks([])
      sub_fig.axes.set_yticks([])
      plt.imshow((visualize_filter[index-1, ...] - np.min(visualize_filter[index-1]))/ (np.max(visualize_filter[index-1] - np.min(visualize_filter[index-1]))))
    plt.show()
    return

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
#======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
#--------------------------------------------------------------------------------------
loss = []
test_acc = []
train_acc = []
val_acc = []
for Dropout in dropout :

    print("######################################################################")
    print("Current value of dropout is:", Dropout)
    print("######################################################################")

model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer).to(device)



# Q2.a - Initialize the model with correct batch norm layer

model.apply(weights_init)

# Print the model
print(model)

# Print model size

#======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
#======================================================================================
PrintModelSize(model)
#======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
#======================================================================================
VisualizeFilter(model)


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

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

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

        current_val_acc = 100 * correct / total
        val_acc.append(current_val_acc)

        if early_stopping == True:
            if current_val_acc >= np.amax(val_acc):
                    torch.save(model.state_dict(), 'model'+str(epoch+1)+'.ckpt')


        """
        ## Track matrics
        ## Log the loss and accuracy values at the end of each epoch
        val_acc = 100 * correct / total
        wandb.log({
           "Epoch": epoch,
          "Train Loss": loss.item(),
           "Valid Acc": val_acc})      
        """
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

if early_stopping == True:
        last_model = model
        best_id = np.argmax(val_acc)
        model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer).to(device)
        model.load_state_dict(torch.load('model'+str(best_id+1)+'.ckpt'))
        model.eval()




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

    test_acc.append(100 * correct / total)
    if early_stopping: 
            print("Best Epoch: ", best_id)

    if early_stopping == True:

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = last_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if total == 1000:
                    break

            print('Accuracy of the last network on the {} test images: {} %'.format(total, 100 * correct / total))


# Q1.c: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
VisualizeFilter(model)
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')