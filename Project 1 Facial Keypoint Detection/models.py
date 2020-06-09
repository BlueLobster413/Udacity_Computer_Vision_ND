## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # As seen from paper, Conv, Activation, Pool, Dropout is the pattern used
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        # Image size is [1,224,224] - output size =(W-F)/s +1 --> (224-5)/1 +1 = 220
        # After maxpool with 2x2 window, size -> [32,110,110]
        self.pool = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(32, 64, 5) # output = (110-5)/1 +1 = 106. After max pooling, [64,53,53]
        self.conv3 = nn.Conv2d(64, 128, 3) # output = (53-3)/1 +1 = 51, after max pooling, [128,25,25]
        self.conv4 = nn.Conv2d(128, 256, 3) #output = (25-3)/1 +1 = 23, after max pooling, [256,11,11]
        self.conv5 = nn.Conv2d(256, 512, 3) # output = (11-3)/1 +1 = 9, after max pooling, [512,4,4] - each feature map [1,4,4]
        
You could have tried to implement 2D BatchNorm layers after the conv layers and 1D BatchNorm layers after the linear layers as well.
Would definitely have improved model accuracy.

For example, here is a batchnorm layer that can be introduced after the first conv layer.
	#self.batchnorm32 = nn.BatchNorm2d(32) # layer definition
	#x = self.pool(F.relu(self.batchnorm32(self.conv1(x)))) # forward call
        
	self.dense1 = nn.Linear(512*4*4, 3000)
        self.dense2 = nn.Linear(3000, 800)
        self.dense3 = nn.Linear(800, 136)
        
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.4)
        self.drop7 = nn.Dropout(p=0.4)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = self.pool(F.relu(self.conv5(x)))
        x = self.drop5(x)
        
        
        x = x.view(x.size(0), -1)
        
        
        x = F.relu(self.dense1(x))
        x = self.drop6(x)
        
        x = F.relu(self.dense2(x))
        x = self.drop7(x)
        
        x = self.dense3(x)
        # Not putting a softmax because this is regression, not classification
        # a modified x, having gone through all the layers of your model, should be returned
        return x
