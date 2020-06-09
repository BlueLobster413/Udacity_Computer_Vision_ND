import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True) #Bringing/Using Resnet as our CNN
        for param in resnet.parameters(): # For parameters in all layers
            param.requires_grad_(False) # Requires no changes to parameters, so freezing them
        
        modules = list(resnet.children())[:-1] 
        #Choosing all layers in Resnet except the last layer which generally gives an output for Resnet, but we are modifying it.
        
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size) 
        # Embedding layer (FC linear) taking in the features from CNN and giving out with number of nodes = embed_size. These will be input to RNN

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size # Defined as 512
        self.hidden_size = hidden_size # Defined as 512
        self.vocab_size = vocab_size # For vocab_length 4, size is 9955
        self.num_layers = num_layers
        self.device = device
        
        
        # Initialize Embedding layer
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        # Initialize LSTM Layer
        self.lstm = nn.LSTM(input_size = self.embed_size,
                            num_layers = self.num_layers,
                            hidden_size = self.hidden_size,
                            batch_first = True)
        
        # Initialize Fully Connected Layer
        self.fc1 = nn.Linear(self.hidden_size, self.vocab_size)
        
    
    def forward(self, features, captions):
        
        # Initialize the hidden state
	self.batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                	torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)) # (1,batch,512)
        
        # Embedding the captions
        embedded = self.embed(captions[:,:-1])
        
        # Embedding Image Features
        embedded = torch.cat((features.unsqueeze(1), embedded), dim=1)
        
        # LSTM Cell taking embedded as input
        lstm_out, hidden_out = self.lstm(embedded, self.hidden)
        
        # Passing the LSTM output to Linear FC layer
        fc_out = self.fc1(lstm_out)
        
        return fc_out
               
        
    def sample(self, inputs, states=None, max_len=20):
        output = []
        # As it's a sample, Batch_size is 1
        hidden = (torch.randn(1, 1, 512).to(device),
                  torch.randn(1, 1, 512).to(device))

        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.fc1(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_pred_index = torch.max(outputs, dim = 1)
            output.append(max_pred_index.cpu().numpy()[0].item())
            if (max_pred_index == 1):
                break
            inputs = self.embed(max_pred_index)
            inputs = inputs.unsqueeze(1)
        return output
        
