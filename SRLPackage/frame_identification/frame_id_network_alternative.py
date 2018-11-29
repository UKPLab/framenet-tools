import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchtext
from torchtext import data
from torchtext import vocab
import pandas as pd 

from reader import Data_reader


reader = Data_reader("../data/experiments/xp_001/data/train.sentences", "../data/experiments/xp_001/data/train.frame.elements")
reader.read_data()
dataset = reader.get_dataset()

dataset_size = len(dataset)

raw_data = {'Sentence' : [i[0] for i in dataset], 'Frame': [j[1] for j in dataset]}
df = pd.DataFrame(raw_data, columns=["Sentence", "Frame"])

df.to_csv("train.csv", index=False)

xs = [i[0] for i in dataset]
ys = [i[1] for i in dataset]

max_length = 0

for x in xs:
    if len(x) > max_length:
        max_length = len(x)

input_field = data.Field(dtype=torch.float, fix_length= max_length)
output_field = data.Field(dtype=torch.long)

data_fields = [('Sentence', input_field), ('Frame', output_field)]

my_dataset = data.TabularDataset("train.csv", format="csv", fields=data_fields)

input_field.build_vocab(my_dataset,vectors="glove.6B.300d")
output_field.build_vocab(my_dataset)

train_iter = data.BucketIterator(my_dataset, batch_size=1, shuffle=False)


print(len(output_field.vocab))

print(raw_data.keys())

#for i in iter(train_iter):
#	print(i.Sentence[0])


input_size = max_length
hidden_size = 500
num_classes = len(output_field.vocab)
num_epochs = 5
batch_size = 1
learning_rate = 0.001


#Creat network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#Check for CUDA
use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")    
print(device)

net = Net(input_size, hidden_size, num_classes)

net.to(device)   

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

iterations = round(dataset_size/batch_size)

# Train the Model
for epoch in range(num_epochs):   
    i = 0

    for batch in iter(train_iter):
        #print(batch.Sentence.view(batch_size, 66))
        #print(batch.Frame[0])
        i += 1
        sent = Variable(batch.Sentence.view(batch_size, max_length)).to(device)
        #print(batch.Frame[0])
        labels = Variable(batch.Frame[0]).to(device)

        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(sent)

        #print(outputs)
        #print(labels)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            #print(loss)
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, iterations, loss.data[0]))

# Test the Model
correct = 0
total = 0
for batch in iter(train_iter):
    sent = Variable(batch.Sentence.view(batch_size, max_length)).to(device)
    labels = Variable(batch.Frame[0]).to(device)

    outputs = net(sent)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    #print(predicted)
    #print(labels)
    correct += (predicted == labels).sum()
    #print(correct)

print(correct)
print(total)

