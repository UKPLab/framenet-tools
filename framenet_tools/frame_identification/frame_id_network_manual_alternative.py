import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchtext
from torchtext import data
from torchtext import vocab
import pandas as pd 
import numpy as np

from reader import Data_reader

def load_glove_model(glove_file):
	print("Loading Glove Model")
	f = open(glove_file,'r')
	model = {}
	for line in f:
		split_line = line.split()
		word = split_line[0]
		embedding = np.array([float(val) for val in split_line[1:]])
		model[word] = embedding
	return model

def embed_dataset(xs, model):
	embeddings_size = 300
	embedded_xs = []

	for x in xs:
		embedded_sentence = []

		for word in x:
			#Default value
			embedded_word = [0] * embeddings_size 

			if word in model:
				embedded_word = model[word]
			
			#print(embedded_word.eval())
			#exit()
			
			embedded_sentence.append(embedded_word)

		embedded_xs.append(embedded_sentence)

	return embedded_xs

def pad_dataset(embeddded_xs):
	embeddings_size = 300

	max_length = get_max_length(embedded_xs)

	padded_xs = []

	for x in embedded_xs:
		while len(x)< max_length:
			x.append([0]*embeddings_size)

		padded_xs.append(x)

	return padded_xs


def get_max_length(xs):
	max_length = 0

	for x in xs:
		if len(x) > max_length:
			max_length = len(x)

	return max_length

def build_dict(ys):
	_dict = dict()
	count = 0

	for y in ys:
		if not y in _dict:
			_dict[y] = count
			count += 1

	return _dict

def dict_convert_dataset(ys, _dict):
	converted_ys = []

	for y in ys:
		converted_ys.append(_dict[y])

	return converted_ys

def to_one_hot(converted_ys, _dict):
	one_hot_ys = []
	_dict_length = len(_dict)

	for y in converted_ys:
		one_hot_y = [0]* _dict_length
		one_hot_y[y] = 1

		one_hot_ys.append(one_hot_y)

	return one_hot_ys


#reader = Data_reader("../data/experiments/xp_001/data/dev.sentences", "../data/experiments/xp_001/data/dev.frames")

reader = Data_reader("../data/experiments/xp_001/data/train.sentences", "../data/experiments/xp_001/data/train.frame.elements")
reader.read_data()
dataset = reader.get_dataset()

dataset_size = len(dataset)



xs = [i[0] for i in dataset]
ys = [i[1] for i in dataset]

#print(xs[0])

#vectors = vocab.GloVe(name='6B', dim=300)
model = load_glove_model(".vector_cache/glove.6B.300d.txt")
embedded_xs = embed_dataset(xs, model)
max_length = get_max_length(embedded_xs)
padded_xs = pad_dataset(embedded_xs)

#print(padded_xs[0])
#exit()

_dict = build_dict(ys)
converted_ys = dict_convert_dataset(ys, _dict)
one_hot_ys = to_one_hot(converted_ys, _dict)

padded_xs = torch.tensor(padded_xs, dtype=torch.float)
one_hot_ys = torch.tensor(one_hot_ys)
converted_ys = torch.tensor(converted_ys)
#print(one_hot_ys[0])



input_size = max_length*300
hidden_size = 500
num_classes = len(_dict)
num_epochs = 1
embed_dim = 66*300
batch_size = 1
learning_rate = 0.001


#Creat network
class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(Net, self).__init__()
		#self.embedding = nn.Embedding(input_size, 300).from_pretrained(input_field.vocab.vectors)
		self.fc1 = nn.Linear(input_size, hidden_size) 
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_classes)  
	
	def forward(self, x):
		#out = self.embedding(x)
		#print(x)
		#print(out)

		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out

	

net = Net(input_size, hidden_size, num_classes)
net.cuda()   

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

iterations = round(dataset_size/batch_size)

# Train the Model
for epoch in range(num_epochs):   
	i = 0

	for x,y in zip(padded_xs,converted_ys):
		
		x = x.view(1,-1)
		i += 1
		
		sent = Variable(x).cuda()
		#print(batch.Frame[0])
		labels = Variable(y).cuda()

		
		# Forward + Backward + Optimize
		optimizer.zero_grad()  # zero the gradient buffer
		outputs = net(sent)

		#print(outputs)
		#print(torch.argmax(labels))
		loss = criterion(outputs, labels.view(1))
		loss.backward()
		optimizer.step()
		
		if (i+1) % 100 == 0:
			#print(loss)
			print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
				   %(epoch+1, num_epochs, i+1, iterations, loss.data[0]))

# Test the Model
correct = 0
total = 0
for x,y in zip(padded_xs,converted_ys):
	x = x.view(1,-1)
	sent = Variable(x).cuda()
	labels = Variable(y).cuda()

	outputs = net(sent)
	_, predicted = torch.max(outputs.data, 1)
	total += 1
	correct += (predicted == labels).sum()
print(correct)
print(total)