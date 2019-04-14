import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext

from torch.autograd import Variable
from tqdm import tqdm
from typing import List

from framenet_tools.config import ConfigManager


class Net(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_sizes: list,
        activation_functions: list,
        num_classes: int,
        embedding_layer: torch.nn.Embedding,
        device: torch.device,
    ):
        super(Net, self).__init__()

        self.device = device
        self.embedding_layer = embedding_layer

        self.dropout = nn.Dropout(p=0.2)
        self.input_size = embedding_size * 2
        self.hidden_size = hidden_sizes[0]

        # print(embedding_size)
        self.lstm = nn.LSTM(self.input_size, hidden_sizes[0], 2, bidirectional=True, dropout=0.25)

        self.hidden_to_tag = nn.Linear(hidden_sizes[0] * 1 * 2, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            Variable(torch.zeros(1, 1, self.hidden_size)).to(self.device),
            Variable(torch.zeros(1, 1, self.hidden_size)).to(self.device),
        )

    def forward(self, x):

        outputs = []

        lookup_tensor = x.to(self.device)
        x = self.embedding_layer(lookup_tensor)

        words = []

        for i in range(len(x) - 1):
            word = torch.cat((x[i], x[-1]), 1)

            words += [word]

        words = torch.stack(words)

        # b = torch.Tensor(len(x)-1, 1, 600)
        # torch.cat(words, out=b)

        # b = b.to(self.device)

        y, (h_n, c_n) = self.lstm(words)
        # print(y)

        for i in y:
            # word = torch.cat((x[i], x[-1]), 0)

            outputs += [self.hidden_to_tag(i)]

        #outputs = outputs[:-1]

        #y = self.hidden_to_tag(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)))
        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs

        for i in range(len(x) - 1):
            word = torch.cat((x[i], x[-1]), 0)

            word = word.view(1, 1, self.input_size)
            lstm_out, self.hidden = self.lstm(word, self.hidden)

            lstm_out = self.hidden_to_tag(lstm_out)
            outputs += [lstm_out]
            # [F.log_softmax(lstm_out, dim=1)]

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs


class SpanIdNetwork(object):
    def __init__(
        self, cM: ConfigManager, embedding_layer: torch.nn.Embedding, num_classes: int
    ):

        self.cM = cM

        # Check for CUDA
        use_cuda = self.cM.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logging.debug(f"Device used: {self.device}")

        self.embedding_layer = embedding_layer
        self.num_classes = num_classes

        self.net = Net(
            self.cM.embedding_size,
            [250],
            self.cM.activation_functions,
            num_classes,
            embedding_layer,
            self.device,
        )

        self.net.to(self.device)

        # Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.cM.learning_rate
        )

    def predict(self, sent: List[int]):
        """
        Predicts the BIO-Tags of a given sentence.

        :param sent: The sentence to predict (already converted by the vocab)
        :return: A list of possibilities for each word for each tag
        """

        sent = [[t] for t in sent]

        sent = torch.tensor(sent, dtype=torch.long)

        outputs = self.net(sent)
        # _, predicted = torch.max(outputs.data, 1)

        return outputs.to("cpu")

    def reset_hidden(self):
        """
        Resets the hidden states of the LSTM.

        :return:
        """

        self.net.hidden = self.net.init_hidden()

    def train_model(
        self,
        dataset_size: int,
        train_iter: torchtext.data.Iterator,
        dev_iter: torchtext.data.Iterator = None,
    ):
        """
        Trains the model with the given dataset
        Uses the model specified in net

        :param train_iter: The train dataset iterator including all data for training
        :param dataset_size: The size of the dataset
        :return:
        """

        for epoch in range(self.cM.num_epochs):

            total_loss = 0
            total_hits = 0
            count = 0

            progress_bar = tqdm(train_iter)

            for batch in progress_bar:

                output_dim = len(batch.BIO)
                sent = batch.Sentence
                # print(batch.BIO)
                labels = torch.reshape(batch.BIO, (1, output_dim))
                labels = Variable(labels).to(self.device)

                self.net.hidden = self.net.init_hidden()

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()  # zero the gradient buffer
                outputs = self.net(sent)

                outputs = torch.reshape(outputs, (1, 3, output_dim))

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_hits += (predicted == labels).sum().item()

                count += output_dim  # labels.size(0)

                # Just update every 20 iterations
                if count % 20 == 0:
                    train_loss = round((total_loss / count), 4)
                    train_acc = round((total_hits / count), 4)
                    progress_bar.set_description(
                        f"Epoch {(epoch + 1)}/{self.cM.num_epochs} Loss: {train_loss} Acc: {train_acc} Frames: {count}/{dataset_size}"
                    )
