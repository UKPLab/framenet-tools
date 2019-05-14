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
        device: torch.device,
    ):
        super(Net, self).__init__()

        self.device = device

        self.dropout = nn.Dropout(p=0.2)
        self.input_size = 400
        self.hidden_size = hidden_sizes[0]

        # print(embedding_size)
        self.lstm = nn.LSTM(self.input_size, hidden_sizes[0], 2, bidirectional=True, dropout=0.25)

        self.hidden_to_tag = nn.Linear(hidden_sizes[0] * 2, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            Variable(torch.zeros(1, 1, self.hidden_size)).to(self.device),
            Variable(torch.zeros(1, 1, self.hidden_size)).to(self.device),
        )

    def forward(self, x):

        outputs = []

        x = torch.cat(x).view(len(x), 1, -1)

        x = Variable(x).to(self.device)

        #lookup_tensor = x.to(self.device)
        #x = self.embedding_layer(lookup_tensor)

        #words = []

        #for i in range(len(x) - 1):
        #    word = torch.cat((x[i], x[-1]), 1)

        #    words += [word]

        #words = torch.stack(words)

        # b = torch.Tensor(len(x)-1, 1, 600)
        # torch.cat(words, out=b)

        # b = b.to(self.device)

        y, (h_n, c_n) = self.lstm(x)
        # print(y)
        outputs = []

        for i in y:
            outputs.append(self.hidden_to_tag(i))

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs

        hidden_states = []

        for i in range(len(x) - 1):
            word = torch.cat((x[i], x[-1]), 0)

            word = word.view(1, 1, self.input_size)
            lstm_out, self.hidden = self.lstm(word, self.hidden)

            hidden_states.append(self.hidden)


            lstm_out = self.hidden_to_tag(lstm_out)
            outputs += [lstm_out]
            # [F.log_softmax(lstm_out, dim=1)]

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs


class SpanIdNetwork(object):
    def __init__(
        self, cM: ConfigManager, num_classes: int
    ):

        self.cM = cM

        # Check for CUDA
        use_cuda = self.cM.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logging.debug(f"Device used: {self.device}")

        #self.embedding_layer = embedding_layer
        self.num_classes = num_classes

        self.net = Net(
            self.cM.embedding_size,
            [250],
            self.cM.activation_functions,
            num_classes,
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

        #sent = [[t] for t in sent]

        #sent = torch.tensor(sent, dtype=torch.long)

        self.reset_hidden()

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
        xs, ys, dev_xs, dev_ys
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
            perf_match = 0
            count = 0
            occ = 0
            dataset_size = len(xs)

            progress_bar = tqdm(zip(xs, ys))

            for x, y in progress_bar:

                output_dim = len(x)
                #sent = batch.Sentence
                # print(batch.BIO)

                #labels = []

                labels = Variable(torch.tensor(y)).to(self.device)
                labels = torch.reshape(labels, (1, output_dim))

                #x = Variable(x).to(self.device)

                self.net.hidden = self.net.init_hidden()

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()  # zero the gradient buffer
                outputs = self.net(x)

                outputs = torch.reshape(outputs, (1, 3, output_dim))


                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                su = sum(predicted[0])

                occ += su # /len(predicted[0])
                total_hits += (predicted == labels).sum().item()/len(predicted[0])
                if (predicted == labels).sum().item() == len(predicted[0]):
                    perf_match += 1

                count += 1  # labels.size(0)

                # Just update every 20 iterations
                if count % 20 == 0:
                    train_loss = round((total_loss / count), 4)
                    train_acc = round((total_hits / count), 4)
                    perf_acc = round((perf_match / count), 4)
                    progress_bar.set_description(
                        f"Epoch {(epoch + 1)}/{self.cM.num_epochs} Loss: {train_loss} Acc: {train_acc} Perfect: {perf_acc} Frames: {count}/{dataset_size} OccSpans: {occ}"
                    )

            self.eval_dev(dev_xs, dev_ys)

    def eval_dev(self, xs, ys):
        """

        :param xs:
        :param ys:
        :return:
        """

        hits = 0
        span_hits = 0
        total = 0

        for x, y in zip(xs, ys):
            bio_tags =self.predict(x)[0]

            bio_tags = torch.argmax(bio_tags, 1)

            for gold, pred in zip(y, bio_tags):
                if gold == pred:
                    hits += 1

            total += len(y)

        print(f"DEV-Acc: {round((hits/total),4)} Span-acc: {round((span_hits/total),4)}")