import logging
import torch
import torch.nn as nn

from torch.autograd import Variable
from tqdm import tqdm
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.utils.static_utils import shuffle_concurrent_lists


class Net(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        frame_embedding_size: int,
        hidden_sizes: list,
        layers: list,
        num_classes: int,
        device: torch.device,
        embedding_layer: torch.nn.Embedding,
    ):
        super(Net, self).__init__()

        self.device = device

        self.embedding_layer = embedding_layer

        self.input_size = 401
        self.hidden_size = 450
        self.hidden_size2 = 200

        # Dynamic instantiation of the activation function
        #act_func = getattr(nn, span_layers[i])().to(self.device)

        logging.debug(f"Hidden sizes: {hidden_sizes}")
        logging.debug(f"Activation functions: {layers}")

        self.hidden_layers = []
        last_size = embedding_size + frame_embedding_size + 4

        for i in range(len(hidden_sizes)):

            if layers[i].lower() == "dropout":
                # Add dropout
                self.add_module(str(i), nn.Dropout(hidden_sizes[i]))
                self.hidden_layers.append(getattr(self, str(i)))

                continue

            hidden_sizes[i] = int(hidden_sizes[i])

            self.add_module(str(i),  getattr(nn, layers[i])(last_size, hidden_sizes[i], bidirectional=True).to(self.device))

            # Saving function ref
            self.hidden_layers.append(getattr(self, str(i)))

            # Double due to the bidirectional processing
            last_size = hidden_sizes[i] * 2

        # Last layer
        self.hidden_to_tag = nn.Linear(last_size, num_classes)

    def forward(self, x):

        sent_len = len(x)

        x = torch.tensor(x).to(self.device)

        embedded = self.embedding_layer(x[:, :1].type(torch.long))

        x = torch.cat((embedded.view(len(embedded), -1), x[:, 1:]), 1)

        x = x.view(sent_len, 1, -1)

        x = Variable(x).to(self.device)

        # As every sequence is porcessed at once, only the outputs are required
        for hidden_layer in self.hidden_layers:
            if isinstance(hidden_layer, nn.Dropout):
                x = hidden_layer(x)
                continue

            x, _ = hidden_layer(x)

        outputs = []

        for i in x:
            outputs.append(self.hidden_to_tag(i))

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs


class SpanIdNetwork(object):
    def __init__(self, cM: ConfigManager, num_classes: int, embedding_layer: torch.nn.Embedding,):

        self.cM = cM
        self.best_acc = 0

        # Check for CUDA
        use_cuda = self.cM.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logging.debug(f"Device used: {self.device}")

        self.num_classes = num_classes

        self.net = Net(
            self.cM.embedding_size,
            100,
            self.cM.span_hidden_sizes,
            self.cM.span_layers,
            num_classes,
            self.device,
            embedding_layer,
        )

        self.net.to(self.device)

        # Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.cM.span_learning_rate
        )

    def predict(self, sent: List[int]):
        """
        Predicts the BIO-Tags of a given sentence.

        :param sent: The sentence to predict (already converted by the vocab)
        :return: A list of possibilities for each word for each tag
        """

        self.reset_hidden()

        outputs = self.net(sent)

        return outputs.to("cpu")

    def reset_hidden(self):
        """
        Resets the hidden states of the LSTM.

        :return:
        """

        # NOT needed anymore
        #self.net.hidden = self.net.init_hidden()
        #self.net.hidden2 = self.net.init_hidden2()

    def train_model(
        self,
        xs: List[torch.tensor],
        ys: List[List[int]],
        dev_xs: List[torch.tensor] = None,
        dev_ys: List[List[int]] = None,
    ):
        """
        Trains the model with the given dataset
        Uses the model specified in net

        :param xs: The training sequences, given as a list of tensors
        :param ys: The labels of the sequences
        :param dev_xs: The development sequences, given as a list of tensors
        :param dev_ys: The labels of the sequences
        :return:
        """

        dataset_size = len(xs)

        for epoch in range(self.cM.span_num_epochs):

            total_loss = 0
            total_hits = 0
            perf_match = 0
            count = 0
            occ = 0

            shuffle_concurrent_lists([xs, ys])

            progress_bar = tqdm(zip(xs, ys))

            for x, y in progress_bar:

                output_dim = len(x)

                labels = Variable(torch.tensor(y)).to(self.device)
                labels = torch.reshape(labels, (1, output_dim))

                self.reset_hidden()

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

                occ += su
                total_hits += (predicted == labels).sum().item() / len(predicted[0])
                if (predicted == labels).sum().item() == len(predicted[0]):
                    perf_match += 1

                count += 1

                # Just update every 20 iterations
                if count % 20 == 0:
                    train_loss = round((total_loss / count), 4)
                    train_acc = round((total_hits / count), 4)
                    perf_acc = round((perf_match / count), 4)
                    progress_bar.set_description(
                        f"Epoch {(epoch + 1)}/{self.cM.num_epochs} Loss: {train_loss} Acc: {train_acc} Perfect: {perf_acc} Frames: {count}/{dataset_size} OccSpans: {occ}"
                    )

            self.eval_dev(dev_xs, dev_ys)

    def eval_dev(self, xs: List[torch.tensor] = None, ys: List[List[int]] = None):
        """
        Evaluates the model directly on the a prepared dataset

        :param xs: The development sequences, given as a list of tensors
        :param ys: The labels of the sequence
        :return:
        """

        hits = 0
        span_hits = 0
        total = 0

        for x, y in zip(xs, ys):
            bio_tags = self.predict(x)[0]

            bio_tags = torch.argmax(bio_tags, 1)

            for gold, pred in zip(y, bio_tags):
                if gold == pred:
                    hits += 1

            total += len(y)

        acc = round((hits / total), 4)

        print(f"DEV-Acc: {acc} Span-acc: {round((span_hits/total),4)}")

        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model("data/models/span_test.m")

    def save_model(self, path: str):
        """
        Saves the current model at the given path

        :param path: The path to save the model at
        :return:
        """

        torch.save(self.net.state_dict(), path)

    def load_model(self, path: str):
        """
        Loads the model from a given path

        :param path: The path from where to load the model
        :return:
        """

        self.net.load_state_dict(torch.load(path))
