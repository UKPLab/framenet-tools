import torch
import torch.nn as nn
import torchtext
from torch.autograd import Variable
from tqdm import tqdm

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

        self.hidden_layers = []

        last_size = embedding_size * 2

        # Programmatically add new layers according to the config file
        for i in range(len(hidden_sizes)):

            self.add_module(str(i), nn.Linear(last_size, hidden_sizes[i]))

            # Saving function ref
            self.hidden_layers.append(getattr(self, str(i)))

            # Dynamic instantiation of the activation function
            act_func = getattr(nn, activation_functions[i])().to(self.device)
            self.hidden_layers.append(act_func)

            last_size = hidden_sizes[i]

        self.out_layer = nn.Linear(last_size, num_classes)

    def set_embedding_layer(self, embedding_layer: torch.nn.Embedding):
        """
        Setter for the embedding_layer

        :param embedding_layer: The new embedding_layer
        :return:
        """
        self.embedding_layer = embedding_layer

    def average_sentence(self, sent: torch.tensor):
        """
        Averages a sentence/multiple sentences by taking the mean of its embeddings

        :param sent: The given sentence as numbers from the vocab
        :return: The averaged sentence/sentences as a tensor (size equals the size of one word embedding for each sentence)
        """

        lookup_tensor = sent.to(self.device)
        embedded_sent = self.embedding_layer(lookup_tensor)

        averaged_sent = embedded_sent.mean(dim=0)

        # Reappend the FEE

        appended_avg = []

        for sentence in averaged_sent:
            inc_FEE = torch.cat((embedded_sent[0][0], sentence), 0)
            appended_avg.append(inc_FEE)

        averaged_sent = torch.stack(appended_avg)

        return averaged_sent

    def forward(self, x: torch.tensor):
        """
        The forward function, specifying the processing path

        :param x: A input value
        :return: The prediction of the network
        """

        x = Variable(self.average_sentence(x)).to(self.device)

        # Programmatically pass x through all layers
        # NOTE: hidden_layers also includes activation functions!
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        out = self.out_layer(x)

        return out


class FrameIDNetwork(object):
    def __init__(
        self, cM: ConfigManager, embedding_layer: torch.nn.Embedding, num_classes: int
    ):

        self.cM = cM

        # Check for CUDA
        use_cuda = self.cM.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print("Device used: " + str(self.device))

        self.embedding_layer = embedding_layer
        self.num_classes = num_classes

        self.net = Net(
            self.cM.embedding_size,
            self.cM.hidden_sizes,
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

    def train_model(
        self, train_iter: torchtext.data.Iterator, dataset_size: int, batch_size: int
    ):
        """
        Trains the model with the given dataset
        Uses the model specified in net

        :param train_iter: The train dataset iterator including all data for training
        :param dataset_size: The size of the dataset
        :param batch_size: The batchsize to use for training
        :return:
        """

        # batch_count = sum(1 for _ in train_iter)

        for epoch in range(self.cM.num_epochs):

            total_loss = 0
            count = 0

            pbar = tqdm(train_iter)

            for batch in pbar:

                sent = batch.Sentence
                labels = Variable(batch.Frame[0]).to(self.device)

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()  # zero the gradient buffer
                outputs = self.net(sent)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                count += 1

                # Just update every 20 iterations
                if count % 20 == 0:
                    pbar.set_description("Epoch %d/%d Loss: %f" %((epoch+1), self.cM.num_epochs, (total_loss/count)))

    def predict(self, dataset_iter: torchtext.data.Iterator):
        """
        Uses the model to predict all given input data

        :param dataset_iter: The dataset to predict
        :return: A list of predictions
        """
        predictions = []

        for batch in iter(dataset_iter):
            sent = batch.Sentence
            # sent = torch.tensor(sent, dtype=torch.long)

            outputs = self.net(sent)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.to("cpu"))

        return predictions

    def eval_model(self, dev_iter: torchtext.data.Iterator):
        """ Evaluates the model on the given dataset

            NOTE: only works on gold FEEs, therefore deprecated
                  use f1 evaluation instead

            :param dev_iter: The dataset to evaluate on
            :return: The accuracy reached on the given dataset
        """
        correct = 0.0
        total = 0.0
        for batch in iter(dev_iter):
            sent = batch.Sentence
            sent = torch.tensor(sent, dtype=torch.long)
            labels = Variable(batch.Frame[0]).to(self.device)

            outputs = self.net(sent)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        correct = correct.item()

        print(correct)
        print(total)
        accuracy = correct / total

        return accuracy

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
