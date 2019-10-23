import logging
import torch
import torch.nn as nn
import torchtext
import os

from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
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

        self.hidden_layers = []

        last_size = embedding_size * 2

        logging.debug(f"Hidden sizes: {hidden_sizes}")
        logging.debug(f"Activation functions: {activation_functions}")

        # Programmatically add new layers according to the config file
        for i in range(len(hidden_sizes)):

            if activation_functions[i].lower() == "dropout":
                # Add dropout
                self.add_module(str(i), nn.Dropout(hidden_sizes[i]))
                self.hidden_layers.append(getattr(self, str(i)))

                continue

            hidden_sizes[i] = int(hidden_sizes[i])

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

        appended_avg = []

        for sentence in lookup_tensor:

            # Cut off padding from torchtext, as it messes up the averaging process!
            sentence = sentence[: (sentence != 1).nonzero()[-1].item() + 1]
            embedded_sent = self.embedding_layer(sentence)

            averaged_sent = embedded_sent.mean(dim=0)

            # Reappend the FEE
            inc_FEE = torch.cat((embedded_sent[0], averaged_sent), 0)
            appended_avg.append(inc_FEE)

        averaged_sent = torch.stack(appended_avg)

        return averaged_sent

    def forward(self, x: torch.tensor):
        """
        The forward function, specifying the processing path

        :param x: A input value
        :return: The prediction of the network
        """

        x = torch.transpose(x, 0, 1)
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
        logging.debug(f"Device used: {self.device}")

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
        self,
        dataset_size: int,
        train_iter: torchtext.data.Iterator,
        dev_iter: torchtext.data.Iterator = None,
    ):
        """
        Trains the model with the given dataset
        Uses the model specified in net

        :param dev_iter: The dev dataset for performance measuring
        :param train_iter: The train dataset iterator including all data for training
        :param dataset_size: The size of the dataset
        :param batch_size: The batch size to use for training
        :return:
        """

        highest_acc = 0
        auto_stopper = self.cM.autostopper and dev_iter is not None
        last_improvement = 0
        autostopper_threshold = self.cM.autostopper_threshold

        writer = SummaryWriter()

        for epoch in range(self.cM.num_epochs):

            total_loss = 0
            total_hits = 0
            count = 0

            progress_bar = tqdm(train_iter)

            for batch in progress_bar:

                sent = batch.Sentence
                labels = Variable(batch.Frame[0]).to(self.device)

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()  # zero the gradient buffer
                outputs = self.net(sent)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_hits += (predicted == labels).sum().item()

                count += labels.size(0)

                # Just update every 20 iterations
                if count % 20 == 0:
                    train_loss = round((total_loss / count), 4)
                    train_acc = round((total_hits / count), 4)
                    progress_bar.set_description(
                        f"Epoch {(epoch + 1)}/{self.cM.num_epochs} Loss: {train_loss} Acc: {train_acc} Frames: {count}/{dataset_size}"
                    )

            train_loss = total_loss / count
            train_acc = total_hits / count

            if dev_iter is None:
                logging.info(f"Train Acc: {train_acc}, Train Loss: {train_loss}")

                writer.add_scalars("data/loss", {"train_loss": train_loss}, epoch)

                writer.add_scalars("data/acc", {"train_acc": train_acc}, epoch)
                continue

            dev_acc, dev_loss = self.eval_model(dev_iter)

            last_improvement += 1

            if dev_acc > highest_acc:
                highest_acc = dev_acc

                last_improvement = 0

                self.save_model(self.cM.saved_model + ".auto")

            logging.info(
                f"Train Acc: {train_acc}, Dev Acc: {dev_acc}, Train Loss: {train_loss}, Dev Loss: {dev_loss}"
            )

            writer.add_scalars(
                "data/loss", {"train_loss": train_loss, "dev_loss": dev_loss}, epoch
            )

            writer.add_scalars(
                "data/acc", {"train_acc": train_acc, "dev_acc": dev_acc}, epoch
            )

            if auto_stopper and (last_improvement > autostopper_threshold):
                writer.close()
                return

        writer.close()

    def query(self, x: List[int]):
        """
        Query a single sentence
        
        :param x: A list of ints representing words according to the embedding dictionary
        :return: The prediction of the frame
        """

        x = torch.tensor(x)

        output = self.net(x)
        # _, predicted = torch.max(output.data, 1)

        return output.data.to("cpu")

    def predict(self, dataset_iter: torchtext.data.Iterator):
        """
        Uses the model to predict all given input data

        :param dataset_iter: The dataset to predict
        :return: A list of predictions
        """
        predictions = []

        for batch in iter(dataset_iter):
            sent = batch.Sentence

            outputs = self.net(sent)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.to("cpu"))

        return predictions

    def eval_model(self, dev_iter: torchtext.data.Iterator):
        """ Evaluates the model on the given dataset

            UPDATE: again required and integrated for evaluating the accuracy during training.
            Still not recommended for final evaluation purposes.

            NOTE: only works on gold FEEs, therefore deprecated
                  use f1 evaluation instead

            :param dev_iter: The dataset to evaluate on
            :return: The accuracy reached on the given dataset
        """

        eval_criterion = nn.CrossEntropyLoss()

        correct = 0.0
        total = 0.0

        loss = 0.0

        for batch in iter(dev_iter):
            sent = batch.Sentence

            labels = Variable(batch.Frame[0]).to(self.device)

            outputs = self.net(sent)
            batch_loss = eval_criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            total += self.cM.batch_size

            correct += (predicted == labels).sum()

            loss += batch_loss.item()

        correct = correct.item()

        logging.debug(f"Correct predictions: {correct} Total examples: {total}")

        accuracy = correct / total
        loss = loss / total

        return accuracy, loss

    def save_model(self, path: str):
        """
        Saves the current model at the given path

        :param path: The path to save the model at
        :return:
        """

        upper_path = path[:path.rfind('/')]

        if not os.path.isdir(upper_path):
            os.makedirs(upper_path)

        torch.save(self.net.state_dict(), path)

    def load_model(self, path: str):
        """
        Loads the model from a given path

        :param path: The path from where to load the model
        :return:
        """

        self.net.load_state_dict(torch.load(path))
