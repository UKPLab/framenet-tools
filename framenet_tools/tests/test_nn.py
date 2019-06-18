import pytest
import random
import torch
import torch.nn as nn

from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.frame_identification.frameidentifier import FrameIdentifier
from framenet_tools.frame_identification.frameidnetwork import Net, FrameIDNetwork
from framenet_tools.tests.test_reader import create_random_string, RandomFiles

ACTIVATION_FUNCTIONS = [
    "LogSoftmax",
    "Softmax2d",
    "Softmax",
    "Softmin",
    "Tanhshrink",
    "Softsign",
    "PReLU",
    "Softshrink",
    "Softplus",
    "LogSigmoid",
    "LeakyReLU",
    "Hardshrink",
    "GLU",
    "SELU",
    "ELU",
    "Tanh",
    "Sigmoid",
    "ReLU6",
    "Hardtanh",
    "RReLU",
    "ReLU",
]


def create_network(
    embedding_vocab_size: int = 4,
    embedding_dim: int = 2,
    hidden_sizes: List[int] = [128],
    activation_functions: List[str] = ["ReLU"],
    num_classes: int = 2,
):
    """
    Helper function for parameterizable creation of the neural network

    :param embedding_vocab_size: Number of words in the vocab
    :param embedding_dim: Dimension size of the embeddings
    :param hidden_sizes: A list including the sizes of the hidden layers
    :param activation_functions: A list of the names of activation functions to be used
    :param num_classes: The number of possible predictable classes
    :return:
    """

    embedding_layer = nn.Embedding(embedding_vocab_size, embedding_dim)
    net = Net(
        embedding_dim,
        hidden_sizes,
        activation_functions,
        num_classes,
        embedding_layer,
        torch.device("cpu"),
    )

    return net


def generate_layers(layer_limit: int, size_limit: int, matching: bool):
    """
    Helper function for generating a random layer set

    NOTE: Randomized!

    :param layer_limit: The maximum amount of layers to generate
    :param size_limit: The maximum size of each layer
    :param matching: Specifies whether the amount of layers, and activation functions is equal
    :return: Two lists, the first containing the sizes of each layer, the second one the activation functions
    """

    layer_amount = random.randint(1, layer_limit)
    activation_amount = random.randint(1, layer_limit)

    while (
        (activation_amount != layer_amount)
        if matching
        else (activation_amount >= layer_amount)
    ):
        activation_amount = random.randint(1, layer_limit)
        layer_amount = random.randint(1, layer_limit)

    rnd_hidden = [random.randint(1, size_limit) for _ in range(layer_amount)]
    rnd_activation = ["ReLU"] * activation_amount

    return rnd_hidden, rnd_activation


def generate_parameters(iterations: int, matching: bool):
    """
    Helper function for generating multiple sets of layers

    :param iterations: The amount of layer sets generated
    :param matching: Specifies whether the amount of layers, and activation functions is equal
    :return: A list of layer sets, each consisting of hidden sizes and activation functions
    """

    rnd_parameters = [generate_layers(20, 2048, matching) for _ in range(iterations)]

    return rnd_parameters


def test_net():
    """
    Simply test if network creation works

    :return:
    """

    create_network()


def test_net_train():
    """
    Tests if the created network can be trained with random data

    :return:
    """

    net = create_network()
    test_tensor = torch.tensor([[0], [1]], dtype=torch.long)
    net(test_tensor)


def test_net_avg():
    """
    Tests if the averaging function of the network is correctly working

    :return:
    """

    net = create_network()
    test_tensor = torch.tensor([[2, 3]], dtype=torch.long)
    x = net.average_sentence(test_tensor)

    ten0 = torch.tensor([2], dtype=torch.long)
    ten1 = torch.tensor([3], dtype=torch.long)

    ten0 = net.embedding_layer(ten0)
    ten1 = net.embedding_layer(ten1)

    print(ten0)
    assert (ten0.data[0][0] + ten1.data[0][0]) / 2 == x.data[0][2]
    assert (ten0.data[0][1] + ten1.data[0][1]) / 2 == x.data[0][3]


def test_net_dim():
    """
    Tests if the dimensions of the averaging function are plausible.
    Because the averaging function looks like this:
    [fee, [word for word in sentence]] -> [[fee_embedding], [sentence_embedding (averaged)]]

    :return:
    """

    net = create_network()
    test_tensor = torch.tensor([[0, 2]], dtype=torch.long)
    x = net.average_sentence(test_tensor)

    ten = torch.tensor([0], dtype=torch.long)
    ten = net.embedding_layer(ten)

    assert len(ten.data[0]) * 2 == len(x.data[0])


@pytest.mark.parametrize("rnd_hidden, rnd_activation", generate_parameters(50, True))
def test_arbitrary_layers(rnd_hidden: List[int], rnd_activation: List[str]):
    """
    A test for checking the creation of random networks, with different random layers.

    NOTE: Randomized!

    :param rnd_hidden: A list of hidden sizes
    :param rnd_activation: A list of activation functions
    :return:
    """

    create_network(hidden_sizes=rnd_hidden, activation_functions=rnd_activation)


@pytest.mark.parametrize("rnd_hidden, rnd_activation", generate_parameters(50, False))
def test_unbalanced_layers(rnd_hidden: List[int], rnd_activation: List[str]):
    """
    Tests if passing different lengths of hidden_sizes and activation_functions causes an Exception.

    :return:
    """

    with pytest.raises(Exception):
        create_network(hidden_sizes=rnd_hidden, activation_functions=rnd_activation)


@pytest.mark.parametrize("activation", ACTIVATION_FUNCTIONS)
def test_arbitrary_activation(activation: List[str]):
    """
    Test for using different activation functions.

    :param activation: A list of activation functions
    :return:
    """

    create_network(activation_functions=[activation])


# TODO replace by own prediction tests
'''
@pytest.mark.parametrize("max_sentence_length", [random.randint(1, 50) for _ in range(10)])
def test_prediction_fee_only(max_sentence_length: int):
    """
    Tests the prediction of Frame Evoking Elements for raw text.

    NOTE: Randomized!
    :param max_sentence_length: The maximum sentence length of the generated raw text file
    :return:
    """

    with RandomFiles(max_sentence_length=max_sentence_length) as m_rndfiles:

        f_i = FrameIdentifier(ConfigManager())

        out_file = create_random_string()
        f_i.write_predictions(m_rndfiles.files[0], out_file, fee_only=True)

        m_rndfiles.files.append(out_file)
'''

@pytest.mark.parametrize("max_sentence_length", [random.randint(1, 50) for _ in range(10)])
def test_prediction_no_network(max_sentence_length: int):
    """
    Tests if an Exception is raised in case there is no network defined

    NOTE: Randomized!
    :param max_sentence_length: The maximum sentence length of the generated raw text file
    :return:
    """

    with RandomFiles(max_sentence_length=max_sentence_length) as m_rndfiles:

        f_i = FrameIdentifier(ConfigManager())

        out_file = create_random_string()

        with pytest.raises(Exception):
            f_i.write_predictions(m_rndfiles.files[0], out_file, fee_only=False)

        m_rndfiles.files.append(out_file)


@pytest.mark.parametrize("embedding_dim, num_classes", [(random.randint(1, 50), random.randint(1, 50)) for _ in range(10)])
def test_prediction_with_network(embedding_dim: int, num_classes: int):
    """
    Tests the prediction of Frames using a random network.

    NOTE: Randomized!
    :param embedding_dim: The dimensions of the embedding output of the network
    :param num_classes: The number of classes the network can predict
    :return:
    """

    with RandomFiles(max_sentence_length=random.randint(1, 50)) as m_rndfiles:

        with open(m_rndfiles.files[0]) as file:
            raw = file.read()
            word_count = sum([1 for line in raw.rsplit("\n") for _ in line.rsplit(" ")])

        cM = ConfigManager()
        embedding_layer = nn.Embedding(word_count, embedding_dim)

        f_i = FrameIdentifier(cM)
        f_i.network = FrameIDNetwork(cM, embedding_layer, num_classes)

        out_file = create_random_string()

        with pytest.raises(Exception):
            f_i.write_predictions(m_rndfiles.files[0], out_file, fee_only=False)

        m_rndfiles.files.append(out_file)
