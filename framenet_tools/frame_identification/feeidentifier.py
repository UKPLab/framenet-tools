import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree

# lemmatizer = WordNetLemmatizer()

# Static definitions
punctuation = (".", ",", ";", ":", "!", "?", "/", "(", ")", "'")  # added
forbidden_words = (
    "a",
    "an",
    "as",
    "for",
    "i",
    "in particular",
    "it",
    "of course",
    "so",
    "the",
    "with",
)
preceding_words_of = (
    "%",
    "all",
    "face",
    "few",
    "half",
    "majority",
    "many",
    "member",
    "minority",
    "more",
    "most",
    "much",
    "none",
    "one",
    "only",
    "part",
    "proportion",
    "quarter",
    "share",
    "some",
    "third",
)
following_words_of = ("all", "group", "their", "them", "us")
loc_preps = (
    "above",
    "against",
    "at",
    "below",
    "beside",
    "by",
    "in",
    "on",
    "over",
    "under",
)
temporal_preps = ("after", "before")
dir_preps = ("into", "through", "to")
forbidden_pos_prefixes = (
    "PR",
    "CC",
    "IN",
    "TO",
    "PO",
)  # added "PO": POS = genitive marker
direct_object_labels = ("OBJ", "DOBJ")  # accomodates MST labels and Standford labels


def get_pos_constants(tag: str):
    """
    Static function for tag conversion

    :param tag: The given pos tag
    :return: The corresponding letter
    """

    if tag.startswith("J"):
        return "a"
    elif tag.startswith("V"):
        return "v"
    elif tag.startswith("N"):
        return "n"
    elif tag.startswith("R"):
        return "r"
    else:
        return "n"


def should_include_token(p_data: list):
    """
    A static syntactical prediction of possible Frame Evoking Elements

    :param p_data: A list of lists containing token, pos_tag, lemma and NE
    :return: A list of possible FEEs
    """

    num_tokens = len(p_data)
    targets = []
    no_targets = []
    for idx in range(num_tokens):
        token = p_data[idx][0].lower().strip()
        pos = p_data[idx][1].strip()
        lemma = p_data[idx][2].strip()
        ne = p_data[idx][3].strip()
        if idx >= 1:
            precedingWord = p_data[idx - 1][0].lower().strip()
            preceding_pos = p_data[idx - 1][1].strip()
            preceding_lemma = p_data[idx - 1][2].strip()
            preceding_ne = p_data[idx - 1][3]
        if idx < num_tokens - 1:
            following_word = p_data[idx + 1][0].lower()
            following_pos = p_data[idx + 1][1]
            following_ne = p_data[idx + 1][3]
        if (
            token in forbidden_words
            or token in loc_preps
            or token in dir_preps
            or token in temporal_preps
            or token in punctuation
            or pos[: min(2, len(pos))] in forbidden_pos_prefixes
            or token == "course"
            and precedingWord == "of"
            or token == "particular"
            and precedingWord == "in"
        ):
            no_targets.append(token)
        elif token == "of":
            if (
                preceding_lemma in preceding_words_of
                or following_word in following_words_of
                or preceding_pos.startswith("JJ")
                or preceding_pos.startswith("CD")
                or following_pos.startswith("CD")
            ):
                targets.append(token)
            if following_pos.startswith("DT"):
                if idx < num_tokens - 2:
                    following_following_pos = p_data[idx + 2][1]
                    if following_following_pos.startswith("CD"):
                        targets.append(token)
            if (
                following_ne.startswith("GPE")
                or following_ne.startswith("LOCATION")
                or preceding_ne.startswith("CARDINAL")
            ):
                targets.append(token)
        elif token == "will":
            if pos == "MD":
                no_targets.append(token)
            else:
                targets.append(token)
        elif lemma == "be":
            no_targets.append(token)
        else:
            targets.append(token)
    # print("targets: " + str(targets))
    # print("NO targets:\n" + str(notargets))
    return targets


class FeeIdentifier(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def identify_targets(self, sentence: list):
        """
        Identifies targets for a given sentence

        :param sentence: A list of words in a sentence
        :return: A list of targets
        """

        tokens = nltk.word_tokenize(sentence)
        p_data = self.get_tags(tokens)
        targets = should_include_token(p_data)

        return targets

    def get_tags(self, tokens: list):
        """
        Gets lemma, pos and NE for each token

        :param tokens: A list of tokens from a sentence
        :return: A 2d-Array containing lemma, pos and NE for each token
        """

        pos_tags = []
        lemmas = []
        nes = []
        # print(tokens)
        tags = nltk.pos_tag(tokens)
        for tag in tags:
            pos_tags.append(tag[1])
            lemmas.append(
                self.lemmatizer.lemmatize(tag[0], pos=get_pos_constants(tag[1]))
            )
        chunks = nltk.ne_chunk(tags)
        for chunk in chunks:
            if isinstance(chunk, Tree):
                nes.append(chunk.label())
            else:
                nes.append("-")
        pData = []
        for t, p, l, n in zip(tokens, pos_tags, lemmas, nes):
            pData.append([t, p, l, n])

        return (pData, pos_tags, lemmas)

    """
    def load_dataset(self, file):
        reader = Data_reader(file[0], file[1])
        reader.read_data()
        dataset = reader.get_dataset()

        return self.sum_FEEs(dataset)
    """

    def query(self, x: list):
        """
        Query a prediction of FEEs for a given sentence

        :param x: A list of words in a sentence
        :return: A list of predicted FEEs
        """

        tokens = x[0]

        pData, postags, lemmas = self.get_tags(tokens)
        possible_fees = should_include_token(pData)

        return possible_fees

    def predict_fees(self, dataset: list):
        """
        Predicts all FEEs for a complete datset

        :param dataset: The dataset to predict
        :return: A list of predictions
        """
        predictions = []

        for data in dataset:
            prediction = self.query(data)
            predictions.append(prediction)

        return predictions

    def evaluate_acc(self, dataset: list):
        """
        Evaluates the accuracy of the Frame Evoking Element Identifier

        NOTE: F1-Score is a better way to evaluate the Identifier, because it tends to predict too many FEEs

        :param dataset: The dataset to evaluate
        :return: A Triple of the count of correct elements, total elements and the accuracy
        """
        correct = 0
        total = 0

        for data in dataset:
            predictions = self.query(data)

            total += len(data[1])

            for prediction in predictions:
                if prediction in data[1]:
                    correct += 1

        acc = correct / total

        return correct, total, acc
