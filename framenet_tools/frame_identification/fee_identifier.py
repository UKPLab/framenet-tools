import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree

from framenet_tools.frame_identification.reader import Data_reader

lemmatizer = WordNetLemmatizer()

punctuation = (".", ",", ";", ":", "!", "?", "/", "(", ")", "'") # added
forbidden_words = ("a", "an", "as", "for", "i", "in particular", "it", "of course", "so", "the", "with")
preceding_words_of = ("%", "all", "face", "few", "half", "majority", "many", "member", "minority", "more", "most", "much", "none", "one", "only", "part", "proportion", "quarter", "share", "some", "third")
following_words_of = ("all", "group", "their", "them", "us")
loc_preps = ("above", "against", "at", "below", "beside", "by", "in", "on", "over", "under")
temporal_preps = ("after", "before")
dir_preps = ("into", "through", "to")
forbidden_pos_prefixes = ("PR", "CC", "IN", "TO", "PO") # added "PO": POS = genitive marker
direct_object_labels = ("OBJ", "DOBJ")    # accomodates MST labels and Standford labels

class fee_identifier(object):

    #Standard calculation for F1 score, taken from Open-SESAME
    def calc_f(self, tp, fp, fn):
        if tp == 0.0 and fp == 0.0:
            pr = 0.0
        else:
            pr = tp / (tp + fp)
        if tp == 0.0 and fn == 0.0:
            re = 0.0
        else:
            re = tp / (tp + fn)
        if pr == 0.0 and re == 0.0:
            f = 0.0
        else:
            f = 2.0 * pr * re / (pr + re)
        return pr, re, f

    def identify_targets(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        pData = getTags(tokens)
        targets = shouldIncludeToken(pData)
        return(targets)
        

    # gets a sentence in form of a list of tokens
    # returns 2d array containing lemma, pos and NE for each token
    def getTags(self, tokens):
        postags = []
        lemmas = []
        nes = []
        tags = nltk.pos_tag(tokens)
        for tag in tags:
            postags.append(tag[1])
            lemmas.append(lemmatizer.lemmatize(tag[0], pos=self.get_pos_constants(tag[1])))
        chunks = nltk.ne_chunk(tags)
        for chunk in chunks:
            if isinstance(chunk, Tree):
                nes.append(chunk.label())
            else:
                nes.append("-")
        pData = []
        for t, p, l, n in zip(tokens, postags, lemmas, nes):
            pData.append([t, p, l, n])
        
        return(pData, postags, lemmas)

    def get_pos_constants(self, tag):
        if tag.startswith('J'):
            return 'a'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('N'):
            return 'n'
        elif tag.startswith('R'):
            return 'r'
        else:
            return 'n'


    # pData: list of lists containing token, pos tag, lemma, NE
    # def shouldIncludeToken(idxs, pData, mNodeList):
    def shouldIncludeToken(self, pData):
        numTokens = len(pData)
        targets = []
        notargets = []
        for idx in range(numTokens):
            token = pData[idx][0].lower().strip()
            pos = pData[idx][1].strip()
            lemma = pData[idx][2].strip()
            ne = pData[idx][3].strip()
            if idx >= 1:
                precedingWord = pData[idx-1][0].lower().strip()
                precedingPOS = pData[idx-1][1].strip()
                precedingLemma = pData[idx-1][2].strip()
                precedingNE = pData[idx-1][3]
            if idx < numTokens - 1:
                followingWord = pData[idx+1][0].lower()
                followingPOS = pData[idx+1][1]
                followingNE = pData[idx+1][3]
            if token in forbidden_words or token in loc_preps or token in dir_preps or token in temporal_preps or token in punctuation or pos[:min(2, len(pos))] in forbidden_pos_prefixes or token == "course" and precedingWord == "of" or token == "particular" and precedingWord == "in":
                notargets.append(token)
            elif token == "of":
                if precedingLemma in preceding_words_of or followingWord in following_words_of or precedingPOS.startswith("JJ") or precedingPOS.startswith("CD") or followingPOS.startswith("CD"):
                    targets.append(token)
                if followingPOS.startswith("DT"):
                    if idx < numTokens - 2:
                        followingFollowingPOS = pData[idx+2][1]
                        if followingFollowingPOS.startswith("CD"):
                            targets.append(token)
                if followingNE.startswith("GPE") or followingNE.startswith("LOCATION") or precedingNE.startswith("CARDINAL"):
                    targets.append(token)
            elif token == "will":
                if pos == "MD":
                    notargets.append(token)
                else:
                    targets.append(token)
            elif lemma == "be":
                notargets.append(token)
            else:
                targets.append(token)
        #print("targets: " + str(targets))
        #print("NO targets:\n" + str(notargets))
        return(targets)

    def sum_FEEs(self, dataset):

        last_sentence = -1
        dataset_FEE = []


        for data in dataset:
            cur_sentence = int(data[5])

            if cur_sentence == last_sentence:
                dataset_FEE[last_sentence][1].append(data[4])
            else:
                #print("new sent: " + str(last_sentence))
                last_sentence = cur_sentence
                dataset_FEE.append([data[0],[data[4]]])

        return dataset_FEE



    def load_dataset(self, file):
        reader = Data_reader(file[0], file[1])
        reader.read_data()
        dataset = reader.get_dataset()

        return self.sum_FEEs(dataset)

    def query(self, x):
        tokens = x[0]

        pData, postags, lemmas = self.getTags(tokens)
        posible_FEEs = self.shouldIncludeToken(pData)
        
        return posible_FEEs

    def predict_FEEs(self, dataset):
        predictions = []

        for data in dataset:
            prediction = self.query(data)
            predictions.append(prediction)

        return predictions

    def evaluate_f1(self, dataset):
        tp = 0
        fp = 0
        fn = 0

        for data in dataset:
            predictions = self.query(data)

            for gold_fee in data[1]:
                if gold_fee in predictions:
                    tp += 1
                else:
                    fn += 1

            for prediction in predictions:
                if prediction not in data[1]:
                    fp += 1

        print(tp, fp, fn)

        return self.calc_f(tp, fp, fn)

    def evaluate_acc(self, dataset):
        correct = 0
        total = 0

        for data in dataset:
            predictions = self.query(data)

            total += len(data[1])

            for prediction in predictions:
                if prediction in data[1]:
                    correct += 1

        acc = correct/total

        return correct, total, acc



#train_file = ["../data/experiments/xp_001/data/train.sentences", "../data/experiments/xp_001/data/train.frame.elements"]
#dev_file = ["../data/experiments/xp_001/data/dev.sentences", "../data/experiments/xp_001/data/dev.frames"]

#fee_finder = fee_identifier()

#dataset_FEE = fee_finder.load_dataset(dev_file)
#pred = fee_finder.predict_FEEs(dataset_FEE)
#print(pred[0])
#print(len(pred))
#print(fee_finder.evaluate_f1(dataset_FEE))

#example = "The leaves are falling in Germany's forests."
#tokens = nltk.word_tokenize(example)
#print(tokens)
#pData, postags, lemmas = getTags(tokens)
#t = shouldIncludeToken(pData)
#print(t)
