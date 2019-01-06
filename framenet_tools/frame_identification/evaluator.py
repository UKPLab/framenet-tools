from framenet_tools.frame_identification.reader import Data_reader
from framenet_tools.frame_identification.fee_identifier import Fee_identifier

#Standard calculation for F1 score, taken from Open-SESAME
def calc_f(tp, fp, fn):
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

def evaluate_fee_identification(m_data_reader):
	gold_sentences = m_data_reader.annotations.copy()

	m_data_reader.predict_fees()

	tp = fp = fn = 0

	for gold_annotations, predictied_annotations in zip(gold_sentences, m_data_reader.annotations):
		for gold_annotation in gold_annotations:
			if gold_annotation.fee_raw in [x.fee_raw for x in predictied_annotations]:
				tp += 1
			else:
				fn += 1

		for predicted_annotation in predictied_annotations:
			if predicted_annotation not in [x.fee_raw for x in gold_annotations]:
				fp += 1

	print(tp, fp, fn)

	return calc_f(tp, fp, fn)

def evaluate_frame_identification():
	return None

def evaluate_fee_frame_identification():
	return None

train_file = ["../data/experiments/xp_001/data/train.sentences", "../data/experiments/xp_001/data/train.frame.elements"]
dev_file = ["../data/experiments/xp_001/data/dev.sentences", "../data/experiments/xp_001/data/dev.frames"]

m_data_reader = Data_reader()
m_data_reader.read_data(train_file[0], train_file[1])

f1 = evaluate_fee_identification(m_data_reader)
print(f1)

#print(m_data_reader.annotations[0].sentence)
#for i in range(10):
#	print(m_data_reader.annotations[i].fee_raw)
#fee_finder = Fee_identifier()