from framenet_tools.frame_identification.frame_identifier import Frame_Identifier

from framenet_tools.paths import *
from framenet_tools.frame_identification.reader import Data_reader


# Standard calculation for F1 score, taken from Open-SESAME
def calc_f(tp: int, fp: int, fn: int):
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


def evaluate_fee_identification(files: list):
	m_data_reader = Data_reader()
	m_data_reader.read_data(files[0], files[1])

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


def evaluate_frame_identification(model: str, files: list):
	f_i = Frame_Identifier()
	f_i.load_model(model)

	tp, fp, fn = f_i.evaluate_file(files)
	print(tp, fp, fn)
	return calc_f(tp, fp, fn)


f1 = evaluate_fee_identification(DEV_FILES)
print(f1)

f1 = evaluate_frame_identification(SAVED_MODEL, DEV_FILES)
print(f1)