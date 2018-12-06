
class Data_reader(object):

	def __init__(self, path_sent ,path_elements):
		self.loaded = False
		self.i = 0
		self.path_sent = path_sent
		self.path_elements = path_elements
		self.dataset = []

	def digest_rawdata(self, elements, sentences):

		elements_line = 0

		
		for i in range(len(sentences)):
			#Next sentence

			words = sentences[i].split(" ")

			same_sentence = True

			while same_sentence:

				#Reached EOF
				if(elements_line >= len(elements)):
					break

				#New Frame
				frame_des = elements[elements_line].split("\t")
				frame = frame_des[3] #Frame
				fee = frame_des[4] #Frame evoking element
				position = frame_des[5] #Position of word in sentence
				fee_raw = frame_des[6] #Frame evoking element as it appeared
				sent_num = frame_des[7] #Sentence number
				
				
				if not i == int(sent_num):
					#Next sentence
					#print("unequal")
					same_sentence = False
				else:
					self.dataset.append([words, frame, fee, position, fee_raw, sent_num])	
					#Shift for next iteration
					elements_line += 1				





	def read_data(self):

		file = open(self.path_sent, "r")
		sentences = file.read()
		file.close()

		file = open(self.path_elements, "r")
		elements = file.read()
		file.close()

		sentences = sentences.split("\n")
		elements = elements.split("\n")
		
		#Remove empty line at the end
		if elements[len(elements)-1] == "":
			print("Removed empty line at eof")
			elements = elements[:len(elements)-1]
		
		if sentences[len(sentences)-1] == "":
			print("Removed empty line at eof")
			sentences = sentences[:len(sentences)-1]

		#print(sentences)

		self.digest_rawdata(elements, sentences)

	def get_dataset(self):
		return self.dataset







