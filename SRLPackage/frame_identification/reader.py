
class Data_reader(object):

	def __init__(self, path_sent ,path_elements):
		self.loaded = False
		self.i = 0
		self.path_sent = path_sent
		self.path_elements = path_elements

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

		elements_line = 0
		frame_des = elements[elements_line].split("\t")

		
		for i in range(len(sentences)):
			print("Next sentence!")

			words = sentences[i].split(" ")

			same_sentence = True

			while same_sentence:

				print("Found new Frame!")
				frame_des[3] #Frame
				frame_des[4] #FEE
				
				elements_line += 1
				
				if(elements_line >= len(elements)):
					break

				frame_des = elements[elements_line].split("\t")

				print(i)
				print(frame_des[len(frame_des)-1])

				if not i == int(frame_des[len(frame_des)-1]):
					print("unequal")
					same_sentence = False





