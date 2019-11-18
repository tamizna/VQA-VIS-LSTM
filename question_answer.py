import numpy as np
import embedding as ebd
import prepare_data
import models
import argparse
import sys
import keras.backend as K
from nltk import word_tokenize
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def extract_image_features(img_path):
	model = models.VGG_16('weights/vgg16_weights_th_dim_ordering_th_kernels.h5')
	img = image.load_img(img_path,target_size=(224,224))
	x = image.img_to_array(img)
	x = np.expand_dims(x,axis=0)
	x = preprocess_input(x)
	last_layer_output = K.function([model.layers[0].input,K.learning_phase()],
		[model.layers[-1].input])
	features = last_layer_output([x,0])[0]
	return features

def preprocess_question(question):
	word_idx = ebd.load_idx()
	tokens = word_tokenize(question)
	seq = []
	for token in tokens:
		seq.append(word_idx.get(token,0))
	seq = np.reshape(seq,(1,len(seq)))
	return seq

def generate_answer(img_path, question):
	with open('weights/modele25_architecture.json', 'r') as f:
		model = model_from_json(f.read())
   	model.load_weights('weights/modele25_weights.h5')
	img_features = extract_image_features(img_path)
	seq = preprocess_question(question)
	x = [img_features, seq]
	probabilities = model.predict(x)[0]
	answers = np.argsort(probabilities[:1000])
	top_answers = [prepare_data.top_answers[answers[-1]]]
	
	return top_answers

def main():
	K.set_image_dim_ordering('th')

	print('---------------Testing---------------')
	image = "examples/children.jpg"
	question = "How many children in this image?"

	top_answers = generate_answer(image, question)

	# plot hasil uji
	fig, ax = plt.subplots()

	# title and labels, setting initial sizes
	fig.suptitle("Pertanyaan = {}\n 3 jawaban prediksi = {}, {}, {}".format(question, top_answers[0],top_answers[1],top_answers[2]), fontsize=12)
	img=mpimg.imread(image)
	imgplot = plt.imshow(img)
	plt.axis('off')
	plt.show()

if __name__ == '__main__':main()
