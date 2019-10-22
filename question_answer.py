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

def generate_answer(img_path, question, model):
	img_features = extract_image_features(img_path)
	seq = preprocess_question(question)
	if model == 1:
		x = [img_features, seq]
	else:
		x = [img_features, seq, img_features]
	probabilities = model.predict(x)[0]
	answers = np.argsort(probabilities[:1000])
	top_answers = [prepare_data.top_answers[answers[-1]],
		prepare_data.top_answers[answers[-2]],
		prepare_data.top_answers[answers[-3]]]
	
	return top_answers

def main():
	"""parser = argparse.ArgumentParser()
	parser.add_argument('-image', type=str, required=True)
	parser.add_argument('-question', type=str, required=True)
	args = parser.parse_args()"""

	K.set_image_dim_ordering('th')
	model_path = 'weights/model_1load.h5'

	print('---------------Testing---------------')
	model = models.vis_lstm()
	model.load_weights(model_path)
	image = "examples/COCO_val2014_000000000136.jpg"
	question = "Which animal is this?"
	print(question)
	top_answers = generate_answer(image, question, model)
	print('Top answers: %s, %s, %s.' %(top_answers[0],top_answers[1],top_answers[2]))

	image = "examples/COCO_val2014_000000000073.jpg"
	question = "What is this?"
	print(question)
	top_answers = generate_answer(image, question, model)
	print('Top answers: %s, %s, %s.' %(top_answers[0],top_answers[1],top_answers[2]))

	image = "examples/COCO_val2014_000000000073.jpg"
	question = "How many bicycle in this image?"
	print(question)
	top_answers = generate_answer(image, question, model)
	print('Top answers: %s, %s, %s.' %(top_answers[0],top_answers[1],top_answers[2]))
	
	
	"""top_answers = generate_answer(args.image, args.question, args.model)
	print('Top answers: %s, %s, %s.' % (top_answers[0],top_answers[1],top_answers[2]))"""

if __name__ == '__main__':main()
