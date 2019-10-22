import numpy as np
import prepare_data
import models
import argparse
import sys

import embedding as ebd
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
	x = [img_features, seq]
	probabilities = model.predict(x)[0]
	answers = np.argsort(probabilities[:1000])
	top_answers = [prepare_data.top_answers[answers[-1]],
		prepare_data.top_answers[answers[-2]],
		prepare_data.top_answers[answers[-3]]]
	
	return top_answers


def main():
	K.set_image_dim_ordering('th')
	model_path = 'weights/model_1load.h5'

	"""print('Loading questions ...')
	questions_train = prepare_data.get_questions_matrix('train')
	questions_val = prepare_data.get_questions_matrix('val')
	print('Loading answers ...')
	answers_train = prepare_data.get_answers_matrix('train')
	answers_val = prepare_data.get_answers_matrix('val')
	print('Loading image features ...')
	img_features_train = prepare_data.get_coco_features('train')
	img_features_val = prepare_data.get_coco_features('val')
	print('Creating model ...')
	
	model = models.vis_lstm()
	X_train = [img_features_train, questions_train]
	X_val = [img_features_val, questions_val]

	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	model.fit(X_train,answers_train,
		epochs=10,
		batch_size=200,
		validation_data=(X_val,answers_val),
		verbose=1)

	model.save(model_path)"""
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
	

if __name__ == '__main__':main()

