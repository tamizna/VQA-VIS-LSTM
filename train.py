import numpy as np
import prepare_data
import matplotlib.pyplot as plt
import models
import argparse
import sys

import embedding as ebd
import keras.backend as K
from nltk import word_tokenize
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from sklearn.model_selection import train_test_split

def main():
	K.set_image_dim_ordering('th')
	model_path = 'weights/model_1e10.h5'

	print('Loading questions ...')
	questions_traintest = prepare_data.get_questions_matrix('train')
	questions_val = prepare_data.get_questions_matrix('val')
	print('Loading answers ...')
	answers_traintest = prepare_data.get_answers_matrix('train')
	answers_val = prepare_data.get_answers_matrix('val')
	print('Loading image features ...')
	img_features_traintest = prepare_data.get_coco_features('train')
	img_features_val = prepare_data.get_coco_features('val')
	print('Creating model ...')
	
	model = models.vis_lstm()

	questions_train, questions_test = train_test_split(questions_traintest, test_size = 0.2, random_state = 0)
	answers_train, answer_test = train_test_split(answers_traintest, test_size=0.2, random_state = 0)
	img_features_train, img_features_test = train_test_split(img_features_traintest, test_size = 0.2, random_state = 0)

	X_train = [img_features_train, questions_train]
	X_val = [img_features_val, questions_val]
	X_test = [img_features_test, questions_test]

	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	history = model.fit(X_train,answers_train,
		epochs=10,
		batch_size=200,
		validation_data=(X_val,answers_val), verbose=1)

	model.save(model_path)
	
	# hitung nilai akurasi dan loss test
	score = model.evaluate(X_test,answer_test)
	print('Accuracy : ', score[1]*100, '%')
	print('Loss : ', score[0]*100, '%')
	
	# plot nilai akurasi training & validasi
	acc = history.history['acc']
	val_acc = history.history['val_acc']
 
	epochs = range(len(acc))+1
 	
	plt.figure()
	plt.plot(epochs, acc, 'b', label='Training acc')
	plt.plot(epochs, val_acc, 'r', label='Validation acc')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.savefig('accvaluese10.png')
	
	# plot nilai loss training & validasi
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	plt.figure()
	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.title('Training and validation loss')
	plt.legend()
	plt.savefig('lossvaluese10.png')
	

if __name__ == '__main__':main()
