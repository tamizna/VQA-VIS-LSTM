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
	#model_path = 'weights/model_1e10.h5'

	print('Loading questions ...')
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

	'''questions_train, questions_test = train_test_split(questions_traintest, test_size = 0.2, random_state = 0)
	answers_train, answer_test = train_test_split(answers_traintest, test_size=0.2, random_state = 0)
	img_features_train, img_features_test = train_test_split(img_features_traintest, test_size = 0.2, random_state = 0)'''

	X_train = [img_features_train, questions_train]
	X_val = [img_features_val, questions_val]
	EPOCHS = 10
	BATCH = 200

	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	history = model.fit(X_train,answers_train,
		epochs= EPOCHS,
		batch_size=BATCH,
		validation_data=(X_val,answers_val), verbose=1)

	#model.save(model_path)
	model.save_weights('weights/modele10b300_weights.h5')
	with open('weights/modele10b300_architecture.json', 'w') as f:
		f.write(model.to_json())
	
	# hitung nilai akurasi dan loss test
	#score = model.evaluate(X_test,answer_test)
	#print('Accuracy : ', score[1]*100, '%')
	#print('Loss : ', score[0]*100, '%')
	
	'''# plot nilai akurasi training & validasi
	acc = history.history['acc']
	val_acc = history.history['val_acc']
 
	epochs = range(len(acc))
 	
	plt.figure()
	plt.plot(epochs, acc, 'b', label='Training acc')
	plt.plot(epochs, val_acc, 'r', label='Validation acc')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.savefig('e2-acc.png')
	
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
	plt.savefig('e2-loss.png')'''

	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	N = EPOCHS
	
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), acc, label="train_acc")
	plt.plot(np.arange(0, N), val_acc, label="val_acc")
	plt.title("VQA Training & Validation Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("e10b300-acc.png")

	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), loss, label="train_loss")
	plt.plot(np.arange(0, N), val_loss, label="val_loss")
	plt.title("VQA Training & Validation Loss")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig("e10b300-loss.png")
		

if __name__ == '__main__':main()
