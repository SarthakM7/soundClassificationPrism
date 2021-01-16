  
# way to upload audio
# way to save audio
# function to predict the output
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import muda

import librosa
import muda
import jams

import os 

#from scipy.misc import imresize

import sklearn as sk
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support

import pickle

from dcase_util.containers import MetaDataContainer

# from sed_tool.optimizers import DichotomicOptimizer
# from sed_tool.Encoder import Encoder
# from sed_tool.sed_tools import event_based_evaluation
# from sed_tool.sed_tools import eb_evaluator, sb_evaluator

import dcase_util as dcu

import tqdm
import keras 

from keras.models import model_from_json, load_model

from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, \
    Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Activation, TimeDistributed, \
    GRU, Reshape, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout2D, \
    Concatenate, Multiply
from keras import backend as K

import tensorflow as tf

# this is for the imports for flask app
import os
from flask import Flask, request, jsonify, render_template
import pickle
UPLOAD_FOLDER="/Users/hp/Desktop/samsung_prism_code"
app = Flask(__name__, template_folder='/Users/hp/Desktop/samsung_prism_code/API')

@app.route('/',methods=["GET","POST"])
def upload_predict():
	if request.method == "POST":
		audio_file=request.files["audio"]
		if audio_file:
			audio_location=os.path.join(UPLOAD_FOLDER,audio_file.filename)
			# TYPE YOUR MODEL HERE ######
			class_correspondance = {"Alarm_bell_ringing": 0, "Speech": 1, "Dog": 2, "Cat": 3, "Vacuum_cleaner": 4,
                        "Dishes": 5, "Frying": 6, "Electric_shaver_toothbrush": 7, "Blender": 8, "Running_water": 9}
			class_correspondance_reverse = dict()
			for k in class_correspondance:
				class_correspondance_reverse[class_correspondance[k]] = k
			def define_model(at_layer_name='at_output', loc_layer_name='loc_output'):

				time_pooling_factor=1
				
				input_shape = (64, 431, 1)
			 
				melInput = Input(input_shape)
			 
			    # ---- mel convolution part ----
				mBlock1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(melInput)
				mBlock1 = BatchNormalization()(mBlock1)
				mBlock1 = Activation(activation="relu")(mBlock1)
				mBlock1 = MaxPooling2D(pool_size=(4, 1))(mBlock1)
				# mBlock1 = Dropout(0.1)(mBlock1)
				mBlock1 = SpatialDropout2D(0.3, data_format=K.image_data_format())(mBlock1)
			
				mBlock2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(mBlock1)
				mBlock2 = BatchNormalization()(mBlock2)
				mBlock2 = Activation(activation="relu")(mBlock2)
				mBlock2 = MaxPooling2D(pool_size=(4, time_pooling_factor))(mBlock2)
				mBlock2 = SpatialDropout2D(0.3, data_format=K.image_data_format())(mBlock2)
				# mBlock2 = Dropout(0.1)(mBlock2)
				
				mBlock3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(mBlock2)
				mBlock3 = BatchNormalization()(mBlock3)
				mBlock3 = Activation(activation="relu")(mBlock3)
				mBlock3 = MaxPooling2D(pool_size=(4, time_pooling_factor))(mBlock3)
				mBlock3 = SpatialDropout2D(0.3, data_format=K.image_data_format())(mBlock3)
				# mBlock3 = Dropout(0.1)(mBlock3)
			 
				targetShape = int(mBlock3.shape[1] * mBlock3.shape[2])
				mReshape = Reshape(target_shape=(targetShape, 64))(mBlock3)
			
				gru = Bidirectional(
					GRU(kernel_initializer='glorot_uniform', activation='tanh', recurrent_dropout=0.1, \
						dropout=0.1, units=64, return_sequences=True, reset_after=False)
			    )(mReshape)
			 
				gru = Dropout(0.1)(gru)
			 
				output = TimeDistributed(
					Dense(64, activation="relu"),
				)(gru)
			 
				output = Dropout(0.1)(output)
			 
				loc_output = TimeDistributed(
				Dense(10, activation="sigmoid"), name=loc_layer_name,
				)(output)
			 

				gap = GlobalAveragePooling1D()(loc_output)
				gmp = GlobalMaxPooling1D()(loc_output)
			    # flat_gap = Flatten()(gap)
			    # flat_gmp = Flatten()(gmp)
			 
				concat = Concatenate()([gap, gmp])
			 
				d = Dense(1024, activation="relu")(concat)
				d = Dropout(rate=0.5)(d)
			 
				at_output = Dense(10, activation="sigmoid", name=at_layer_name)(d)
			 
				model = Model(inputs=[melInput], outputs=[loc_output, at_output])
				return model
			weight_path='dcase19.90-0.1658-0.3292.h5'

			model1 = define_model(at_layer_name='at_output1', loc_layer_name='loc_output1')
			model1.load_weights(weight_path)
			#fpath='AUD-20210114-WA0001.wav'
			fpath='10.wav'
			signal, sr = librosa.load(fpath, res_type='kaiser_fast')

			hop_length=512
			#             # multiply by random factor for data aug
			#             if self.fact_amp > 0:
			#                 print('amp')
			#                 signal *= rand_amp_arr[i]

			power = librosa.feature.melspectrogram(y=signal,
										sr=sr,
										n_fft=2048, 
										n_mels=64, 
										fmin=0.0, 
										fmax=sr/2.0, 
										htk=False, 
										hop_length=hop_length, 
										power=2.0, 
										norm=1)

			power = librosa.core.power_to_db(power, ref=np.max)
			endpoint_time = np.min([power.shape[1],431])

			x_test = power[:,:endpoint_time]
			x_test = x_test[np.newaxis, :, :, np.newaxis]
			# x_test.shape # (1, 64, 431, 1)

			loc_probs, at_probs = model1.predict(x_test)
			myprobs=at_probs*100
			np.set_printoptions(suppress=True)
			myprobs.flatten()
			probdict={}
			for i in range(10):
			  probdict[class_correspondance_reverse[i]]=myprobs[0][i]
			sorted_dict = dict( sorted(probdict.items(),
			                           key=lambda item: item[1],
			                           reverse=True))
			for key, value in sorted_dict.items():
			    brad={key, ' : ', value}

			return render_template('idex.html',prediction=sorted_dict)
	return render_template('idex.html',prediction=0)


app.run(debug=True)