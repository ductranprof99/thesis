import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import argparse
from keras.models import model_from_json, load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import pdb


class QuadPredictor():
    def __init__(self, model_path, type_model, encoder):
        """Constructor method
        """
        # initial configuration
        self.max_output = False
        self.print_approx = True
        self.typeModel = type_model
        self.ind_to_label_quad = {0: 'Q1 (A+V+)', 1: 'Q2 (A+V-)', 2: 'Q3 (A-V-)', 3: 'Q4 (A-V+)'}
        self.ind_to_label_arou = {0: 'A-', 1: 'A+'}
        self.ind_to_label_vale = {0: 'V-', 1: 'V+'}
        self.sampling_rate = 16000
        path_model_load = os.path.join(model_path)

        # load models
        j_f, w_f = self.model_selector(path_model_load)
        self.model = self.load_pretrained_model(j_f, w_f)

        self.encoder = encoder
        
    def predict_sound(self,input_file):
        # extract spectrogram
        spec_array = self.create_spectrogram(input_file)
        print('*************\nCalculating output for file:', input_file)
        # predict!
        self.format_input = input_file.split('.')[-1]

#         out_file = os.path.join(out_dir, input_file.split('/')[-1].split('.')[0])

        return self.predict_and_save(self.model, spec_array)
       
      
    def model_selector(self, path):
        """ This method selects the weights and structure of the network
        """
        from os import listdir
        from os.path import isfile, join
        files = [ join(path, f) for f in listdir(path) if isfile(join(path, f))]


        if self.typeModel == "hdf5":
            weights_filename = [_ for _ in files if _.endswith('.hdf5')][0]
        else:
            print(files)
            weights_filename = [_ for _ in files if _.endswith('.h5')][0]
        json_filename = [_ for _ in files if _.endswith('.json')][0]
        return json_filename, weights_filename


    def load_pretrained_model(self, json_file, weight_file):
        """ This method loads the pretrained models, loads the 
        weights and adds the new layers"""
        # load model
        j_f = open(json_file, 'r')
        loaded_model = j_f.read()
        j_f.close()
        model = model_from_json(loaded_model)
        # load weights
        model = load_model(weight_file)
        return model
    
    def predict(self, sound_path):
        # Zero Crossing Rate
        def zcr(data, frame_length=2048, hop_length=512):
            zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
            return np.squeeze(zcr)
        #RMS Energy
        def rmse(data, frame_length=2048, hop_length=512):
            rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
            return np.squeeze(rmse)
        #MFCC
        def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
            mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
            return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

        def extract_features(data, sr, frame_length=2048, hop_length=512):
            result = np.array([])
            result = np.hstack((result,
                                zcr(data, frame_length, hop_length),
                                rmse(data, frame_length, hop_length),
                                mfcc(data, sr, frame_length, hop_length)
                                            ))
            return result

        def noise(data,noise_rate=0.015):
            noise_amp = noise_rate*np.random.uniform()*np.amax(data)
            data = data + noise_amp*np.random.normal(size=data.shape[0])
            return data

        def pitch(data, sampling_rate, pitch_factor=0.7):
            return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


        def get_features(path):
            #duration, offset 
            data, sample_rate = librosa.load(path, duration = 2.5)
            
            res1 = extract_features(data,sample_rate)
            result = np.array(res1)
            
            noise_data = noise(data)
            res2 = extract_features(noise_data,sample_rate)
            result = np.vstack((result, res2)) 
            
            data_pitch = pitch(data, sample_rate)
            res3 = extract_features(data_pitch,sample_rate)
            result = np.vstack((result, res3)) 
            
            data_noise_pitch = noise(data_pitch)
            res4 = extract_features(data_noise_pitch,sample_rate)
            result = np.vstack((result, res4)) 
            
            return result

        disassembly_data = get_features(sound_path)
        predict = self.model.predict(disassembly_data)
        return self.encoder.inverse_transform(predict)


def predict_predictor(model_path, mode ,features_path):
    # instanciate predictor
    Features = pd.read_csv(features_path)
    Y = Features['Emotions'].values
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
    return QuadPredictor(model_path, mode, encoder)


model = predict_predictor('./model/', 'h5' ,'./features.csv')

model.predict("./test/fear.wav")