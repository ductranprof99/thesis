import librosa, librosa.feature
import numpy as np
import tensorflow.keras as keras


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def extract_features(data,sample_rate=22050):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path)
    
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    
    return result

    

def load_audio(audio_file_path):
    X = []
    feature = get_features(audio_file_path)
    for ele in feature:
        X.append(ele)
    return np.array(X).reshape((1,162,1))


labels = ['angry', 'calm' ,'disgust', 'fear', 'happy', 'neutral', 'sad','suprise']

# MODEL = keras.models.load_model('./model/Emotion_Voice_Detection_Model.h5')

def sound_to_emotion(audio_path):
    '''
    predict emotion from voice speech with exits audio file
    '''
    audio = load_audio(audio_path)
    predictions = MODEL.predict(audio,)
    return dict(zip(labels,predictions[0]))

class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self):
        """
        Init method is used to initialize the main parameters.
        """
        self.path = './model/Emotion_Voice_Detection_Model.h5'
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self,file_path):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(file_path,res_type='kaiser_fast')
        x = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0).reshape(1,40,1)
        # x = np.expand_dims(mfccs, axis=2)
        # x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict(x)
        print(predictions)
        # print( "Prediction is", " ", self.convert_class_to_emotion(predictions))

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'happy',
                            '2': 'sad',
                            '3': 'angry',
                            '4': 'fearful',
                            '5': 'disgust',
                            '6': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label

predictor = LivePredictions()
predictor.make_predictions('./test/happy.wav')
# data, sampling_rate = librosa.load('./test/happy.wav')
# mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0).reshape(1,40,1)
# # x = np.expand_dims(mfccs, axis=2)
# print(mfccs)
# x = np.expand_dims(x, axis=0)

