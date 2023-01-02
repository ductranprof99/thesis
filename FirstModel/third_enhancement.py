import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input
import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_final_model_data(data_path):
    """
    Loads the data from the final model
    """
    result = []
    with open(data_path, 'r') as f:
        data = json.load(f)
    for i in data:
        result.append(json.loads(i))
    return result


class NormalClassifier:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Input(shape=(7)))
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dense(units=7, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    def train(self, x_train, y_train):
        result   = self.model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
        self.history = result.history

    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = keras.models.load_model(path)



data = load_final_model_data('./last_network_input.json')
train, test = train_test_split(data, test_size=0.2)

label_output = [[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]

def reform_data(input_list):
    x,y = [],[]
    for i in input_list:
        text_vector = np.array(list(json.loads(json.dumps(i['text_vector'], sort_keys=True)).values()))
        
        sound_vector = np.array(list(json.loads(json.dumps(i['sound_vector'], sort_keys=True)).values()))
        x.append(list(text_vector*0.3 + sound_vector*0.7))
        y.append(label_output[i['label']])
    return x,y
        
    
x_train, y_train = reform_data(train)
x_test, y_test = reform_data(test)
    
predictor = NormalClassifier()
predictor.train(x_train, y_train)
# print('test loss:' + str(score[0]))
# print('test accuracy:' + str(score[1]))

predictor.save('./last_network.h5')

plt.plot(predictor.history['loss'])
plt.plot(predictor.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('final_network_loss.png')
plt.close()
# Accuracy plotting
plt.plot(predictor.history['accuracy'])
plt.plot(predictor.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('final_network_accuracy.png')


