#feature_axis.py: identify the normalized and orthogonalized feature axes in the latent space.
import numpy as np
import pandas as pd
import dnnlib
import config
import pickle
import gzip
import os
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Activation
from keras.optimizers import SGD, Adam


def gram_schmidt(vectors):
  #columns correspond to feature directions
    basis = []
    vectors = np.array(vectors, dtype=float)
    for v in vectors.T:
        tempt = [np.dot(v,b)*b for b in basis]
        w = v - np.sum(tempt, axis = 0)
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis).T


def main():
    """ Step 1: preparing for the training data: dlatent_array, feature_array """
    LATENT_TRAINING_DATA = 'https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t'
    with dnnlib.util.open_url(LATENT_TRAINING_DATA, cache_dir=config.cache_dir) as f:
        qlatent_data, dlatent_data, labels_data = pickle.load(gzip.GzipFile(fileobj=f))

    feature_dict = dict()
    feature_label_list = ['gender', 'makeup', 'glasses', 'age', 'smile']
    for feature_label in feature_label_list:
        if feature_label == 'gender':
            feature_dict[feature_label] = np.array([x['faceAttributes'][feature_label] == 'male' for x in labels_data])
        elif feature_label == 'makeup':
            feature_dict[feature_label] = np.array([x['faceAttributes'][feature_label]['eyeMakeup'] for x in labels_data]) + np.array([x['faceAttributes'][feature_label]['lipMakeup'] for x in labels_data])
        elif feature_label == 'glasses':
            feature_dict[feature_label] = np.array([x['faceAttributes']['glasses'] == 'ReadingGlasses' for x in labels_data]) + np.array([x['faceAttributes']['glasses'] == 'Sunglasses' for x in labels_data]) + np.array([x['faceAttributes']['glasses'] == 'SwimmingGoggles' for x in labels_data])
        else:
            feature_dict[feature_label] = np.array([x['faceAttributes'][feature_label] for x in labels_data])

    feature_label_list_emotion = ['anger', 'contempt', 'disgust', 'fear', 'neutral', 'sadness', 'surprise']
    for feature_label in feature_label_list_emotion:
        feature_dict[feature_label] = np.array([x['faceAttributes']['emotion'][feature_label] for x in labels_data])

    #feature_label_list_facialHair = ['beard', 'moustache', 'sideburns']
    #for feature_label in feature_label_list_facialHair:
    #    feature_dict[feature_label] = np.array([x['faceAttributes']['facialHair'][feature_label] for x in labels_data])
    feature_dict['beard'] = np.array([x['faceAttributes']['facialHair']['beard'] for x in labels_data])
    feature_dict['bald'] = np.array([x['faceAttributes']['hair']['bald'] for x in labels_data])

    num_example = len(qlatent_data)
    feature_label_list_hairColor = ['blond', 'black']
    for feature_label in feature_label_list_hairColor:
        feature_dict[feature_label+'Hair'] = np.full(num_example, True, dtype = bool)
        for i in range(num_example):
            tempt = labels_data[i]['faceAttributes']['hair']['hairColor']
            if tempt:
                feature_dict[feature_label+'Hair'][i] = [x['confidence'] for x in tempt if x['color'] == feature_label][0] >= 0.95
    
         
    feature_DataFrame = pd.DataFrame(feature_dict)
    feature_array = np.array(feature_DataFrame, dtype = float)
    feature_array = (feature_array - np.min(feature_array, axis=0))/(np.max(feature_array, axis=0) - np.min(feature_array, axis=0))
    dlatent_array = dlatent_data.reshape((-1, 18*512))

    

    
    """ Step 2: train a logistic model to get feature direction """
    model = Sequential()
    model.add(Dense(feature_array.shape[1], activation='sigmoid'))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
    sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd)
    model.fit(dlatent_array, feature_array, validation_split=0.3, epochs=100)
    model = Model(model.input, model.layers[-1].output)
    #model.predict(dlatent_array[0:10])
    feature_direction = model.layers[1].get_weights()[0]


    

    """ Step 3: normalize and orthogonize feature_direction to get feature axes """
    feature_axis = gram_schmidt(feature_direction)
    #feature_axis = feature_direction
    
    
    
    
    """ Step 4: save the results """
    feature_label_list = list(feature_DataFrame.columns)
    feature_axis_dict = dict()
    for i in range(len(feature_label_list)):
        feature_axis_dict[feature_label_list[i]] = feature_axis[:, i]
    
    os.makedirs('data', exist_ok=True)
    feature_axis_DataFrame = pd.DataFrame(feature_axis_dict)
    feature_axis_DataFrame.to_hdf('./data/feature_axis.h5', key='df', mode='w')


if __name__ == "__main__":
    main()