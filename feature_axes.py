import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Activation
from keras.optimizers import SGD


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


""" Step 1: preparing for the training data: dlatent_array, feature_array """
dlatent_array = np.load('./data/latent_dataset.npy')
feature_array = np.load('./data/feature_dataset.npy')


""" Step 2: train logistic model to get feature direction """
model = Sequential()
model.add(Dense(feature_array.shape[1], activation='sigmoid'))
model.compile('adam', 'binary_crossentropy')

model.fit(dlatent_array, feature_array, validation_split=0.2, epochs=40)
model = Model(model.input, model.layers[-1].output)
#model.predict(dlatent_array[0:10])

feature_direction = model.layers[1].get_weights()[0]


""" Step 3: normalize and orthogonize feature_direction to get feature axes """
feature_axis = gram_schmidt(feature_direction)


np.save('./data/feature_axis.npy', feature_axis)
