import numpy as np
import dnnlib as dnnlib
import config as config
import pickle
import gzip

LATENT_TRAINING_DATA = 'https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t'
with dnnlib.util.open_url(LATENT_TRAINING_DATA, cache_dir=config.cache_dir) as f:
    qlatent_data, dlatent_data, labels_data = pickle.load(gzip.GzipFile(fileobj=f))
    

dlatent_array = dlatent_data.reshape((-1, 18*512))

feature_gender = np.array([x['faceAttributes']['gender'] == 'male' for x in labels_data])
feature_age = np.array([x['faceAttributes']['age'] for x in labels_data])
feature_smile = np.array([x['faceAttributes']['smile'] for x in labels_data])

feature_eyemakeup = np.array([x['faceAttributes']['makeup']['eyeMakeup'] for x in labels_data])
feature_lipmakeup = np.array([x['faceAttributes']['makeup']['lipMakeup'] for x in labels_data])

feature_anger = np.array([x['faceAttributes']['emotion']['anger'] for x in labels_data])
feature_contempt = np.array([x['faceAttributes']['emotion']['contempt'] for x in labels_data])
feature_disgust = np.array([x['faceAttributes']['emotion']['disgust'] for x in labels_data])
feature_fear = np.array([x['faceAttributes']['emotion']['fear'] for x in labels_data])
feature_happiness = np.array([x['faceAttributes']['emotion']['happiness'] for x in labels_data])
feature_neutral = np.array([x['faceAttributes']['emotion']['neutral'] for x in labels_data])
feature_sadness = np.array([x['faceAttributes']['emotion']['sadness'] for x in labels_data])
feature_surprise = np.array([x['faceAttributes']['emotion']['surprise'] for x in labels_data])

feature_beard = np.array([x['faceAttributes']['facialHair']['beard'] for x in labels_data])
feature_moustache = np.array([x['faceAttributes']['facialHair']['moustache'] for x in labels_data])
feature_sideburns = np.array([x['faceAttributes']['facialHair']['sideburns'] for x in labels_data])
feature_bald = np.array([x['faceAttributes']['hair']['bald'] for x in labels_data])

num_example = len(qlatent_data)
feature_blondhair = np.zeros(num_example)
feature_brownhair = np.zeros(num_example)
feature_blackhair = np.zeros(num_example)
feature_redhair = np.zeros(num_example)
feature_grayhair = np.zeros(num_example)
feature_otherhair = np.zeros(num_example)
for i in range(num_example):
    tempt = labels_data[i]['faceAttributes']['hair']['hairColor']
    if tempt:
        feature_blondhair[i] = [x['confidence'] for x in tempt if x['color'] == 'blond'][0] >= 0.95
        feature_brownhair[i] = [x['confidence'] for x in tempt if x['color'] == 'brown'][0] >= 0.95
        feature_blackhair[i] = [x['confidence'] for x in tempt if x['color'] == 'black'][0] >= 0.95
        feature_redhair[i] = [x['confidence'] for x in tempt if x['color'] == 'red'][0] >= 0.95
        feature_grayhair[i] = [x['confidence'] for x in tempt if x['color'] == 'gray'][0] >= 0.95
        feature_otherhair[i] = [x['confidence'] for x in tempt if x['color'] == 'other'][0] >= 0.95


feature_array = np.zeros((num_example, 23), dtype = float)
feature_array[:,0] = feature_gender
feature_array[:,1] = feature_age
feature_array[:,2] = feature_smile

feature_array[:,3] = feature_eyemakeup
feature_array[:,4] = feature_lipmakeup

feature_array[:,5] = feature_anger
feature_array[:,6] = feature_contempt
feature_array[:,7] = feature_disgust
feature_array[:,8] = feature_fear
feature_array[:,9] = feature_happiness
feature_array[:,10] = feature_neutral
feature_array[:,11] = feature_sadness
feature_array[:,12] = feature_surprise

feature_array[:,13] = feature_beard
feature_array[:,14] = feature_moustache
feature_array[:,15] = feature_sideburns
feature_array[:,16] = feature_bald

feature_array[:,17] = feature_blondhair
feature_array[:,18] = feature_brownhair
feature_array[:,19] = feature_blackhair
feature_array[:,20] = feature_redhair
feature_array[:,21] = feature_grayhair
feature_array[:,22] = feature_otherhair

feature_array_normalized = (feature_array - np.min(feature_array, axis=0))/(np.max(feature_array, axis=0) - np.min(feature_array, axis=0))

latent_dataset = {'latent_representation': dlatent_array, 'feature_vector':feature_array_normalized}
np.save('./data/latent_dataset.npy', latent_dataset)