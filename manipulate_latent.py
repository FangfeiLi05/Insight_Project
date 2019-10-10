import pickle
import config
import dnnlib
import dnnlib.tflib as tflib
import numpy as np
import PIL


#Load StyleGAN model to get StyleGAN generator
URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

def latent_to_image(latent_vector):
    #latent_vector.shape = (18,512)
    latent_vector = np.expand_dims(latent_vector ,axis=0)
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False))
    images = Gs_network.components.synthesis.run(latent_vector, randomize_noise=False, **synthesis_kwargs)
    images = images.transpose((0,2,3,1))[0]
    images_show = PIL.Image.fromarray(images, 'RGB').resize((256,256), PIL.Image.LANCZOS)
    return images_show


def latent_to_imageRGB(latent_vector):
    #latent_vector.shape = (18,512)
    latent_vector = np.expand_dims(latent_vector ,axis=0)
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False))
    images = Gs_network.components.synthesis.run(latent_vector, randomize_noise=False, **synthesis_kwargs)
    images = images.transpose((0,2,3,1))[0]
    return images


def tune_latent(latent_vector, direction, coeff, layers=list(range(0,18))):
    #latent_vector.shape = (18,512)
    new_latent_vector = latent_vector.copy()
    new_latent_vector[layers] = (latent_vector + coeff*direction)[layers]
    return new_latent_vector