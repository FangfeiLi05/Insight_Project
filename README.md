## Portrait-GANerator -- Deep Learning for Portrait Editing

This repository is for my 4 week project at [Insight Data Science](https://www.insightdatascience.com).

Portrait-GANerator is a deep learning portrait editing pipline built based on StyleGAN and latent space manipulation.

* A Google Slides presenting the main ideas of this project is available at the link [bit.ly/2nzoFBl](https://docs.google.com/presentation/d/1A2kYn3ROiRvGmY4l9Wl4ahF8fPFGcvkpsWgNYpymV4Y/edit#slide=id.g649c22c645_1_444).
* A blog post expaining more details about the motivation, analysis, and results will be posted soon.
* An GUI demo will be demonstrated in Kaggle soon.







### StyleGAN encoder (codes from [@pbaylies](https://github.com/pbaylies/stylegan-encoder))

* Train encoder - two types of encoder
    * Train a ResNet encoder with *train_resnet.py*, or [download a pre-trained model](https://drive.google.com/open?id=1tZLucJ1pZ8GA9JTRwF9d-Thr0zhR-i6l) and put it as *~/data/finetuned_resnet.h5*
      ```
      #python ./src/train_resnet.py --help
      #output '~/data/latent_feature_dataset.npy'
      
      python ./src/train_resnet.py --test_size 256 --batch_size 1024 --loop 10 --max_patience 1
      ```
    
    * Train a EfficientNet encoder with *train_effnet.py*, or [download a pre-trained model](https://drive.google.com/open?id=1LFTlv0RFo2zXz2GKVEYZDBRL7wFIj5Cc) and put it as *~/data/finetuned_effnet.h5*
      ```
      #python ./src/train_effnet.py --help
      
      python ./src/train_effnet.py --test_size 256 --batch_size 1024 --loop 10 --max_patience 1
      ```


* Align images - center and crop images with *align_images.py*
  ```
  mkdir images_aligned  #a folder is needed to store aligned images

  python ./src/align_images.py images_raw/ images_aligned/
  ```


* Encode images - encode images into latent representations ((18,512)), and reconstruct images using StyleGAN generator with *encode_images.py*
  * Encode images with ResNet
    ```
    #lantent representations are stored in folder 'images_latent', and reconstructed images are stored in folder 'images_reconstructed'
  
    python ./src/encode_images.py --batch_size=2 --output_video=True --load_resnet=data/finetuned_resnet.h5 images_aligned/ images_reconstructed/ images_latent/
    ```
  * Encode images with EfficientNet
    ```
    #lantent representations are stored in folder 'images_latent', and reconstructed images are stored in folder 'images_reconstructed'
  
    python ./src/encode_images.py --batch_size=2 --output_video=True --load_effnet=data/finetuned_effnet.h5 images_aligned/ images_reconstructed/ images_latent/
    ```


### Latent space manipulation (reference from [@SummitKwan](https://github.com/SummitKwan/transparent_latent_gan))
* Generate dataset - generate dataset cotaining the latent representation (20307, 18*512) and corresponding 23 features (20307, 23) of 20,307 images, with *latent_feature_dataset.py*. Features are *Gender, Age, Smile, EyeMakeup, LipMakeup, Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise, Beard, Moustache, Sideburns, Bald, BlondHair, BrownHair, BlackHair, RedHair, GrayHair, OtherHair*. Original dataset is from [original dataset](https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t). 
  ```
  #output '~/data/latent_feature_dataset.npy'
  
  python latent_feature_dataset.py
  ```


* Identify feature axes - train a logistic regression model (not sure what this model should be called) to get feature axes normalized and orthogonal with *feature_axes.py*, or [download a pre-trained model](https://drive.google.com/open?id=1G_a48GFl9SPgXKui5Z2aY-aH5gUi6sR2) and put it as *~/data/feature_axis.npy*
  ```
  #output '~/data/feature_axis.npy'
  
  python feature_axes.py
  ```

* Tune features - functions used to tune features continousely and show image
  ```
  import numpy as np
  import PIL
  from manipulate_latent import latent_to_imageRGB
  
  image_latent = np.load('./images_latent/000001_01.npy')
  
  image_array = latent_to_imageRGB(image_latent)
  PIL.Image.fromarray(image_array, 'RGB').resize((256,256), PIL.Image.LANCZOS)
  ```

  ```
  import numpy as np
  from manipulate_latent import latent_to_image
  
  image_latent = np.load('./images_latent/000001_01.npy')
  
  latent_to_image(image_latent)
  ```
  
  ```
  import numpy as np
  from manipulate_latent import latent_to_image
  from manipulate_latent import tune_latent
  
  feature_axis = np.load('./data/feature_axis.npy')
  i = 0
  direction = feature_axis[:,i].reshape((18, 512))
  coeff = 5
  
  image_latent_tuned = tune_latent(image_latent, direction, coeff, list(range(8))
  latent_to_image(image_latent_tuned)
  ```


### Results demonstration
* To be continued
* Requirements needed to install
* GUI

