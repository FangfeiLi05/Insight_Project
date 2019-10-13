## Portrait-GANerator -- Deep Learning for Portrait Editing

This repository is for my 4-week project at [Insight Data Science](https://www.insightdatascience.com).

Portrait-GANerator is a portrait editing pipline that built based on a StyleGAN encoder ([@pbaylies](https://github.com/pbaylies/stylegan-encoder)) and latent space manipulation ([@SummitKwan](https://github.com/SummitKwan/transparent_latent_gan)).

* A Google Slides presenting the main ideas of this project is available [at this link](https://docs.google.com/presentation/d/1A2kYn3ROiRvGmY4l9Wl4ahF8fPFGcvkpsWgNYpymV4Y/edit#slide=id.g649c22c645_1_444).
* A blog post expaining more details about the motivation, analysis, and results will be posted soon.

<!-- An GUI demo will be demonstrated [at this link]() in Kaggle. -->  


![Alt text](./figures_readme/ganerator_pipline.png)


<!-- All the following in root directory -->

<!-- Tested on Nvidia K80 GPU with CUDA 9.0, with Anaconda Python 3.6
p2.xlarge: Ubuntu 16.04+k 80 GPU
NVIDIA K80 GPU -->


* Clone this repository.
  ```
  git clone https://github.com/FangfeiLi05/Insight_Project.git
  ``` 

<!--  to the root directory of the project (the folder containing the README.md)-->



* Train a ResNet with `train_resnet.py`, or a EfficientNet with `train_effnet.py`. This trained model will convert a image to a latent vector (18*512), which is used as the initial value in latent vector optimization in StyGAN encoder. You can also download a pre-trained ResNet [finetuned_resnet.h5](https://drive.google.com/open?id=12nM4KU7IBXGV5b5j1QV9f_3XQ2WmI8El) or a pre-trained EfficientNet [finetuned_effnet.h5](https://drive.google.com/open?id=12zWrGc3W0YuPANn3Rnl3OrNPskBO69fz), and put them in the folder `~/data/`.
  ```
  #python train_resnet.py --help
  #python train_effnet.py --help
      
  python train_resnet.py --test_size 256 --batch_size 1024 --loop 1 --max_patience 1
  python train_effnet.py --test_size 256 --batch_size 1024 --loop 1 --max_patience 1
  ``` 


* Align (center and crop) images with `align_images.py`. The aligned images are stored in the folder `~/images_aligned/`
  ```
  python align_images.py images_raw/ images_aligned/
  ```


* Convert each image in the folder `~/images_aligned/` into a latent vector with `encode_images.py`. The lantent representations are stored in the folder `~/images_latent/`. The reconstructed images are also outputted to a folder `~/images_reconstructed/`.
  ```
  python encode_images.py --batch_size=1 --output_video=True --load_effnet=data/finetuned_effnet --use_vgg_loss=1 images_aligned/ images_generate/ images_latent/
  ```


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

