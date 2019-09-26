## Deep Learning Portrait Editing using StyleGAN and Latent Space Manipulation


### StyleGAN encoder (from [@pbaylies](https://github.com/pbaylies/stylegan-encoder))

* Train encoder - two types of encoder
    * Train a ResNet encoder with *train_resnet.py*, or [download a pre-trained model](https://drive.google.com/open?id=1tZLucJ1pZ8GA9JTRwF9d-Thr0zhR-i6l) and put it as *~/data/finetuned_resnet.h5*
      ```
      #python ./src/train_resnet.py --help

      python ./src/train_resnet.py --test_size 256 --batch_size 1024 --loop 10 --max_patience 1
      ```
    
    * Train a EfficientNet encoder with *train_effnet.py*, or [download a pre-trained model](https://drive.google.com/open?id=1LFTlv0RFo2zXz2GKVEYZDBRL7wFIj5Cc) and put it as *~/data/finetuned_effnet.h5*
      ```
      #python ./src/train_effnet.py --help
      
      python ./src/train_effnet.py --test_size 256 --batch_size 1024 --loop 10 --max_patience 1
      ```


* Align images - center and crop images with *align_images.py*
    ```
    mkdir images_aligned_test  #a folder is needed to store aligned images

    python ./src/align_images.py images_raw_test/ images_aligned_test/
    ```


* Encode images - encode images into latent representations ((18,512)), and reconstruct images using StyleGAN generator with *encode_images.py*
    ```
    #lantent representations are stored in the folder 'images_latent_test', and reconstructed images are stored in the folder 'images_reconstructed_test'
    
    python ./src/encode_images.py --batch_size=2 --output_video=True --load_effnet=data/finetuned_effnet.h5 images_aligned_test/ images_reconstructed_test/ images_latent_test/
    ```


### Latent space manipulation (from [@SummitKwan](https://github.com/SummitKwan/transparent_latent_gan))
* Prepare for dataset - [download the dataset](https://drive.google.com/open?id=161rQuFYWObxNrzcKoI1bDp9eRmBZFLlg) and put it as *~/data/latent_dataset.npy*. The dataset cotains the latent representation ((9261,1)) and corresponding feature vector ((23,1)) of 20,307 images. The 23 features are *Gender, Age, Smile, EyeMakeup, LipMakeup, Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise, Beard, Moustache, Sideburns, Bald, BlondHair, BrownHair, BlackHair, RedHair, GrayHair, OtherHair*. The dataset is obtained from [original dataset](https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t) using *latent_dataset.py*.
    ```
    #output 'latent_dataset.npy'
    python ./src/latent_dataset.py
    ```

* Identify feature axes - train a logistic regression model to get feature directions for all features simultatiously in latent space, and further make these feature directions normalized and orthogonal with *???.py*
    ```
    #output '???.npy'
    python ./src/???.py
    ```

* Tune features - tune features continousely

* To be continued

