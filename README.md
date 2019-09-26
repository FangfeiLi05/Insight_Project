## Deep Learning Portrait Editing using StyleGAN and Latent Space Manipulation


### StyleGAN Encoder (from @pbaylies)

* Training encoder - two types of encoder
    * Train a ResNet encoder with *train_resnet.py*, or [download a pre-trained model](https://drive.google.com/open?id=1tZLucJ1pZ8GA9JTRwF9d-Thr0zhR-i6l) and put it as *~/data/finetuned_resnet.h5*
      ```
      python ./src/train_resnet.py --test_size 256 --batch_size 1024 --loop 10 --max_patience 1
      ```
    
    * Train a EfficientNet encoder with *train_effnet.py*, or [download a pre-trained model](https://drive.google.com/open?id=1LFTlv0RFo2zXz2GKVEYZDBRL7wFIj5Cc) and put it as *~/data/finetuned_effnet.h5*
      ```
      python ./src/train_effnet.py --test_size 256 --batch_size 1024 --loop 10 --max_patience 1
      ```

* Align image - center and crop image with *align_images.py*
    ```
    mkdir images_aligned_test  #store aligned images

    python ./src/align_images.py images_raw_test/ images_aligned_test/
    ```



* Images encoder - output latent representations (18*512 matrix)  and reconstructed images using StyleGAN generator 
```
python ./src/encode_images.py --batch_size=2 --output_video=True --load_effnet=data/finetuned_effnet.h5 images_aligned_test/ images_reconstructed_test/ images_latent_test/
```

To be continued

