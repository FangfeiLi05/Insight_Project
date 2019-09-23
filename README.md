## Deep Learning Portrait Editing using StyleGAN and Latent Space Manipulation


StyleGAN Encoder is from @pbaylies

* Images aligment - crop and align images (just faces kept) with *images_align.py*
```
mkdir images_aligned_test

python ./src/images_align.py images_raw_test/ images_aligned_test/
```

* Images encoder trainning - here are two types of encoder
    * ResNet encoder - train with *train_resnet.py*, or [download pre-trained model (finetuned_resnet.h5)](https://drive.google.com/open?id=1tZLucJ1pZ8GA9JTRwF9d-Thr0zhR-i6l) and put it in *~/data/finetuned_resnet.h5*
    
    * EfficientNet encoder - train with *train_effnet.py*, or [download pre-trained model (finetuned_effnet.h5)](https://drive.google.com/open?id=1LFTlv0RFo2zXz2GKVEYZDBRL7wFIj5Cc) and put it in *~/data/finetuned_effnet.h5*
```
python ./src/train_resnet.py --test_size 256 --batch_size 1024 --loop 10 --max_patience 1

python ./src/train_effnet.py --test_size 256 --batch_size 1024 --loop 10 --max_patience 1
```

* Images encoder - output latent representations (18*512 matrix)  and reconstructed images using StyleGAN generator 
```
python ./src/images_encode.py --batch_size=2 --output_video=True --load_effnet=data/finetuned_effnet.h5 images_aligned_test/ images_reconstructed_test/ images_latent_test/
```

