import os
import sys
import bz2
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import config
import multiprocessing
from keras.utils import get_file
from keras.models import load_model
import dnnlib
import dnnlib.tflib as tflib
from encoder.ffhq_dataset.face_alignment import image_align
from encoder.ffhq_dataset.landmarks_detector import LandmarksDetector
from encoder.encoder.generator_model import Generator
from encoder.encoder.perceptual_model import PerceptualModel, load_images


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
        

def main():
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    parser = argparse.ArgumentParser(description='Align faces from input imagesm and find latent representation of aligned images using perceptual losses', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('raw_dir', help='Directory with raw images for face alignment')
    parser.add_argument('aligned_dir', help='Directory for storing aligned images')
    parser.add_argument('generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')

    parser.add_argument('--output_size', default=1024, help='The dimension of images for input to the model', type=int)
    parser.add_argument('--x_scale', default=1, help='Scaling factor for x dimension', type=float)
    parser.add_argument('--y_scale', default=1, help='Scaling factor for y dimension', type=float)
    parser.add_argument('--em_scale', default=0.1, help='Scaling factor for eye-mouth distance', type=float)
    parser.add_argument('--use_alpha', default=False, help='Add an alpha channel for masking', type=bool)
    parser.add_argument('--data_dir', default='data', help='Directory for storing optional models')
    parser.add_argument('--mask_dir', default='masks', help='Directory for storing optional masks')
    parser.add_argument('--load_last', default='', help='Start with embeddings from directory')
    parser.add_argument('--dlatent_avg', default='', help='Use dlatent from file specified here for truncation instead of dlatent_avg from Gs')
    parser.add_argument('--model_url', default='https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ', help='Fetch a StyleGAN model to train on from this URL') # karras2019stylegan-ffhq-1024x1024.pkl
    parser.add_argument('--model_res', default=1024, help='The dimension of images in the StyleGAN model', type=int)
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--resnet_image_size', default=256, help='Size of images for the Resnet model', type=int)
    parser.add_argument('--lr', default=0.02, help='Learning rate for perceptual model', type=float)
    parser.add_argument('--decay_rate', default=0.9, help='Decay rate for learning rate', type=float)
    parser.add_argument('--iterations', default=100, help='Number of optimization steps for each batch', type=int)
    parser.add_argument('--decay_steps', default=10, help='Decay steps for learning rate decay (as a percent of iterations)', type=float)
    parser.add_argument('--load_effnet', default='data/finetuned_effnet.h5', help='Model to load for EfficientNet approximation of dlatents')
    parser.add_argument('--load_resnet', default='data/finetuned_resnet.h5', help='Model to load for ResNet approximation of dlatents')

    # Loss function options
    parser.add_argument('--use_vgg_loss', default=0.4, help='Use VGG perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_vgg_layer', default=9, help='Pick which VGG layer to use.', type=int)
    parser.add_argument('--use_pixel_loss', default=1.5, help='Use logcosh image pixel loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_mssim_loss', default=100, help='Use MS-SIM perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_lpips_loss', default=100, help='Use LPIPS perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_l1_penalty', default=1, help='Use L1 penalty on latents; 0 to disable, > 0 to scale.', type=float)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    parser.add_argument('--tile_dlatents', default=False, help='Tile dlatents to use a single vector at each scale', type=bool)
    parser.add_argument('--clipping_threshold', default=2.0, help='Stochastic clipping of gradient values outside of this threshold', type=float)

    # Masking params
    parser.add_argument('--load_mask', default=False, help='Load segmentation masks', type=bool)
    parser.add_argument('--face_mask', default=False, help='Generate a mask for predicting only the face area', type=bool)
    parser.add_argument('--use_grabcut', default=True, help='Use grabcut algorithm on the face mask to better segment the foreground', type=bool)
    parser.add_argument('--scale_mask', default=1.5, help='Look over a wider section of foreground for grabcut', type=float)

    # Video params
    parser.add_argument('--video_dir', default='videos', help='Directory for storing training videos')
    parser.add_argument('--output_video', default=False, help='Generate videos of the optimization process', type=bool)
    parser.add_argument('--video_codec', default='MJPG', help='FOURCC-supported video codec name')
    parser.add_argument('--video_frame_rate', default=24, help='Video frames per second', type=int)
    parser.add_argument('--video_size', default=512, help='Video size in pixels', type=int)
    parser.add_argument('--video_skip', default=1, help='Only write every n frames (1 = write every frame)', type=int)
    
    args, other_args = parser.parse_known_args()
    
    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    RAW_IMAGES_DIR = args.raw_dir
    
    os.makedirs(args.aligned_dir, exist_ok=True)
    ALIGNED_IMAGES_DIR = args.aligned_dir

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in os.listdir(RAW_IMAGES_DIR):
        print('Aligning %s ...' % img_name)
        try:
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], 1)
            if os.path.isfile(fn):
                continue
            print('Getting landmarks...')
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                try:
                    print('Starting face alignment...')
                    face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=args.output_size, x_scale=args.x_scale, y_scale=args.y_scale, em_scale=args.em_scale, alpha=args.use_alpha)
                    print('Wrote result %s' % aligned_face_path)
                except:
                    print('Exception in face alignment!')
        except:
            print('Exception in landmark detection!')   
        
    args.decay_steps *= 0.01 * args.iterations # Calculate steps as a percent of total iterations

    if args.output_video:
        import cv2
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=args.batch_size)

    ref_images = [os.path.join(args.aligned_dir, x) for x in os.listdir(args.aligned_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.aligned_dir)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.generated_images_dir, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)
    
    if args.load_mask:
        os.makedirs(args.mask_dir, exist_ok=True)
    if args.output_video:
        os.makedirs(args.video_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url(args.model_url, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, args.batch_size, clipping_threshold=args.clipping_threshold, tiled_dlatent=args.tile_dlatents, model_res=args.model_res, randomize_noise=args.randomize_noise)
    if (args.dlatent_avg != ''):
        generator.set_dlatent_avg(np.load(args.dlatent_avg))

    perc_model = None
    if (args.use_vgg_loss > 0.00000001):
        with dnnlib.util.open_url('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2', cache_dir=config.cache_dir) as f:
            perc_model =  pickle.load(f)
            
    perceptual_model = PerceptualModel(args, perc_model=perc_model, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator)

    ff_model = None

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images)//args.batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
        if args.output_video:
            video_out = {}
            for name in names:
                video_out[name] = cv2.VideoWriter(os.path.join(args.video_dir, f'{name}.avi'),cv2.VideoWriter_fourcc(*args.video_codec), args.video_frame_rate, (args.video_size,args.video_size))

        perceptual_model.set_reference_images(images_batch)
        dlatents = None
        if (args.load_last != ''): # load previous dlatents for initialization
            for name in names:
                dl = np.expand_dims(np.load(os.path.join(args.load_last, f'{name}.npy')),axis=0)
                if (dlatents is None):
                    dlatents = dl
                else:
                    dlatents = np.vstack((dlatents,dl))
        else:
            if (ff_model is None):
                if os.path.exists(args.load_resnet):
                    print('Loading ResNet Model:')
                    ff_model = load_model(args.load_resnet)
                    from keras.applications.resnet50 import preprocess_input
            if (ff_model is None):
                if os.path.exists(args.load_effnet):
                    import efficientnet
                    print('Loading EfficientNet Model:')
                    ff_model = load_model(args.load_effnet)
                    from efficientnet import preprocess_input
            if (ff_model is not None): # predict initial dlatents with ResNet model
                dlatents = ff_model.predict(preprocess_input(load_images(images_batch,image_size=args.resnet_image_size)))
        if dlatents is not None:
            generator.set_dlatents(dlatents)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations)
        pbar = tqdm(op, leave=False, total=args.iterations)
        vid_count = 0
        best_loss = None
        best_dlatent = None
        for loss_dict in pbar:
            pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v)
                    for k, v in loss_dict.items()]))
            if best_loss is None or loss_dict['loss'] < best_loss:
                best_loss = loss_dict['loss']
                best_dlatent = generator.get_dlatents()
            if args.output_video and (vid_count % args.video_skip == 0):
                batch_frames = generator.generate_images()
                for i, name in enumerate(names):
                    video_frame = PIL.Image.fromarray(batch_frames[i],
                                                      'RGB').resize((args.video_size,args.video_size),PIL.Image.LANCZOS)
                    video_out[name].write(cv2.cvtColor(np.array(video_frame).astype('uint8'), 
                                                       cv2.COLOR_RGB2BGR))
            generator.stochastic_clip_dlatents()
        print(" ".join(names), " Loss {:.4f}".format(best_loss))

        if args.output_video:
            for name in names:
                video_out[name].release()

        # Generate images from found dlatents and save them
        generator.set_dlatents(best_dlatent)
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()


if __name__ == "__main__":
    main()
    
