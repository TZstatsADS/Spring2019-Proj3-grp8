from Network_gan import Generator, Discriminator

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
import keras
import keras.backend as K
from keras.layers import Lambda, Input
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
from skimage.transform import rescale, resize
from scipy.misc import imresize
import os
import glob 
import cv2
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

np.random.seed(10)
image_shape = (384,384,3)

def vgg_loss(y_true, y_pred):
    
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


def get_paths(folder_name_string):
    im_dir = os.path.join(os.getcwd(), folder_name_string)
    paths = glob.glob(os.path.join(im_dir, "*.jpg"))
    paths = list(paths)
    return paths


files = get_paths('/Users/matthewvitha/Downloads/train_set/HR')

def pad_edit_imgs(hr_paths):
    
    image_size = 384
    color_dim = 3
    step = 21
    padding = 6

    HR_sequence = []
    
    for i in range(len(hr_paths)):

            HR=cv2.imread(hr_paths[i])
            
            if len(HR.shape) == 3: 
                h, w, c = HR.shape
            else:
                h, w = HR.shape 
     
            nx, ny = 0, 0
            for x in range(0, h - image_size + 1, step):
                nx += 1; ny = 0
                for y in range(0, w - image_size + 1, step):
                    ny += 1
    
                    HR_cropped = HR[x + padding: x + padding + image_size, y + padding: y + padding + image_size] 
                    HR_cropped =  HR_cropped / 255.0
                    HR_sequence.append(HR_cropped)

    return HR_sequence
files1  = files[0:100]
hr_img = pad_edit_imgs(files1)

x_train = hr_img[:500]
x_test = hr_img[600:900]


def hr_images(images):
    images_hr = array(images)
    return images_hr

def lr_images(images_real , downscale):
    
    images = []
    for img in  range(len(images_real)):
        images.append(imresize(images_real[img], [images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale], interp='bicubic', mode=None))
    images_lr = array(images)
    return images_lr

def preprocess_HR(x):
    return np.divide(x.astype(np.float32), 127.5) - np.ones_like(x,dtype=np.float32)


def deprocess_HR(x):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8) 


def preprocess_LR(x):
    return np.divide(x.astype(np.float32), 255.)


def deprocess_LR(x):
    x = np.clip(x*255, 0, 255)
    return x

def normalize(input_data):

    return (input_data - 127.5)/127.5 
    #.astype(np.float32) was removed
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8) 

def deprocess_LRS(x):
    x = np.clip(x*255, 0, 255)
    return x.astype(np.uint8)



def 4dimarray(ndarray, image_size = 32):
    four_dim = np.empty([len(ndarray), image_size, image_size, 3])
    for i in range(len(ndarray)):
        if ndarray[i].shape == (image_size,image_size,3):
            four_dim[i] = ndarray[i]
        
    return four_dim  


x_train_hr = hr_images(x_train)
x_train_hr = 4dimarray(x_train_hr,384)
x_train_hr = normalize(x_train_hr)

x_train_lr = lr_images(x_train, 4)
x_train_lr = 4dimarray(x_train_lr,96)
x_train_lr = normalize(x_train_lr)


x_test_hr = hr_images(x_test)
x_test_hr = 4dimarray(x_test_hr,384)
x_test_hr = normalize(x_test_hr)

x_test_lr = lr_images(x_test, 4)
x_test_lr = 4dimarray(x_test_lr,96)
x_test_lr = normalize(x_test_lr)

print("data processed")


def ndarray_to_4dim(ndarray, image_size = 32):
    four_dim = np.empty([len(ndarray), image_size, image_size, 3])
    for i in range(len(ndarray)):
        if ndarray[i].shape == (image_size,image_size,3):
            four_dim[i] = ndarray[i]
        
    return four_dim  

#x_train_hr = ndarray_to_4dim(x_train_hr,384)
#x_train_lr = ndarray_to_4dim(x_train_lr,384)
#x_test_hr = ndarray_to_4dim(x_test_hr,384)
#x_test_lr = ndarray_to_4dim(x_test_lr,384)




def plot_generated_images(epoch,generator, examples=3 , dim=(1, 3), figsize=(15, 5)):
    
    rand_nums = np.random.randint(0, x_test_hr.shape[0], size=examples)
    image_batch_hr = denormalize(x_test_hr[rand_nums])
    image_batch_lr = x_test_lr[rand_nums]
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    
    #generated_image = deprocess_HR(generator.predict(image_batch_lr))
    
    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[1], interpolation='nearest')
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[1], interpolation='nearest')
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[1], interpolation='nearest')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/matthewvitha/Downloads/V7/gan_generated_image_epoch_%d.png' % epoch)


  

def train(epochs=1, batch_size=128):

    downscale_factor = 4
    
    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    
    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=vgg_loss, optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)
    
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, 3)
    gan = get_gan_network(discriminator, shape, generator, adam)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in range(batch_count):
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2
            
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            #d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
            
        print("Loss HR , Loss LR, Loss GAN")
        print(d_loss_real, d_loss_fake, loss_gan)

        if e == 1 or e % 5 == 0:
            plot_generated_images(e, generator)
        if e % 300 == 0:
            generator.save('/Users/matthewvitha/Downloads/V7/gen_model%d.h5' % e)
            discriminator.save('/Users/matthewvitha/Downloads/V7/dis_model%d.h5' % e)
            gan.save('/Users/matthewvitha/Downloads/V7/gan_model%d.h5' % e)

train(1,100)


