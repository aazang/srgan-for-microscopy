# Custom imports
from main import dataset_class, models

import time
import glob
import os
import git
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

random.seed(42)
import warnings
warnings.filterwarnings("ignore")


def main(n_epochs, dataset_path, batch_size, hr_height, hr_width):
    # Loading pretrained models
    load_pretrained_models = False
    # adam: learning rate
    lr = 0.00008
    # adam: decay of first order momentum of gradient
    b1 = 0.5
    # adam: decay of second order momentum of gradient
    b2 = 0.999
    # epoch from which to start lr decay
    decay_epoch = 100
    # number of cpu threads to use during batch generation
    n_cpu = 0
    # number of color channels
    channels = 3
    # Specific directory for patricular input parameter
    specific_dir_name = fr"/{os.path.basename(dataset_path)}_ep{n_epochs}_batchs{batch_size}_dim{hr_width}x{hr_height}"

    # os.makedirs("images", exist_ok=True)
    # os.makedirs("saved_models", exist_ok=True)

    cuda = torch.cuda.is_available()
    hr_shape = (hr_height, hr_width)

    # Normalization parameters for pre-trained PyTorch models
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    train_paths, test_paths = train_test_split(sorted(glob.glob(dataset_path + r"/*.*")), test_size=0.02, random_state=42)
    train_dataloader = DataLoader(dataset_class.ImageDataset(train_paths, hr_shape=hr_shape), batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    test_dataloader = DataLoader(dataset_class.ImageDataset(test_paths, hr_shape=hr_shape), batch_size=int(batch_size*0.75), shuffle=True, num_workers=n_cpu)
    # Initialize generator and discriminator
    generator = models.GeneratorResNet()
    discriminator = models.Discriminator(input_shape=(channels, *hr_shape))
    feature_extractor = models.FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Tensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    train_gen_losses, train_disc_losses, train_counter = [], [], []
    test_gen_losses, test_disc_losses = [], []
    test_counter = [idx*len(train_dataloader.dataset) for idx in range(1, n_epochs+1)]

    t_start_whole = time.time()
    
    print(specific_dir_name)

    for epoch in range(n_epochs):

    ### Training
        t_start_training = time.time()

        gen_loss, disc_loss = 0, 0
        tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))
        for batch_idx, imgs in enumerate(tqdm_bar):

            # t_start_training_one_it = time.time()

            generator.train(); discriminator.train()
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            # print(imgs_lr.dtype)
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            # print(imgs_hr.dtype)
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            ### Train Generator
            optimizer_G.zero_grad()
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN
            loss_G.backward()
            optimizer_G.step()

            ### Train Discriminator
            optimizer_D.zero_grad()
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            gen_loss += loss_G.item()
            train_gen_losses.append(loss_G.item())
            disc_loss += loss_D.item()
            train_disc_losses.append(loss_D.item())
            train_counter.append(batch_idx*batch_size + imgs_lr.size(0) + epoch*len(train_dataloader.dataset))
            tqdm_bar.set_postfix(gen_loss=gen_loss/(batch_idx+1), disc_loss=disc_loss/(batch_idx+1))

            # t_end_training_one_it = time.time()
        
        t_end_training = time.time()

        vram_a_training = torch.cuda.max_memory_allocated(0)
        torch.cuda.reset_peak_memory_stats()

        # Testing
        t_start_testing = time.time()

        gen_loss, disc_loss = 0, 0
        tqdm_bar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch} ', total=int(len(test_dataloader)))
        for batch_idx, imgs in enumerate(tqdm_bar):
            
            # t_start_testing_one_it = time.time()

            generator.eval(); discriminator.eval()
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            # print(imgs_lr.dtype)
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            # print(imgs_hr.dtype)
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            ### Eval Generator
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            ### Eval Discriminator
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake)

            gen_loss += loss_G.item()
            disc_loss += loss_D.item()
            tqdm_bar.set_postfix(gen_loss=gen_loss/(batch_idx+1), disc_loss=disc_loss/(batch_idx+1))

            # test_gen_losses.append(gen_loss/len(test_dataloader))
            # test_disc_losses.append(disc_loss/len(test_dataloader))
            test_gen_losses.append(loss_G.item())
            test_disc_losses.append(loss_D.item())

            # t_end_testing_one_it = time.time()

            # Save image grid with upsampled inputs and SRGAN outputs
            if random.uniform(0,1)<0.1:
                # images_dir = r"/home/mario/Github/srgan-for-microscopy/images"
                images_dir = r"./images"

                image_save_dir = images_dir + specific_dir_name
                os.makedirs(image_save_dir,  exist_ok=True)

                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
                gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
                save_image(img_grid, fr"images" + specific_dir_name + fr"/{batch_idx}.png", normalize=False)

        t_end_testing = time.time()

        vram_a_testing = torch.cuda.max_memory_allocated(0)

        saved_models_dir = r"./saved_models"
        # saved_models_dir = r"/home/mario/Github/srgan-for-microscopy/saved_models"
        parameter_save_dir = saved_models_dir + specific_dir_name
        os.makedirs(parameter_save_dir,  exist_ok=True)

        # Save model checkpoints
        if np.argmin(test_gen_losses) == len(test_gen_losses)-1:
            torch.save(generator.state_dict(), r"saved_models" + specific_dir_name + r"/generator.pth")
            torch.save(discriminator.state_dict(), r"saved_models" + specific_dir_name + r"/discriminator.pth")

    t_end_whole = time.time()


    # Dataframes to save the loss to
    dftrain0 = pd.DataFrame(data=train_counter, columns=["Batches used"])
    dftrain1 = pd.DataFrame(data=train_disc_losses, columns=["Discriminator loss"])
    dftrain2 = pd.DataFrame(data=train_gen_losses, columns=["Generator loss"])
    joined_train = (dftrain0.join(dftrain1)).join(dftrain2)

    dftest1 = pd.DataFrame(data=test_disc_losses, columns=["Discriminator loss"])
    dftest2 = pd.DataFrame(data=test_gen_losses, columns=["Generator loss"])
    joined_test = dftest1.join(dftest2)

    # timings
    t_whole = t_end_whole - t_start_whole
    t_training = t_end_training - t_start_training
    t_testing = t_end_testing - t_start_testing

    # iamges in dataset_path
    len_dataset_path = len([i for i in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, i))])

    # Creating Folder to save information to
    data_dir = r"./data"
    data_save_dir = data_dir + specific_dir_name
    os.makedirs(data_save_dir,  exist_ok=True)

    # Saving files to created folder
    joined_train.to_csv(data_save_dir + "/training_loss.csv")
    joined_test.to_csv(data_save_dir + "/test_loss.csv")
    with open(data_save_dir + "/info.txt", "w") as info_file:
        info_file.write("Time for entire training process: " + str(t_whole) + " seconds")
        info_file.write("\nImage count of dataset: " + str(len_dataset_path))
        info_file.write("\nMemory usage while training: " + str(vram_a_training/1000000) + " megabytes")
        info_file.write("\nMemory usage while testing: " + str(vram_a_testing/1000000) + " megabytes")

if __name__ == '__main__':
    repo = git.Repo('../srgan-for-microscopy')
    origin = repo.remote(name='origin')

    torch.cuda.empty_cache()

    origin.pull()
    trainingset = r"./input/PBC"
    for bs in range(2,11):
        main(2, trainingset, bs, 256, 256)
    repo.git.add('--all')
    repo.git.commit('-m', 'Automatic git add, commit and push', author='mario.baars@web.de')
    origin.push()

    origin.pull()
    trainingset = r"./input/PBCxDIV2K_randomcrops"
    main(2, trainingset, 8, 256, 256)
    repo.git.add('--all')
    repo.git.commit('-m', 'Automatic git add, commit and push', author='mario.baars@web.de')
    origin.push()

    origin.pull()
    trainingset = r"./input/PBC"
    for epochen in range(1,11):
        main(epochen, trainingset, 8, 256, 256)
    repo.git.add('--all')
    repo.git.commit('-m', 'Automatic git add, commit and push', author='mario.baars@web.de')
    origin.push()
    
