# Copyright 2014-2017 Bert Carremans
# Author: Bert Carremans <bertcarremans.be>
#
# License: BSD 3 clause


import os
import random
from shutil import copyfile


def img_train_test_split(img_source_dir, train_size ):
    """
    Randomly splits images over a train and validation folder, while preserving the folder structure

    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path   

    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')

    if not (isinstance(train_size, float)):
        raise AttributeError('train_size must be a float')

    # Set up empty folder structure if not exists
    if not os.path.exists('../dataset'):
        os.makedirs('../dataset')
    else:
        if not os.path.exists('../dataset/train'):
            os.makedirs('../dataset/train')
        if not os.path.exists('../dataset/val'):
            os.makedirs('../dataset/val')

    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]

    for subdir in subdirs:
        subdir_fullpath = os.path.join(img_source_dir, subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            break

        train_subdir = os.path.join('../dataset/train', subdir)
        validation_subdir = os.path.join('../dataset/val', subdir)

        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)

        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)

        train_counter = 0
        validation_counter = 0
        random.seed(40)
        # Randomly assign an image to train or validation folder
        for filename in os.listdir(subdir_fullpath):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                fileparts = filename.split('.')

                if random.uniform(0, 1) <= train_size:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(train_subdir, filename))#str(train_counter)
                    train_counter += 1
                else:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(validation_subdir, filename)) #str(validation_counter)
                    validation_counter += 1

        print('Copied ' + str(train_counter) + ' images to dataset/train/' + subdir)
        print('Copied ' + str(validation_counter) + ' images to dataset/val/' + subdir)

if __name__ == "__main__":
    img_source_dir = 'D:/softwareDate/project/dataset2/tar_data/head'
    img_train_test_split(img_source_dir, 0.7)