# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import os
import wget

class CorruptMnist(Dataset):
    def __init__(self, input_filepath, output_filepath, train):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.download_data(train)
        if train:
            content = [ ]
            for i in range(5):
                content.append(np.load(os.path.join(input_filepath,f"train_{i}.npz"), allow_pickle=True))
            data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
        else:
            content = np.load(os.path.join(input_filepath,"test.npz"), allow_pickle=True)
            data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['labels'])
            
        self.data = data
        self.targets = targets
    
    def download_data(self, train):
        files = os.listdir(self.input_filepath)
        if train:
            for file_idx in range(5):
                if os.path.join(self.input_filepath,f'train_{file_idx}.npy') not in files:
                    wget.download(f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/train_{file_idx}.npz",
                    out = self.input_filepath)
        else:
            if os.path.join(self.input_filepath,"test.npy") not in files:    
                wget.download("https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/test.npz",
                out = self.input_filepath)

    def process_data(self):
        # Define a transform to normalize the data
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        

    
    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    dataset_train = CorruptMnist(input_filepath, output_filepath, train=True)
    dataset_test = CorruptMnist(input_filepath, output_filepath, train=False)


    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

    # /Users/invisible_man/Documents/DTU/Courses/MLOps/mlops_handson/mnist_classifier/src/data/make_dataset.py /Users/invisible_man/Documents/DTU/Courses/MLOps/mlops_handson/mnist_classifier/data/raw /Users/invisible_man/Documents/DTU/Courses/MLOps/mlops_handson/mnist_classifier/data/processed
