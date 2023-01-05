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
        self.logger = logging.getLogger(__name__)
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

        if not self.load_data(self.output_filepath): 
            # Download and/or Process Raw Data            
            self.download_data(train)
            files = [f for f in os.listdir(self.input_filepath) if f.endswith('.npz') and f.startswith('train')]
            if train:
                content = [ ]
                for i in range(len(files)):
                    content.append(np.load(os.path.join(input_filepath,f"train_{i}.npz"), allow_pickle=True))
                data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
                targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
            else:
                content = np.load(os.path.join(input_filepath,"test.npz"), allow_pickle=True)
                data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
                targets = torch.tensor(content['labels'])
            
            self.data = data
            self.targets = targets
            
            self.process_data(train)

    def download_data(self, train):
        files = os.listdir(self.input_filepath)
        if train:
            for file_idx in range(5):
                if f'train_{file_idx}.npz' not in files:
                    self.logger.info('raw train data not found in directory...downloading')
                    wget.download(f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/train_{file_idx}.npz",
                    out = self.input_filepath)
        else:
            if "test.npz" not in files:    
                self.logger.info('raw test data not found in directory...downloading')
                wget.download("https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/test.npz",
                out = self.input_filepath)

    def process_data(self, train):
        self.logger.info('processing the data')
        # Define a transform to normalize the data
        self.transform = transforms.Compose(
            [transforms.Normalize((0,), (1,))]
        )

        # self.transformed_data = self.transform.apply_transform(self.data)
        self.transformed_data = torch.stack([self.transform(d) for d in self.data])

        if train:
            np.save(os.path.join(self.output_filepath, "train_x.npy"), self.transformed_data)
            np.save(os.path.join(self.output_filepath, "train_y.npy"), self.targets)
        else:
            np.save(os.path.join(self.output_filepath, "test_x.npy"), self.transformed_data)
            np.save(os.path.join(self.output_filepath, "test_y.npy"), self.targets)

    def load_data(self, train):
        files = os.listdir(self.output_filepath)
        if train and all(f in files for f in ["train_x.npy","train_y.npy"]):
            self.data = torch.from_numpy(np.load(os.path.join(self.output_filepath, "train_x.npy")))
            self.targets = torch.from_numpy(np.load(os.path.join(self.output_filepath, "train_y.npy")))
            return 1
        elif not train and all(f in files for f in ["test_x.npy","test_y.npy"]):
            self.data = torch.from_numpy(np.load(os.path.join(self.output_filepath, "test_x.npy")))
            self.targets = torch.from_numpy(np.load(os.path.join(self.output_filepath, "test_y.npy")))
            return 1
        else:
            self.logger.info("processed files not found")
            return 0

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