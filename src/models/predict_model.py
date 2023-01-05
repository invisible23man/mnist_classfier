import argparse
import sys
import os

import torch

from src.models.model import MyAwesomeModel

import matplotlib.pyplot as plt
import numpy as np

class Predict(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for prediction",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def predict(self):
        print("Running Predictions")
        parser = argparse.ArgumentParser(description='Prediction arguments')
        parser.add_argument('--load_model_from', 
            default = "/Users/invisible_man/Documents/DTU/Courses/MLOps/mlops_handson/mnist_classifier/models/trained_model.pt")
        parser.add_argument('--load_data_from', 
            default= "/Users/invisible_man/Documents/DTU/Courses/MLOps/mlops_handson/mnist_classifier/data")
           
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        model = model.to(self.device)

        data_to_predict = torch.from_numpy(np.load(os.path.join(args.load_data_from, "example_images.npy"))).split(10) # tuple, 1x28x28
        data_to_predict = torch.stack(data_to_predict, 0) # 10 x 1 x 28 x 28
       
        for batch_id, batch in enumerate(data_to_predict):
            x = batch  
            preds = model(x.to(dtype=torch.float32, device=self.device))
            preds = preds.argmax(dim=1)
                        
            print(f"Prediction for the batch {batch_id+1} is {preds.tolist()}")


if __name__ == '__main__':
    Predict()