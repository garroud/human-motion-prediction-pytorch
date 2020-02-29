import sys
sys.path.append('src/')

from datasets.dataset import Dataset
from nri import utils
import torch

class NRIDataset(Dataset):
    def __init__(self, data_dir, suffix, batch_size):
        self.data_dir = data_dir
        self.suffix = suffix
        self.batch_size = batch_size
        self.train_loader, self.validation_loader, self.test_loader, _, _, _, _ \
         = utils.load_data(self.batch_size, self.data_dir, self.suffix)

    def get_batch_train(self):
        return 
