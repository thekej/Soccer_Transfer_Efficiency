"""Loads the game line up data with game result.
"""

import h5py
import numpy as np
import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, indices=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            dataset: annotation hdf5 location.
        """
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        """Returns one data pair and length (question, answer, length).
        """
        if not hasattr(self, 'questions'):
            annos = h5py.File(self.dataset, 'r')
            self.games = annos['games']
            self.home = annos['home']
            self.away = annos['away']
            self.results = annos['results']
        
        if self.indices is not None:
            index = self.indices[index]
        
        game = self.games[index]
        result = self.results[index]
        home = self.home[index]
        away = self.away[index]
        game = torch.from_numpy(np.array(game))
        result = torch.from_numpy(np.array(result))
        return game, result, home, away

    def __len__(self):
        if self.indices is not None:
            return len(self.indices) 
        annos = h5py.File(self.dataset, 'r')
        return annos['games'].shape[0]

def collate_fn(data):
    # Sort a data list by caption length (descending order).
    game, result, home, away = zip(*data)
    game = torch.stack(game, 0).float()
    result = torch.stack(result, 0).long()
    return game, result, list(home), list(away)

def get_game_loader(dataset, batch_size, shuffle, num_workers, indices=None):
    """Returns torch.utils.data.DataLoader for the dataset.
    """
    lm = Dataset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(dataset=lm,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
