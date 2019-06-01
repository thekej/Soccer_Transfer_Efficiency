"""Contains model that uses Stacked Attention for VQA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SoccerPredictionDeepModel(nn.Module):
    """CNN models using pytorch zoo.
    """
    def __init__(self, mode='Resnet', output_size=3):
        super(SoccerPredictionDeepModel, self).__init__()
        self.mode = mode
        if mode == 'resnet':
            model = models.resnet18()
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, output_size)
            self.model = model
        if mode == 'cnn':
            self.feats = nn.Sequential(
                    nn.Conv2d(3, 32, 6, 1, 1),
                    nn.MaxPool2d(2, 2),
                    nn.ReLU(True),
                    nn.BatchNorm2d(32),

                    nn.Conv2d(32, 64, 3,  1, 1),
                    nn.ReLU(True),
                    nn.BatchNorm2d(64),

                    nn.Conv2d(64, 64, 3,  1, 1),
                    nn.MaxPool2d(2, 2),
                    nn.ReLU(True),
                    nn.BatchNorm2d(64),

                    nn.Conv2d(64, 128, 3, 1, 1),
                    nn.ReLU(True),
                    nn.BatchNorm2d(128)
                    )

            self.classifier = nn.Conv2d(128, 3, 1)
            self.avgpool = nn.AvgPool2d(3, 6)
            self.dropout = nn.Dropout(0.5)
        if mode == 'mlp':
            self.model = nn.Sequential(
                            #nn.Dropout(0.2),
                            nn.Linear(814, 512),
                            nn.ReLU(True),
                            nn.BatchNorm1d(512),
                
                            nn.Dropout(0.5),
                            nn.Linear(512, 128),
                            nn.ReLU(True),
                            nn.BatchNorm1d(128),
                
                            nn.Dropout(0.5),
                            nn.Linear(128, 64),
                            nn.ReLU(True),
                            nn.BatchNorm1d(64),
                
                            nn.Dropout(0.2),
                            nn.Linear(64, 32),
                            nn.ReLU(True),
                            nn.BatchNorm1d(32),
                
                            nn.Dropout(0.2),
                            nn.Linear(32, 16),
                            nn.ReLU(True),
                            nn.BatchNorm1d(16),
                
                            nn.Dropout(0.2),
                            nn.Linear(16, output_size)
                            )

    def forward(self, game):
        """Passes the image and the question through a VQA model and generates answers.

        Args:
            game: Batch of game Variables

        Returns:
            Results probabilities
        """
        if self.mode == 'resnet':
            return self.model(game)
        if self.mode == 'cnn':
            out = self.feats(game)
            out = self.dropout(out)
            out = self.classifier(out)
            out = self.avgpool(out)
            out = out.view(-1, 3)
            return out
        if self.mode == 'mlp':
            game = game.view(game.size(0), -1)
            game = self.model(game)
            return game