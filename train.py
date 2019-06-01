"""Training script to train a language autoencoder model.
"""

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import h5py
import json
import logging
import random
import time
import torch
import os

from model import SoccerPredictionDeepModel
from model import get_game_loader


def evaluate(model, data_loader, criterion, epoch, args):
    """Calculates vqg average loss on data_loader.

    Args:
        model: model.
        data_loader: Iterator for the data.
        criterion: The criterion function used to evaluate the loss.
        args: ArgumentParser object.

    Returns:
        A float value of average loss.
    """
    gts, preds = [], []
    model.eval()
    total_loss = 0.0
    iterations = 0
    acc = 0.0
    count = 0
    total_steps = len(data_loader)

    start_time = time.time()
    for i, (game, result, _, _) in enumerate(data_loader):
        # Set mini-batch dataset.
        if torch.cuda.is_available():
            game = game.cuda() 
            result = result.cuda()
        
        # Forward
        out = model(game)
        
        # Calculate the loss.
        loss = criterion(out, result)

        # Backprop and optimize.
        total_loss += loss.item()
        iterations += 1
        _, preds = torch.max(out.data, 1)
        acc += (preds == result).sum().item()
        count += game.shape[0]

    return total_loss / iterations, acc/float(count)

def run_eval(model, data_loader, criterion, args, epoch, scheduler):
    start_time = time.time()
    val_loss, val_acc = evaluate(model, data_loader, criterion, epoch, args)
    delta_time = time.time() - start_time
    scheduler.step(val_loss)
    logging.info('Time: %.4f, Epoch [%d/%d], Val loss: %.4f, Val accuracy: %.4f' % (
        delta_time, epoch, args.num_epochs, val_loss, val_acc))
    logging.info('=' * 80)
    scheduler.step(val_loss)
    
def get_dataset_size(dataset):
    annos = h5py.File(dataset, 'r')
    size = annos['games'].shape[0]
    annos.close()
    return size

def main(args):
    # Setting up seeds.
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create model directory.
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Config logging.
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(args.model_dir, 'train.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Save the arguments.
    with open(os.path.join(args.model_dir, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # Build data loader.
    logging.info("Building data loader...")
    dataset_size = get_dataset_size(args.dataset)
    train_set = random.sample(range(dataset_size), int(dataset_size * 0.9))
    data_loader = get_game_loader(args.dataset, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, 
                                  indices=train_set)
    val_data_loader = get_game_loader(args.dataset, args.batch_size, shuffle=False,
                                      num_workers=args.num_workers,
                                      indices=list(set(range(dataset_size)) - set(train_set)))
    logging.info("Done")

    # Build the models
    logging.info("Building Game Predictor...")
    model = SoccerPredictionDeepModel(mode=args.mode, output_size=args.output_size)
    pre = 0
    if args.pretrained:
        model.load_state_dict(torch.load(args.model_path))
        print torch.load(args.model_path)
        exit()
        pre = int(args.model_path.split('-')[1].split('.')[0])
    logging.info("Done")

    if torch.cuda.is_available():
        model.cuda()

    # Loss and Optimizer.
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion.cuda()        

    # Parameters to train.
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.99, weight_decay = 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                  factor=0.1, patience=args.patience,
                                  verbose=True, min_lr=1e-6)

    # Train the Models.
    total_steps = len(data_loader) * args.num_epochs
    start_time = time.time()
    
    n_steps = 0
    for epoch in range(args.num_epochs):
        l = 0.0
        a = 0.0
        for i, (games, results, _, _) in enumerate(data_loader):
            n_steps += 1

            # Set mini-batch dataset.
            if torch.cuda.is_available():
                games = games.cuda()
                results = results.cuda()            
            # Forward.
            model.train()
            model.zero_grad()
            outputs = model(games)

            # Calculate the loss.
            loss = criterion(outputs, results)

            # Backprop and optimize.
            loss.backward()
            optimizer.step()

            # Eval now.
            #if (n_steps % args.eval_every_n_steps == 0):
            #    run_eval(model, val_data_loader, criterion, 
            #             args, epoch, scheduler)

            _, preds = torch.max(outputs.data, 1)
            a += (preds == results).sum().item()
            l += loss.item()

            # Save the models.
            #if (i+1) % args.save_step == 0:
            #    torch.save(model.state_dict(),
            #               os.path.join(args.model_dir,
            #                            'model-%d-%d.pkl' %(epoch+1, i+1)))

        torch.save(model.state_dict(), os.path.join(args.model_dir,
                   'model-%d.pkl' % (epoch+1+pre)))

        # Evaluation and learning rate updates.
        logging.info('Epoch [%d/%d], Train loss: %.4f, Train accuracy: %.4f' % (
                    epoch, args.num_epochs, l / len(data_loader), a / int(dataset_size * 0.9)))
        run_eval(model, val_data_loader, criterion, args, epoch, scheduler)

    # Save the final model.
    torch.save(model.state_dict(),os.path.join(args.model_dir,'model.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Location parameters.
    parser.add_argument('--model-dir', type=str,
                        default='weights/mlp/',
                        help='Path for saving trained models.')
    parser.add_argument('--model-path', type=str,
                        default='weights/resnet/model.pkl',
                        help='Path for saving trained models.')
    parser.add_argument('--dataset', type=str,
                        default='data/dataset.hdf5',
                        help='Match lineups dataset file.')

    # Session parameters.
    parser.add_argument('--log-step', type=int , default=10,
                        help='Step size for printing log info.')
    parser.add_argument('--save-step', type=int , default=1000,
                        help='Step size for saving trained models.')
    parser.add_argument('--eval-steps', type=int, default=None,
                        help='Number of eval steps to run.')
    parser.add_argument('--eval-every-n-steps', type=int, default=1000,
                        help='Run eval after every N steps.')
    parser.add_argument('--eval-all', action='store_true',
                        help='Run eval after each epoch.')
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--data-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1)

    # Model parameters.
    parser.add_argument('--mode', type=str, default='resnet',
                        help='Type of cnn model.')
    parser.add_argument('--output-size', type=int , default=3,
                        help='Number of classes.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Run eval after each epoch.')
    args = parser.parse_args()

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    main(args)