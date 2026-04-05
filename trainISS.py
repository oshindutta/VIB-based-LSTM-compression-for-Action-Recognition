import torch
import torch.nn as nn
import sys
import os
import numpy as np
import itertools
from models.modelcomp import *
from datasets.datasetucf11 import *
#from prune_model import*
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
import datetime

class AverageMeter(object):
    # Computes running average accuracy
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="ucf11/data/UCF11_updated_mpg-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--split_path", type=str, default="ucf11/data/ucfTrainTestlist", help="Path to train/test split")
    parser.add_argument("--split_number", type=int, default=1, help="train/test split number. One of {1, 2, 3}")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=40, help="Number of frames in each sequence")
    parser.add_argument("--img_dim", type=int, default=224, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=52, help="Dimensionality of the latent representation")
    parser.add_argument("--hidden_dim", type=int, default=9, help="Dimensionality of the LSTM hidden representation")
    parser.add_argument("--reg_coef", type=float, default=0.01, help="group lasso regularization parameter")
    parser.add_argument("--lasso_threshold", type=float, default=0.01, help="group lasso lasso threshold")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for all the parameters")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Regularization coeff for L2 regularizer")
    parser.add_argument("--no_mask_upto", type=int, default=0, help="epoch upto which no masking")
    parser.add_argument("--no_mask_after", type=int, default=100, help="epoch after which no more masking")
    parser.add_argument("--lasso_decay", type=float, default=1, help="mutilplier for lasso lasso_threshold for its decay ")

    parser.add_argument("--checkpoint_model", type=str, default="../ConvLSTMcomp_97.tar", help="Optional path to checkpoint model")
    parser.add_argument("--mask_path", type=str, default="../mask.pt", help="Path to stored ib mask")
    parser.add_argument("--checkpoint_name", type=str, default="", help="Name of checkpoint model")
    parser.add_argument(
        "--checkpoint_interval", type=int, default=1, help="Interval between saving model checkpoints"
    )
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("----- GPU check -------", torch.cuda.is_available())

    image_shape = (opt.channels, opt.img_dim, opt.img_dim)

    # Define training set
    train_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    # Define test set
    test_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # Classification criterion
    cls_criterion = nn.CrossEntropyLoss().to(device)

    ibmask = torch.load(opt.mask_path)

    # Define network
    model = ConvLSTMcomp(
        num_classes=train_dataset.num_classes,
        latent_dim=52,
        lstm_layers=1,
        hidden_dim=9,
        bidirectional=False,
        mhid = 128,
        ib4m = ibmask
    )
    model = model.to(device)

    lasso_threshold = opt.lasso_threshold
   
    # Add weights from VIB trained checkpoint model if specified
    if opt.checkpoint_model:
        checkpoint_data = torch.load(opt.checkpoint_model)
        model.load_state_dict(checkpoint_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay= opt.weight_decay)

    Best_acc = 0
    Best_epoch = 0
    compression_ratio = 1
    last_compression_ratio = 1

    def test_model(epoch, mask, Best_acc, Best_epoch, compression_ratio):
        """ Evaluate the model on the test set """
        print("")
        model.eval()
        test_metrics = {"loss": [], "acc": []}
        top1_test = AverageMeter()
        for batch_i, (X, y) in enumerate(test_dataloader):
            image_sequences = Variable(X.to(device), requires_grad=False)
            labels = Variable(y, requires_grad=False).to(device)
            with torch.no_grad():
                # Reset LSTM hidden state
                model.lstm.reset_hidden_state()
                if(epoch > opt.no_mask_upto):
                    # Masking the small weight groups 
                    mask = mask.reshape(opt.hidden_dim,1)
                    masks = torch.cat((mask, mask, mask, mask),0)
                    masks = masks.reshape(4*opt.hidden_dim,1)
                    state_dict = model.state_dict()
                    state_dict["lstm.lstm.weight_ih_l0"] = masks*state_dict["lstm.lstm.weight_ih_l0"]
                    state_dict["lstm.lstm.weight_hh_l0"] = masks*state_dict["lstm.lstm.weight_hh_l0"]
                    mask = mask.reshape(1,opt.hidden_dim)
                    state_dict["lstm.lstm.weight_hh_l0"] = mask*state_dict["lstm.lstm.weight_hh_l0"]
                    masks = masks.reshape(4*opt.hidden_dim)
                    state_dict["lstm.lstm.bias_ih_l0"] = masks*state_dict["lstm.lstm.bias_ih_l0"]
                    state_dict["lstm.lstm.bias_hh_l0"] = masks*state_dict["lstm.lstm.bias_hh_l0"]
                    model.load_state_dict(state_dict)
                # Get sequence predictions
                predictions = model(image_sequences)
            # Compute metrics
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            loss = cls_criterion(predictions, labels).item()
            top1_test.update(acc, X.size(0))
            # Keep track of loss and accuracy
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)

            # Log test performance
            if(batch_i%20 == 0):
                sys.stdout.write(
                    "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                    % (
                        batch_i,
                        len(test_dataloader),
                        loss,
                        np.mean(test_metrics["loss"]),
                        acc,
                        top1_test.avg,
                    )
                )
        Best_epoch = epoch if (np.mean(test_metrics["acc"]) > Best_acc) else Best_epoch
        Best_acc = top1_test.avg if (top1_test.avg > Best_acc) else Best_acc
        print(f"At compression ratio: {compression_ratio}, Best_test_acc: {Best_acc} , at epoch: {Best_epoch}")
        model.train()
        print("")
        return Best_acc, Best_epoch

    for epoch in range(opt.num_epochs):
        epoch_metrics = {"loss": [], "reg_loss":[], "acc": []}
        prev_time = time.time()
        print(f"--- Epoch {epoch} ---")
        top1_train = AverageMeter()
        for batch_i, (X, y) in enumerate(train_dataloader):

            if X.size(0) == 1:
                continue

            image_sequences = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)

            optimizer.zero_grad()

            # Reset LSTM hidden state
            model.lstm.reset_hidden_state()

            # Get sequence predictions
            predictions = model(image_sequences)

            # Compute metrics with Regularization
            W = torch.cat((model.lstm.lstm.weight_ih_l0, model.lstm.lstm.weight_hh_l0),1)      
            W_sqr = W*W
            W1 = torch.sum(W_sqr, dim=1)          
            W11, W12, W13, W14 = torch.chunk(W1, 4, dim=0)                     # opt.hidden_dimx1     
            W2 = torch.sum(model.lstm.lstm.weight_hh_l0*model.lstm.lstm.weight_hh_l0, dim=0)       # opt.hidden_dimx1      
            W_reg = (W11 + W12 + W13 + W14 + W2)                  # opt.hidden_dimx1
            reg_coef = opt.reg_coef
            Reg = W_reg.add(1e-8).pow(1/2.).sum()
            loss = cls_criterion(predictions, labels)
            reg_loss = (loss + reg_coef*Reg) if epoch < opt.no_mask_after else loss
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            top1_train.update(acc, X.size(0))

            reg_loss.backward()
            optimizer.step()

            # Keep track of epoch metrics
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["reg_loss"].append(reg_loss.item())
            epoch_metrics["acc"].append(acc)

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = opt.num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if(batch_i%29 == 0):
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                    % (
                        epoch,
                        opt.num_epochs,
                        batch_i,
                        len(train_dataloader),
                        reg_loss.item(),
                        np.mean(epoch_metrics["reg_loss"]),
                        acc,
                        top1_train.avg,
                        time_left,
                    )
                )

            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mask = (W_reg > lasso_threshold).float()
        if epoch == opt.no_mask_after :
            mask1 = mask
        if(epoch > opt.no_mask_after):
            lasso_threshold = opt.lasso_decay*lasso_threshold
            mask = mask1
        pruned = np.sum(mask.cpu().numpy()==0)
        print("\n-------------number of removed weight groups ------------", pruned)
        hidden_dim = opt.hidden_dim
        latent_dim = opt.latent_dim
        total_params = 4*hidden_dim*(hidden_dim + latent_dim + 1)
        pruned_params =  4*pruned*(2*hidden_dim + latent_dim - pruned)
        remain_params = total_params - pruned_params
        flops = 0
        compression_ratio = total_params/remain_params

        print('total parameters: {}, pruned parameters: {}, remaining params:{}, remain/total params:{}, remaining flops: {}, '
              .format(total_params, pruned_params, remain_params, 
                    float(total_params-pruned_params)/total_params, flops))

        if (compression_ratio != last_compression_ratio):
            Best_acc = 0
            Best_epoch = 0
        last_compression_ratio = compression_ratio

        # Evaluate the model on the test set
        Best_acc, Best_epoch = test_model(epoch, mask, Best_acc, Best_epoch, compression_ratio)

        # get the pruned model dict
        no_mask_indices = [i for i,j in enumerate(mask) if j==1]
        remain = len(no_mask_indices)
        mhid = 512
        
        # Save model checkpoint
        if (epoch % opt.checkpoint_interval == 0) and (remain != 0) :
            os.makedirs("model_checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"model_checkpoints/{opt.checkpoint_name}_{epoch}_{remain}.pth")
