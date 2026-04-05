import torch
import sys
import os
import numpy as np
#import itertools
#import shutil
from models.model_VIBLSTM import *
#from modele2eVIBLSTM import *
from datasets.datasetucf11 import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
import datetime

class AverageMeter(object):
    """Calculates and updates the average and current values"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="UCF11_updated_mpg-frames", help="Path to training dataset")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=32, help="Number of frames in each training sequence")
    parser.add_argument("--img_dim1", type=int, default=299, help="Height- dimension, example- for end-to-end LSTM- 160, for inceptionnet-v3- 299")
    parser.add_argument("--img_dim2", type=int, default=299, help="Width- dimension example- for end-to-end LSTM- 120, for inceptionnet-v3-299")
    parser.add_argument("--inchannels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--latent_dim", type=int, default=2048, help="Dimensionality of the feature representation input to LSTM")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Dimensionality of the hidden state representation")

    parser.add_argument("--checkpoint_model", type=str, default="", help="optional path to naive-LSTM trained model")
    parser.add_argument("--finetune_model", type=str, default="", help="Optional path to VIB-LSTM model to fine tune by masking VIB parameters and finetuning only LSTM weights")
    parser.add_argument("--resume", type=str, default="", help="Optional path to resuming VIB-LSTM model training from any saved checkpoint")
    parser.add_argument('--e2e',type=bool, default=False, help='Whether end to end LSTM pruning is being performed.')
    
    parser.add_argument('--kl_fac', type=float, default=1e-6,help='Factor for the KL term.')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate.')
    parser.add_argument('--weight_decay', '-wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--ib_lr', type=float, default=1e-3, help='Separate learning rate for information bottleneck params. Set to -1 to follow opt.lr.')
    parser.add_argument('--ib_wd', type=float, default=-1, help='Separate weight decay for information bottleneck params. Set to -1 to follow opt.weight_decay')
    parser.add_argument('--lr_epoch', type=int, default=30, help='Decrease learning rate every x epochs.')
     
    parser.add_argument('--eval', type=bool, default=False, help='Whether to only evaluate model.')
    parser.add_argument('--save_dir', type=str, default='VIB_checkpoint', 
        help='To save end-to-end LSTM trained model checkpoints: E2eIBucf11_checkpoint or to save Convolutional-LSTM model checkpoints: VIB_checkpoint')
    parser.add_argument('--featureExtractor', type=str, default='Inc',help='Select feature extractor- Res for resnet152, Inc for Inceptionnet-v3, Eff for Efficientnet-b1')
    parser.add_argument('--kml', type=float, default=2, help='Compression factor multiplier')
    
    opt = parser.parse_args()
    print(opt)

    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    """"manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = {k: v for k, v in opt._get_kwargs()}
    image_shape = (opt.inchannels, opt.img_dim1, opt.img_dim2)
    
    # Define training set
    train_dataset = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    # Define test set
    test_dataset = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # Classification criterion
    cls_criterion = nn.CrossEntropyLoss().to(device)

    #Select feature extractor- resnet152 or Inceptionnet-v3 or Efficientnet b1
    if (opt.featureExtractor == 'Res' or 'Inc' or 'Eff') and opt.e2e==False:
        print("\n Assigned feature extractor ")
    elif opt.e2e==False:
        print("\n Wrong feature extractor input")
    
    #if fine-tuneing certain VIB_LSTM model, then masking=True
    if opt.finetune_model !="":
        masking= True
    else:
        masking=False

    #Initilizing model for either end-to-end VIB-LSTM or convolutional-VIB-LSTM
    if opt.e2e == True:
        latent_dim = opt.inchannels*opt.img_dim1*opt.img_dim2
        opt.latent_dim=latent_dim
        print("\n Assigned e2e LSTM model")
        model=E2E_LSTM(
            num_classes=train_dataset.num_classes,
            latent_dim=latent_dim,
            lstm_layers=1,
            hidden_dim=opt.hidden_dim,
            kml = opt.kml,
            masking=masking,
            )
    else:
        model = ConvLSTMIB(
            num_classes=train_dataset.num_classes,
            latent_dim=opt.latent_dim,
            lstm_layers=1,
            hidden_dim=opt.hidden_dim,
            kml = opt.kml,
            masking=masking,
            RI= opt.featureExtractor       
        )
    model = model.to(device)
    print(' \n   Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    #Create new directory to save sparce VIB-LSTM model checkpoints
    if not os.path.isdir(opt.save_dir):
        os.makedirs(opt.save_dir) 

    # Selecting mode of training- resume or start VIB-LSTM pruning or fine-tune VIB-LSTM model 
    if opt.resume != "":
        print("\n Resuming VIB-LSTM training from given checkpoint model")
        checkpoint=torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'])
        best_acc= checkpoint['test_acc'] 
        
    elif opt.checkpoint_model != "" :
        print("\n Matching keys between LSTM model and VIB-LSTM model...")
        modkeys= list(model.state_dict().keys())
        state_d= torch.load(opt.checkpoint_model)
        best_acc= 0
        lstm_keys = list(state_d.keys())
        #print("\n len ",lstm_keys)
        c=0
        if opt.checkpoint_model:
            for l in range(len(modkeys)):
                if "ib" in modkeys[l]:
                    continue
                else:
                    #print("\n mod key, lstm keys",modkeys[l],  lstm_keys[c])
                    model.state_dict()[modkeys[l]].copy_(state_d[lstm_keys[c]])
                    c=c+1
            print("\n keys matched. VIB-LSTM model ready to train.")
    
    elif opt.finetune_model !="":
        print("\n  Loading VIB-LSTM model to finetune")
        modkeys= list(model.state_dict().keys())
        checkpoint=torch.load(opt.finetune_model)
        state_d= checkpoint['state_dict']
        lstm_keys = list(state_d.keys())
        c=0
        for l in range(len(modkeys)): #to avoid dropout key error
            if "dropout" in modkeys[l]:
                print("\n dropout not copied")
                continue
            else:
                #print("\n mod key, lstm keys",modkeys[l],  lstm_keys[c])
                model.state_dict()[modkeys[l]].copy_(state_d[lstm_keys[c]])
                c=c+1
        print("\n Keys matched. VIB-LSTM model ready to fine-tune.")
        best_acc= checkpoint['test_acc']
    
    #Assigning optimizer - separate for VIB parameters and other model parameters.
    ib_param_list, ib_name_list, cnn_param_list, cnn_name_list = [], [], [], []
    for name, param in model.named_parameters():
        if 'z_mu' in name or 'z_logD' in name:
            ib_param_list.append(param)
            ib_name_list.append(name)
        else:
            cnn_param_list.append(param)
            cnn_name_list.append(name)    
    optimizer = torch.optim.Adam([{'params': ib_param_list, 'lr': opt.ib_lr, 'weight_decay': opt.weight_decay},
                                      {'params': cnn_param_list, 'lr': opt.lr, 'weight_decay': opt.weight_decay}])
    
    Best_acc = 0
    Best_epoch = 0
    
    def test_model(epoch, Best_acc, Best_epoch):
        """ Evaluate the model on the test set """
        top1 = AverageMeter()
        #print("\n Inside test_model")
        model.eval()
        test_metrics = {"loss": [], "acc": [], "time": []}
        for batch_i, (X, y) in enumerate(test_dataloader):
            prev_time = time.time()
            image_sequences = Variable(X.to(device), requires_grad=False) # get sequences
            labels = Variable(y, requires_grad=False).to(device) #get labels
            with torch.no_grad():
                # Reset VIB-LSTM hidden states
                model.lstm.reset_hidden_state(None)
                #predictions, compression_ratio, _ = model(image_sequences)
                predictions= model(image_sequences)
                if opt.e2e==True:
                    pruned_parameters= model.lstm.lstm.pruned_LSTM_pars #-for all matrices compression
                    num_features_pruned= model.lstm.lstm.num_features_pruned
                    compression_ratio=(model.lstm.lstm.tot_LSTM_pars)/(model.lstm.lstm.tot_LSTM_pars- pruned_parameters)
                else:
                    feature_mask= model.encoder.ib4.get_mask_hard() #feature mask trained with VIB
                    num_features_pruned= np.sum(feature_mask.cpu().numpy()==0)
                    pruned_parameters= model.lstm.lstm.pruned_LSTM_pars+ (num_features_pruned *4 *opt.hidden_dim) + ((model.lstm.lstm.num_states_pruned*4)*(opt.latent_dim - num_features_pruned))
                    compression_ratio=(model.lstm.lstm.tot_LSTM_pars)/(model.lstm.lstm.tot_LSTM_pars- pruned_parameters)
                time_req = datetime.timedelta(seconds=(time.time() - prev_time))
                test_metrics["time"].append(time_req)
            # Computing metrics
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            loss = cls_criterion(predictions, labels).item()
            top1.update(acc, X.size(0))
            # Tracking loss and accuracy
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)
            # Print test performance
            if (batch_i) % 8 ==0 or batch_i==len(test_dataloader)-1:
                sys.stdout.write(
                    "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%), tot/remain: %.5f,LSTM_dimensions- Input: %.5f, Hidden_state: %.5f]\n "
                    % (
                        batch_i,
                        len(test_dataloader),
                        loss,
                        np.mean(test_metrics["loss"]),
                        acc,
                        top1.avg,
                        compression_ratio,
                        (opt.latent_dim-num_features_pruned),
                        (opt.hidden_dim - model.lstm.lstm.num_states_pruned),
                        
                    )
                )
        Best_epoch = epoch if (top1.avg > Best_acc) else Best_epoch
        Best_acc = top1.avg if (top1.avg > Best_acc) else Best_acc
        print(f"\n Best_test_acc: {Best_acc} , at epoch: {Best_epoch}")
        model.train()
        print("")
        return top1.avg,Best_epoch
    

    model.train()
    kl_tot=0
    compression_ratio=0
    best_acc =0
    epochac=[]
    epochpar=[]
    if opt.eval: #to just evaluate
        model.eval()
        test_model(149, Best_acc, Best_epoch)
        exit()
    #Train VIB-LSTM for given number of epochs
    for epoch in range(opt.num_epochs): 
        epoch_metrics = {"loss": [], "acc": []}
        top1 = AverageMeter()
        prev_time = time.time()
        print(f"--- Epoch {epoch}/{opt.num_epochs} ---VIB_LR {optimizer.param_groups[0]['lr']}--LR {optimizer.param_groups[1]['lr']}---")
        for batch_i, (X, y) in enumerate(train_dataloader):
            """if X.size(0) == 1:
                continue
            """
            optimizer.zero_grad()
            # Reset LSTM hidden state
            model.lstm.reset_hidden_state(None)
            labels = Variable(y.to(device),requires_grad=False)
            image_sequences = Variable(X.to(device), requires_grad=False)
            #predictions, compression_ratio, kl_tot = model(image_sequences)
            predictions = model(image_sequences)
            # Computing metrics
            loss = cls_criterion(predictions, labels)
            #adding VIB loss to cross entropy
            if opt.e2e==True:
                VIB_LSTM_loss = model.lstm.lstm.viblstm_loss
                pruned_parameters= model.lstm.lstm.pruned_LSTM_pars #-for all matrices compression
                num_features_pruned= model.lstm.lstm.num_features_pruned
            else:
                feature_mask= model.encoder.ib4.get_mask_hard() #feature mask trained with VIB
                num_features_pruned= np.sum(feature_mask.cpu().numpy()==0)
                VIB_LSTM_loss =model.lstm.lstm.viblstm_loss +model.encoder.ib4.kld
                pruned_parameters= model.lstm.lstm.pruned_LSTM_pars+ (num_features_pruned *4 *opt.hidden_dim) + ((model.lstm.lstm.num_states_pruned*4)*(opt.latent_dim - num_features_pruned))
            loss += VIB_LSTM_loss* opt.kl_fac 
            if (model.lstm.lstm.tot_LSTM_pars- pruned_parameters)==0:
                print("\n Too high compression factor leads to 0 remaining parameters. Reduce compression fcator.\n Training ends.")
                break
            else:
                compression_ratio=(model.lstm.lstm.tot_LSTM_pars)/(model.lstm.lstm.tot_LSTM_pars- pruned_parameters)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            top1.update(acc, X.size(0))
            loss.backward()
            optimizer.step()
            # Tracking epoch metrics
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)
            # Determine approximate time left
            time_done = epoch * len(train_dataloader) + batch_i
            batches_left = opt.num_epochs * len(train_dataloader) - time_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print training performance 
            if (batch_i) %10 ==0 or batch_i==len(train_dataloader)-1:
                sys.stdout.write(
                    "\r[Batch %d/%d] [Total Loss: %f (%f), VIB_LSTM_loss: %f, VIB-feature_loss: %f, Acc: %.2f%% (%.2f%%), tot/remain: %.3f, L: %.3f, H:%.3f], ETA: %s \n"
                    % (
                        batch_i,
                        len(train_dataloader),
                        loss.item(),
                        np.mean(epoch_metrics["loss"]),
                        VIB_LSTM_loss* opt.kl_fac,
                        (model.encoder.ib4.kld if opt.e2e==False else model.lstm.lstm.ib4.kld)* opt.kl_fac,
                        acc,
                        top1.avg,
                        compression_ratio,#compression_ratio+(latent_dim*(512-model.lstm.lstm.num_features_pruned)),
                        (opt.latent_dim- num_features_pruned),
                        (opt.hidden_dim - model.lstm.lstm.num_states_pruned),
                        time_left,
                    )
                )
            
            # clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            #print("\nhid", hid, epoch_metrics["hid"] )

        # Evaluate the model on the test set
        test_acc, Best_epoch=test_model(epoch, best_acc, Best_epoch)
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        
        #Saving VIB-LSTM model checkpoint        
        if test_acc>10.0:
            torch.save({'test_acc': test_acc, 'state_dict': model.state_dict()}, f"{opt.save_dir}/{model.__class__.__name__}_acc{test_acc}_prun{compression_ratio}_L{opt.latent_dim}_H{opt.hidden_dim}.pth.tar")
        
