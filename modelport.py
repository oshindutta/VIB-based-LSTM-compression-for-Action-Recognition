import argparse
import os
import sys
import torch
import torch.nn as nn
from models.model_VIBLSTM import *
from models.modelcomp import *
from datasets.datasetucf11 import *
from torch.utils.data import DataLoader
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

parser = argparse.ArgumentParser()
parser.add_argument('--VIBmodel', default='', type=str, metavar='PATH',help='path to sparse VIB model checkpoint')
parser.add_argument("--dataset_path", type=str, default="UCF11_updated_mpg-frames", help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=32, help="Size of each training batch")
parser.add_argument("--sequence_length", type=int, default=32, help="Number of frames in each sequence")
parser.add_argument("--img_dim", type=int, default=299, help="Height or width dimension")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--latent_dim", type=int, default=2048, help="Dimensionality of the latent representation of un-pruned model")
parser.add_argument("--hidden_dim", type=int, default=2048, help="Dimensionality of the hidden representation of un-pruned model")
parser.add_argument('--featureExtractor', type=str, default='Inc',help='Select feature extractor- Res for resnet152, Inc for Inceptionnet-v3, Eff for Efficientnet-b1')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

image_shape = (opt.channels, opt.img_dim, opt.img_dim) #assuming square shaped image
test_dataset = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=False,
        )

test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

def main():
    
    latent_dim=opt.latent_dim
    hidden_dim=opt.hidden_dim
    modelIB = ConvLSTMIB(
            num_classes=test_dataset.num_classes,
            latent_dim=opt.latent_dim,
            lstm_layers=1,
            hidden_dim=opt.hidden_dim,
            RI= opt.featureExtractor       
            )
    if opt.VIBmodel:
        # Load checkpoint.
        print('==> Loading IB model')
        checkpoint = torch.load(opt.VIBmodel, map_location=device)
        state_dict= checkpoint['state_dict']
        modelIB.load_state_dict(checkpoint['state_dict'])
        
        #get common mask for LSTM gates and feature encoder
        i_mask= modelIB.lstm.lstm.ib0.get_mask_hard()
        f_mask= modelIB.lstm.lstm.ib1.get_mask_hard()
        g_mask= modelIB.lstm.lstm.ib2.get_mask_hard()
        o_mask= modelIB.lstm.lstm.ib3.get_mask_hard()
        gate_mask=torch.from_numpy(i_mask.cpu().numpy() * g_mask.cpu().numpy() * o_mask.cpu().numpy())
        feature_mask= modelIB.encoder.ib4.get_mask_hard()
        
        #Print number of states and features pruned
        num_features_pruned= np.sum(feature_mask.cpu().numpy()==0)
        num_states_pruned= np.sum(gate_mask.cpu().numpy()==0)
        print("\n num_features_pruned= %d and num_states_pruned= %d "%(num_features_pruned, num_states_pruned))

        # Print the compression ratio
        LSTMtotw= (4*hidden_dim*(hidden_dim+ latent_dim))+(8*hidden_dim)
        remain_parameters= (4*(hidden_dim-num_states_pruned)*(hidden_dim-num_states_pruned + latent_dim-num_features_pruned))+(8*(hidden_dim-num_states_pruned))
        print("\n LSTM Compression Ratio(Total/Remaining parameters)= ", LSTMtotw/remain_parameters)
        
        # non zero weights in new matrices 
        w_h_mask=np.concatenate((gate_mask,gate_mask,gate_mask,gate_mask),axis=0)
        w_h_mask=torch.from_numpy(w_h_mask.reshape(4*hidden_dim,1)) # * np.ones(4*hidden_dim,latent_dim)) #.to(device)
        w_i_mask=feature_mask.reshape(1,latent_dim)
        w={}
        w[0] = w_i_mask* (w_h_mask * state_dict["lstm.lstm.weight_ih_l0"].data.cpu())
        w[1] = gate_mask.reshape(1,hidden_dim)* (w_h_mask*state_dict["lstm.lstm.weight_hh_l0"].data.cpu())
        w[1] = (w[1][np.where(w[1] !=0)].reshape(4*(hidden_dim-num_states_pruned),(hidden_dim-num_states_pruned)))
        w[0] = w[0][np.where(w[0] !=0)].reshape(4*(hidden_dim-num_states_pruned),(latent_dim-num_features_pruned))
        w[2]= (w_h_mask.flatten().cuda()*state_dict["lstm.lstm.bias_ih_l0"]).data.cpu()
        w[3]= (w_h_mask.flatten().cuda()*state_dict["lstm.lstm.bias_hh_l0"]).data.cpu()
        w[2]= w[2][np.where(w[2] !=0)]
        w[3]= w[3][np.where(w[3] !=0)]
        #save feature mask- for accumulating non zero feature values
        ib4mask= feature_mask
        torch.save(feature_mask, "CompressedIBmodels/mask.pt")
        
        #smaller FC matrices
        lstmFC= gate_mask.reshape(1,hidden_dim)* state_dict["lstm.final.0.weight"].data.cpu()
        lstmFC= lstmFC[np.where(lstmFC !=0)].reshape(hidden_dim,(hidden_dim-num_states_pruned))
        #print("\n size of lstmfc", feature_mask.size(), state_dict["encoder.final.0.bias"].size())
        """ #if encoder.final present in the model- incase of optional FC kayer in encoder
        enFC= (feature_mask.reshape(latent_dim,1).cpu()* state_dict["encoder.final.0.weight"].data.cpu())[np.where((feature_mask.reshape(latent_dim,1).cpu()* state_dict["encoder.final.0.weight"].data.cpu()) !=0)].reshape((latent_dim-num_features_pruned),2048)
        enFCb= (feature_mask.cpu()* state_dict["encoder.final.0.bias"].data.cpu())[np.where((feature_mask.cpu()* state_dict["encoder.final.0.bias"].data.cpu()) !=0)]
        enFC1=(feature_mask.cpu()* state_dict["encoder.final.1.weight"].data.cpu())[np.where((feature_mask.cpu()* state_dict["encoder.final.1.weight"].data.cpu()) !=0)]
        enFC1b=(feature_mask.cpu()* state_dict["encoder.final.1.bias"].data.cpu())[np.where((feature_mask.cpu()* state_dict["encoder.final.1.bias"].data.cpu())!=0)]
        enFC1c=(feature_mask.cpu()* state_dict["encoder.final.1.running_mean"].data.cpu())[np.where((feature_mask.cpu()* state_dict["encoder.final.1.running_mean"].data.cpu()) !=0)]
        enFC1d=(feature_mask.cpu()* state_dict["encoder.final.1.running_var"].data.cpu())[np.where((feature_mask.cpu()* state_dict["encoder.final.1.running_var"].data.cpu()) !=0)]
        #print("\n size of lstmfc", enFC.size())
        """

       	#initialize new compressred model
       	modelComp= ConvLSTMcomp(
	        num_classes=test_dataset.num_classes,
            orig_hidden_dim= hidden_dim,
	        ib4m=ib4mask,
            latent_dim=latent_dim-num_features_pruned,
	        lstm_layers=1,
	        hidden_dim= hidden_dim-num_states_pruned,
	        RI= opt.featureExtractor      
	       )

       	#copy smaller matrices to compressed model keys
       	print(' \n   Total params in IB model: %.2fM' % (sum(p.numel() for p in modelIB.parameters())/1000000.0))
        print(' \n   Total params in compressed In model: %.2fM' % (sum(p.numel() for p in modelComp.parameters())/1000000.0))
        par= sum(p.numel() for p in modelComp.parameters())/1000000.0
        compkeys= list(modelComp.state_dict().keys()) #keys of compressed model
        ibkeys =list(modelIB.state_dict().keys()) #keys of uncompressed model
        lenIB= len(ibkeys)
        print("\n number of keys- in uncompressed model= %d, in compressed model= %d" %(len(compkeys),lenIB))
        c=0
        myit= iter(range(0,lenIB))
        for l in myit:
            """if "encoder.final" in ibkeys[l]: #if encoder.final is present in modelIB trained
                #print("\n encfinal keys",c,compkeys[c], ibkeys[l])
                modelComp.state_dict()[compkeys[c]].copy_(enFC)
                c=c+1
                next(myit, None)
                #print("\n l=", next(myit, None))
                modelComp.state_dict()[compkeys[c]].copy_(enFCb)
                c=c+1
                next(myit, None)
                #print("\n l=", next(myit, None))
                modelComp.state_dict()[compkeys[c]].copy_(enFC1)
                c=c+1
                next(myit, None)
                #print("\n l=", next(myit, None))
                modelComp.state_dict()[compkeys[c]].copy_(enFC1b)
                c=c+1
                next(myit, None)
                #print("\n l=", next(myit, None))
                modelComp.state_dict()[compkeys[c]].copy_(enFC1c)
                c=c+1
                next(myit, None)
                #print("\n l=", next(myit, None))
                modelComp.state_dict()[compkeys[c]].copy_(enFC1d)
                c=c+1
                #print("\n l=", l,next(myit, None))
                l=next(myit,None)
                print("\n encfinal keys here",c,compkeys[c], ibkeys[l])
                modelComp.state_dict()[compkeys[c]].copy_(state_dict[ibkeys[l]])
                c=c+1"""
            if "encoder.ib4" in compkeys[c]:
                c=c+1
                continue
            elif "lstm.final" in ibkeys[l]:
                if compkeys[c] == "lstm.final.0.weight":
                    modelComp.state_dict()[compkeys[c]].copy_(lstmFC)
                    c=c+1
                else:
                    modelComp.state_dict()[compkeys[c]].copy_(modelIB.state_dict()[ibkeys[l]])
                    c=c+1
            elif "encoder.feature_extractor" in ibkeys[l]:
                modelComp.state_dict()[compkeys[c]].copy_(state_dict[ibkeys[l]])
                c=c+1
            elif "lstm.lstm" in compkeys[c]:
                for k in range(0,4):
                    #print("\n lstm lstm keys",c, compkeys[c])
                    modelComp.state_dict()[compkeys[c]].copy_(w[k])
                    c=c+1
                    l=next(myit,None)
            else:
                continue
          
        print(" Compressed model- keys matched")
       	
        #Test the compressed model
        cls_criterion = nn.CrossEntropyLoss().to(device)
        del modelIB
        modelComp=modelComp.to(device)
        modelComp.eval()
       	print("\nTesting Compressed model")
        test_metrics = {"loss": [], "acc": [],"time":[]}
        top1=AverageMeter()
        for batch_i, (X, y) in enumerate(test_dataloader):
            image_sequences = Variable(X.to(device), requires_grad=False)
            labels = Variable(y, requires_grad=False).to(device)
            with torch.no_grad():
                modelComp.lstm.reset_hidden_state(None) # Reset LSTM hidden state
                predictions= modelComp(image_sequences) #Get sequence predictions
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            top1.update(acc, X.size(0))
            loss = cls_criterion(predictions, labels).item()
            # Keep track of loss and accuracy
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)
            # Log test performance
            if batch_i % 1 ==0:
                sys.stdout.write(
                    "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]\n "
                    % (
                        batch_i,
                        len(test_dataloader),
                        loss,
                        np.mean(test_metrics["loss"]),
                        acc,
                        top1.avg,
                        
                    )
                )
        accul=top1.avg
        os.makedirs("CompressedIBmodels", exist_ok=True)
        torch.save({'latent_dim': (latent_dim-num_features_pruned), 'hidden_dim': (hidden_dim- num_states_pruned), 'FC_dim':hidden_dim, 'state_dict': modelComp.state_dict()}, f"CompressedIBmodels/{modelComp.__class__.__name__}_{accul}_Par{par}M.pth.tar")
    else:
        print("\n No input sparse VIB_model to convert ")
if __name__ == '__main__':
    main()

