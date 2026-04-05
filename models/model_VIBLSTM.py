import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
#from torch.autograd import Variable
from torchvision.models import resnet152
from torchvision.models import inception_v3
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import utils as effutils
from enum import IntEnum
#from typing import Optional, Tuple
from .VIB_layer import *

##############################
#         Encoder
##############################
class Encoder(nn.Module):
    def __init__(self, latent_dim,kml,masking, RI):
        super(Encoder, self).__init__()
        if RI == 'Res':
            resnet = resnet152(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            fin=resnet.fc.in_features
        elif RI == 'Inc':
            self.feature_extractor = inception_v3(aux_logits=False,pretrained=True)
            fin=self.feature_extractor.fc.in_features
            self.feature_extractor.fc = Identity()
            #To train last 2 layers of inceptionnet-v3
            for child in list(self.feature_extractor.children())[-2:]:
                for param in child.parameters():
                    param.requires_grad = True
            for child in list(self.feature_extractor.children())[:-2]:
                for param in child.parameters():
                    param.requires_grad = False
        elif RI == 'Eff':
            self.model = EfficientNet.from_pretrained('efficientnet-b1',effutils.efficientnet(drop_connect_rate =0.4) ) #
            fin= 2048
        else:
            print("\n unknown feature extractor chosen..")
            return
        self.RI= RI
        self.ib4 = InformationBottleneck(fin, kl_mult= kml,masking=masking)
        #optional FC layer
        #self.final = nn.Sequential( nn.Linear(fin, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01))

    def forward(self, x):
        if self.RI== 'Inc':
            x = self.feature_extractor(x)
        elif self.RI== 'Eff':
                x = self.model.extract_features(x)
        else:
            with torch.no_grad():
                x = self.feature_extractor(x)
        x=self.ib4(x)
        x=x.view(x.size(0),-1)
        #x= self.final(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

##############################
#           LSTM
##############################

class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, kml, masking,e2e):
        super(LSTM, self).__init__()
        #self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm = VIBLSTM(latent_dim,hidden_dim, kml, masking,e2e)
        self.hidden_state = None

    def reset_hidden_state(self,hid):
        self.hidden_state = hid

    def forward(self, x):
        x = self.lstm(x)
        return x

##############################
#           VIB-LSTM
##############################
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
class VIBLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, kl_m: int, masking=False, e2e=False):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.e2e=e2e
        self.weight_ih_l0 = nn.Parameter(torch.Tensor( 4*hidden_sz,input_sz))
        self.weight_hh_l0= nn.Parameter(torch.Tensor(4*hidden_sz, hidden_sz))
        self.bias_ih_l0 = nn.Parameter(torch.Tensor(4*hidden_sz))
        self.bias_hh_l0= nn.Parameter(torch.Tensor(4*hidden_sz))
        self.init_weights() #comment when pruning by transfering knowledge of LSTM weights from pre-trained model.
        self.ib0 = InformationBottleneck(hidden_sz, kl_mult= kl_m, masking=masking)
        self.ib1 = InformationBottleneck(hidden_sz, kl_mult= kl_m, masking=masking)
        self.ib2 = InformationBottleneck(hidden_sz, kl_mult= kl_m, masking=masking)
        self.ib3 = InformationBottleneck(hidden_sz, kl_mult= kl_m, masking=masking)
        if e2e==True:
            self.ib4 = InformationBottleneck(input_sz, kl_mult= kl_m, masking= masking)
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):

        bs, seq_sz, _ = x.size()
        hidden_seq = []
        h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        for t in range(seq_sz): # iterate over the time steps
            x_t = x[:, t, :]
            if self.e2e== True:
                x_ib_t = self.ib4(x_t)
                gateo = F.linear(x_ib_t,self.weight_ih_l0, self.bias_ih_l0) + F.linear(h_t,self.weight_hh_l0, self.bias_hh_l0)
            else:
                gateo = F.linear(x_t,self.weight_ih_l0, self.bias_ih_l0) + F.linear(h_t,self.weight_hh_l0, self.bias_hh_l0)
            ing,forgetg,cellg,outg = gateo.chunk(4,1)
            i_t = torch.sigmoid(ing)
            i_ib_t= self.ib0(i_t)
            f_t = torch.sigmoid(forgetg)
            f_ib_t= self.ib1(f_t)
            g_t = torch.tanh(cellg)
            g_ib_t= self.ib2(g_t)
            o_t = torch.sigmoid(outg)
            o_ib_t= self.ib3(o_t)
            c_t = f_ib_t * c_t + i_ib_t * g_ib_t
            h_t = o_ib_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        #VIB masks
        i_mask= self.ib0.get_mask_hard()
        f_mask= self.ib1.get_mask_hard()
        g_mask= self.ib2.get_mask_hard()
        o_mask= self.ib3.get_mask_hard()
        gate_mask=i_mask.cpu().numpy() * g_mask.cpu().numpy() * o_mask.cpu().numpy()
        self.num_states_pruned= np.sum(gate_mask==0)
        #calculating total LSTM parameters and pruned parameters
        if self.e2e==True:
            x_mask= self.ib4.get_mask_hard()
            self.num_features_pruned= np.sum(x_mask.cpu().numpy()==0)
            self.viblstm_loss= self.ib0.kld+self.ib1.kld+self.ib2.kld+self.ib3.kld+self.ib4.kld
            self.pruned_LSTM_pars= (self.num_features_pruned *4 *self.hidden_size) + ((self.num_states_pruned*4)*(self.input_size - self.num_features_pruned)) + (self.num_states_pruned*4*self.hidden_size) + (4*self.num_states_pruned*(self.hidden_size-self.num_states_pruned)) + 2*(4*self.num_states_pruned) #taking all LSTM matrices
        else:
            self.viblstm_loss= self.ib0.kld+self.ib1.kld+self.ib2.kld+self.ib3.kld
            self.pruned_LSTM_pars= (self.num_states_pruned*4*self.hidden_size) + (4*self.num_states_pruned*(self.hidden_size-self.num_states_pruned)) + 2*(4*self.num_states_pruned) #taking only the hh matrices
        self.tot_LSTM_pars= (4*self.hidden_size*(self.hidden_size+ self.input_size))+(8*self.hidden_size)
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq

##############################
#         ConvLSTM
##############################

class ConvLSTMIB(nn.Module):
    def __init__(
        self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=1024, kml=1, masking=False, RI='Inc',e2e=False):
        super(ConvLSTMIB, self).__init__()
        self.encoder = Encoder(latent_dim,kml,masking, RI)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, kml, masking,e2e)
        self.lstm.final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            #nn.Dropout(0.5),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
        x = x[:, -1]
        return self.lstm.final(x)


##############################
#     End-to-end VIB-LSTM
##############################

class E2E_LSTM(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=1024, kml=1, masking=False, e2e=True):
        super(E2E_LSTM, self).__init__()
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, kml,masking,e2e)
        self.drop=nn.Dropout(0.5)
        self.lstm.final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
        x=self.drop(x)
        x = x[:, -1]
        return self.lstm.final(x)
