import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.models import resnet152
from torchvision.models import inception_v3
from enum import IntEnum
from typing import Optional, Tuple
#from ib_layers import *
#from resnet152 import *
import math
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn.modules import Module
from torch.nn.modules import utils
import pdb
##############################
#         Encoder
##############################
#np.random.seed(100)

class Encoder(nn.Module):
    def __init__(self, latent_dim, ib4m,RI):
        super(Encoder, self).__init__()
        if RI == 'Res':
            resnet = resnet152(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            fin=resnet.fc.in_features
        elif RI == 'Inc':
            self.feature_extractor = inception_v3(aux_logits=False,pretrained=True)
            fin=self.feature_extractor.fc.in_features
            self.feature_extractor.fc = Identity()
        elif RI == 'Eff':
            self.model = EfficientNet.from_pretrained('efficientnet-b1',effutils.efficientnet(drop_connect_rate =0.4) ) #
            fin= 2048
        else:
            print("\n unknown feature extractor chosen..")
            return
        self.RI= RI
        self.ib4 = InformationBottleneck(fin, masking=False)
        self.init_weights(ib4m)
        #optional FC layer
        #self.final = nn.Sequential( nn.Linear(fin, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01))

    def init_weights(self, ib4m):
        self.ib4.mask= ib4m

    def forward(self, x):
        if self.RI== 'Inc':
            x = self.feature_extractor(x)
        elif self.RI== 'Eff':
                x = self.model.extract_features(x)
        else:
            with torch.no_grad():
                x = self.feature_extractor(x)
        x=self.ib4(x)
        x = x.view(x.size(0), -1)
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
    def __init__(self, lstminp, num_layers, hidden_dim):
        super(LSTM, self).__init__()
        self.lstm = VIBLSTM(lstminp, hidden_dim)
        self.hidden_state = None

    def reset_hidden_state(self,hid):
        self.hidden_state = None

    def forward(self, x):
        x= self.lstm(x,self.hidden_state)
        return x


##############################
#           Naive-LSTM
##############################
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
class VIBLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        self.weight_ih_l0 = nn.Parameter(torch.Tensor( 4*hidden_sz,input_sz))
        self.weight_hh_l0= nn.Parameter(torch.Tensor(4*hidden_sz, hidden_sz))
        self.bias_ih_l0 = nn.Parameter(torch.Tensor(4*hidden_sz))
        self.bias_hh_l0= nn.Parameter(torch.Tensor(4*hidden_sz))
        #self.init_weights() uncomment only when training from scratch
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
    def forward(self, x: torch.Tensor, init_states: torch.Tensor):

        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states== None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states.to(x.device), init_states.to(x.device)
        for t in range(seq_sz): # iterate over the time steps
            x_t = x[:, t, :]
            #x_ib_t = self.ib4(x_t)
            gateo = ((x_t @ self.weight_ih_l0.data.T )+ self.bias_ih_l0.data) + ((h_t@self.weight_hh_l0.data.T)+self.bias_hh_l0)
            ing,forgetg,cellg,outg = gateo.chunk(4,1)
            i_t = torch.sigmoid(ing)
            f_t = torch.sigmoid(forgetg)
            g_t = torch.tanh(cellg)
            o_t = torch.sigmoid(outg)
            c_t = f_t * c_t + i_t * g_t
            h_t = (o_t * torch.tanh(c_t))
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq.cuda()

##############################
#         ConvLSTM
##############################
class ConvLSTMcomp(nn.Module):
    def __init__(
        self, num_classes, orig_hidden_dim,ib4m,latent_dim=512, lstm_layers=1, seq=40, hidden_dim=1024, RI= 'Inc'):
        super(ConvLSTMcomp, self).__init__()
        self.encoder = Encoder(latent_dim, ib4m,RI)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim)
        self.lstm.final = nn.Sequential(
            nn.Linear( hidden_dim, orig_hidden_dim),
            #nn.Dropout(0.5),
            nn.BatchNorm1d(orig_hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(orig_hidden_dim, num_classes),
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
#         VIB_layer-inference
##############################
class InformationBottleneck(Module):
    def __init__(self, dim, mask_thresh=0, mask=None,divide_w=False, masking=False):
        super(InformationBottleneck, self).__init__()
        self.post_z_mu = Parameter(torch.Tensor(dim))
        self.mask=mask
        self.masking=masking
        self.mask_thresh = mask_thresh

    def adapt_shape(self, src_shape, x_shape):
        # to distinguish conv layers and fc layers
        # see if we need to expand the dimension of x
        new_shape = src_shape if len(src_shape)==2 else (1, src_shape[0])
        if len(x_shape)>2:
            new_shape = list(new_shape)
            new_shape += [1 for i in range(len(x_shape)-2)]
        return new_shape

    def get_mask_weighted(self, threshold=0): #if weighted mask was used during training
        #logalpha = self.logal
        mask = self.post_z_mu.data #* (logalpha < threshold).float()
        return mask

    def forward(self, x):
        if self.masking:
            return x
        num=np.where(self.mask==1) #wherever mask==1

        new_shape = self.adapt_shape(self.mask.size(), x.size())#adapt mask to dimensions of x
        masked_x=x.cpu()*self.mask.view(new_shape) #multiply x with mask
        #print("\n shape masked_x", masked_x.size())

        new_x= masked_x[:,num].view(x.size(0),-1).cuda()#adjust new_x dimensions
        #print("\n new shape", new_x.size())
        return new_x
