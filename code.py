!pip install torchinfo

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from torchinfo import summary
# ==============================
# Abstract Dataset
# ==============================
class AlphaDatasetBase(ABC):
    def __init__(self): pass

    @abstractmethod
    def get_batch(self, batch_size, batch_index): ...

    @abstractmethod
    def num_batches(self, batch_size): ...

    @abstractmethod
    def get_full(self): ...


# ==============================
# Concrete Dataset (Panel-style)
# ==============================
class AlphaDataset:
    def __init__(self, data: pd.DataFrame, lookback: int, split_: float = None,input_norm=100., augment_neg: bool = True,tf : str=None):
        """
        data: pd.DataFrame with time index and assets as columns.
        lookback: how many days of history to use.
        split_index: number of rows to use for training. Remaining is test.
        """
        self.lookback = lookback
        self.augment_neg = augment_neg
        self.input_scale = input_norm
        # Preprocessing: returns & dropna
        tmp = data.copy()
        print(tmp.shape)
        #tmp=tmp.resample('4h').last()

        self.data = (tmp.diff()/tmp.shift(1)).dropna()

        self.full_data = self.data

        if split_ is None:
          split_=0.7
        split_index = int(len(self.full_data) * split_)
        split_index2 = int(len(data) * split_)
        data_in=[]
        print(split_)
        #for H in range(0,4):
          #tmp = data.iloc[:split_index2].resample('4h',offset = f'{H}h0min').last()
          #print(tmp.shape)
          #data_in.append((tmp.diff()/tmp.shift(1)).dropna())
        #data_in=pd.concat(data_in,axis=0)
        #print(data_in.shape)
        print(tmp.shape)
        self.data = (tmp.diff()/tmp.shift(1)).dropna()
        print(self.data.shape)
        #self.train_data = self._preprocess(data_in)
        self.train_data = self._preprocess(self.full_data[:split_index])
        print(self.train_data.shape)
        self.test_data = self.full_data[split_index:]

        self.train_inputs,self.train_rets,self.train_idx = self._make_inputs(self.train_data)
        self.test_inputs,self.test_rets,self.test_idx = self._make_inputs(self.test_data)

    def _preprocess(self, data):
        if self.augment_neg:
            return pd.concat([data, -data], axis=0).dropna()
        return data.dropna()

    def _make_inputs(self, df: pd.DataFrame) -> np.ndarray:
        rets_ = df.dropna()
        rets_arr=rets_.to_numpy()
        input_arr,idx_select = preprocess_cumsum(rets_arr,lookback=self.lookback)
        rets_arr  = rets_arr[self.lookback:,:]
        idx_arr =  rets_.index[self.lookback:]
        return input_arr.astype(np.float32)*self.input_scale,rets_arr.astype(np.float32),idx_arr

    def get_train_batch(self, batch_size, batch_idx):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(self.train_inputs))
        input_arr_batch = self.train_inputs[start:end,:]
        rets_arr_batch = self.train_rets[start:end,:]
        t_obs,N,_ = input_arr_batch.shape
        input_nn = input_arr_batch.reshape(t_obs*N,self.lookback).astype('float32')
        #print('****')
        #print(input_nn.shape)
        #print(batch_idx)
        #print(batch_size)
        #print(start)
        #print(end)
        return input_nn,rets_arr_batch,t_obs,N

    def get_test_batch(self, batch_size, batch_idx):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(self.test_inputs))
        input_arr_batch = self.test_inputs[start:end,:]
        rets_arr_batch = self.test_rets[start:end,:]
        t_obs,N,_ = input_arr_batch.shape
        input_nn = input_arr_batch.reshape(t_obs*N,self.lookback).astype('float32')
        return input_nn,rets_arr_batch,t_obs,N

    def num_train_batches(self, batch_size):
        return int(np.floor(len(self.train_inputs) / batch_size))

    def num_test_batches(self, batch_size):
        return int(np.floor(len(self.test_inputs) / batch_size))

    def get_train_full(self):
        return self.train_inputs

    def get_test_full(self):
        return self.test_inputs

class AlphaDataset_feat:
    def __init__(self, data: pd.DataFrame,  split_: float = None,input_norm=100., augment_neg: bool = True,tf : int=4):
        """
        data: pd.DataFrame with time index and assets as columns.
        lookback: how many days of history to use.
        split_index: number of rows to use for training. Remaining is test.
        """
        self.tf=tf
        self.augment_neg = augment_neg
        self.input_scale = input_norm
        # Preprocessing: returns & dropna
        tmp = data.copy()

        if split_ is None:
          split_=0.7
        split_index = int(len(tmp) * split_)

        data_test = data.iloc[split_index:,:].resample(f'{tf}h').last()
        data_test=(data_test.diff()/data_test.shift(1)).dropna()

        data_df_in = data.iloc[:split_index]
        data_in=[]
        print(split_)
        for H in range(0,tf):
          tmp = data_df_in.resample(f'{tf}h',offset = f'{H}h0min').last()
          print(tmp.shape)
          data_in.append((tmp.diff()/tmp.shift(1)).dropna())
        data_in=pd.concat(data_in,axis=0)
        #print(data_in.shape)
        print(tmp.shape)
        self.data = (tmp.diff()/tmp.shift(1)).dropna()
        print(self.data.shape)
        self.train_data = self._preprocess(data_in)
        self.test_data = data_test
        self.train_inputs,self.train_rets,self.train_idx = self._make_inputs(self.train_data)
        self.test_inputs,self.test_rets,self.test_idx = self._make_inputs(self.test_data)

    def _preprocess(self, data):
        if self.augment_neg:
            return pd.concat([data, -data], axis=0).dropna()
        return data.dropna()

    def _make_inputs(self, df: pd.DataFrame) -> np.ndarray:

        df=df.dropna()
        df_ret = df.copy()
        df_feat = df.shift(1).copy()

        feat_list =[]
        feat_list.append(df_feat.fillna(0).to_numpy())
        feat_list.append(df_feat.shift(1).fillna(0).to_numpy())
        feat_list.append(df_feat.shift(2).fillna(0).to_numpy())
        feat_list.append(df_feat.shift(3).fillna(0).to_numpy())
        tf = 5
        #feat_list.append(df_feat.rolling(tf).mean().fillna(0).to_numpy())

        tmp = (df_feat -df_feat.rolling(tf).min())/(df_feat.rolling(tf).max() -df_feat.rolling(tf).min())
        tmp=tmp.fillna(0)
        feat_list.append(tmp.to_numpy()/20)

        input_arr = np.stack(feat_list,axis=2)
        rets_arr  = df_ret.to_numpy()
        idx_arr =  df_ret.index
        self.lookback = input_arr.shape[-1]
        print(input_arr.shape)
        return input_arr.astype(np.float32)*self.input_scale,rets_arr.astype(np.float32),idx_arr

    def get_train_batch(self, batch_size, batch_idx):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(self.train_inputs))
        #print('***')
       # print(self.train_inputs.shape)
        input_arr_batch = self.train_inputs[start:end,:]
        rets_arr_batch = self.train_rets[start:end,:]
        t_obs,N,_ = input_arr_batch.shape
        #print(input_arr_batch.shape)
        #print(self.lookback)
        input_nn = input_arr_batch.reshape(t_obs*N,self.lookback).astype('float32')
        #print('****')
        #print(input_nn.shape)
        #print(batch_idx)
        #print(batch_size)
        #print(start)
        #print(end)
        return input_nn,rets_arr_batch,t_obs,N

    def get_test_batch(self, batch_size, batch_idx):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(self.test_inputs))
        input_arr_batch = self.test_inputs[start:end,:]
        rets_arr_batch = self.test_rets[start:end,:]
        t_obs,N,_ = input_arr_batch.shape
        input_nn = input_arr_batch.reshape(t_obs*N,self.lookback).astype('float32')
        return input_nn,rets_arr_batch,t_obs,N

    def num_train_batches(self, batch_size):
        return int(np.floor(len(self.train_inputs) / batch_size))

    def num_test_batches(self, batch_size):
        return int(np.floor(len(self.test_inputs) / batch_size))

    def get_train_full(self):
        return self.train_inputs

    def get_test_full(self):
        return self.test_inputs
class AlphaDataset_norm:
    def __init__(self, data: pd.DataFrame,  split_: float = None,input_norm=100., augment_neg: bool = True,tf : int=4):
        """
        data: pd.DataFrame with time index and assets as columns.
        lookback: how many days of history to use.
        split_index: number of rows to use for training. Remaining is test.
        """
        self.tf=tf
        self.augment_neg = augment_neg
        self.input_scale = input_norm
        # Preprocessing: returns & dropna
        tmp = data.copy()

        if split_ is None:
          split_=0.7
        split_index = int(len(tmp) * split_)

        data_test = data.iloc[split_index:,:].resample(f'{tf}h').last()
        data_test=(data_test.diff()/data_test.shift(1)).dropna()

        data_df_in = data.iloc[:split_index]
        data_in=[]
        print(split_)
        for H in range(0,tf):
          tmp = data_df_in.resample(f'{tf}h',offset = f'{H}h0min').last()
          print(tmp.shape)
          data_in.append((tmp.diff()/tmp.shift(1)).dropna())
        data_in=pd.concat(data_in,axis=0)
        #print(data_in.shape)
        print(tmp.shape)
        self.data = (tmp.diff()/tmp.shift(1)).dropna()
        print(self.data.shape)
        self.train_data = self._preprocess(data_in)
        self.test_data = data_test
        self.train_inputs,self.train_rets,self.train_idx = self._make_inputs(self.train_data)
        self.test_inputs,self.test_rets,self.test_idx = self._make_inputs(self.test_data)

    def _preprocess(self, data):
        if self.augment_neg:
            return pd.concat([data, -data], axis=0).dropna()
        return data.dropna()

    def _make_inputs(self, df: pd.DataFrame) -> np.ndarray:

        df=df.dropna()
        df_ret = df.copy()
        df_feat = df.shift(1).copy()

        feat_list =[]
        feat_list.append(df_feat.fillna(0).to_numpy())
        feat_list.append(df_feat.shift(1).fillna(0).to_numpy())
        feat_list.append(df_feat.shift(2).fillna(0).to_numpy())
        #feat_list.append(df_feat.shift(3).fillna(0).to_numpy())
        tf = 5
        #feat_list.append(df_feat.rolling(tf).mean().fillna(0).to_numpy())

        tmp = (df_feat -df_feat.rolling(tf).min())/(df_feat.rolling(tf).max() -df_feat.rolling(tf).min())
        tmp=tmp.fillna(0)
        #feat_list.append(tmp.to_numpy()/20)

        input_arr = np.stack(feat_list,axis=2)
        rets_arr  = df_ret.to_numpy()
        idx_arr =  df_ret.index
        self.lookback = input_arr.shape[-1]
        print(input_arr.shape)
        return input_arr.astype(np.float32)*self.input_scale,rets_arr.astype(np.float32),idx_arr

    def get_train_batch(self, batch_size, batch_idx):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(self.train_inputs))
        #print('***')
       # print(self.train_inputs.shape)
        input_arr_batch = self.train_inputs[start:end,:]
        rets_arr_batch = self.train_rets[start:end,:]
        t_obs,N,_ = input_arr_batch.shape
        #print(input_arr_batch.shape)
        #print(self.lookback)
        input_nn = input_arr_batch.reshape(t_obs*N,self.lookback).astype('float32')
        #print('****')
        #print(input_nn.shape)
        #print(batch_idx)
        #print(batch_size)
        #print(start)
        #print(end)
        return input_nn,rets_arr_batch,t_obs,N

    def get_test_batch(self, batch_size, batch_idx):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(self.test_inputs))
        input_arr_batch = self.test_inputs[start:end,:]
        rets_arr_batch = self.test_rets[start:end,:]
        t_obs,N,_ = input_arr_batch.shape
        input_nn = input_arr_batch.reshape(t_obs*N,self.lookback).astype('float32')
        return input_nn,rets_arr_batch,t_obs,N

    def num_train_batches(self, batch_size):
        return int(np.floor(len(self.train_inputs) / batch_size))

    def num_test_batches(self, batch_size):
        return int(np.floor(len(self.test_inputs) / batch_size))

    def get_train_full(self):
        return self.train_inputs

    def get_test_full(self):
        return self.test_inputs
# ==============================
# CNN Block
# ==============================


# ==============================
# Alpha Model Base
# ==============================
class AlphaModelBase(nn.Module, ABC):
    def __init__(self): super().__init__()

    @abstractmethod
    def forward(self, x): ...


# ==============================
# CNN Alpha Model
# ==============================
class CNN_Block(AlphaModelBase):
    def __init__(self, in_filters=1, out_filters=8, normalization=True, filter_size=2,residual_connection=True):
        super(CNN_Block, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.residual_connection=residual_connection
        self.conv1 = nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=filter_size,
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=out_filters, out_channels=out_filters, kernel_size=filter_size,
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)
        self.left_zero_padding = nn.ConstantPad1d((filter_size-1,0),0)

        self.normalization1 = nn.InstanceNorm1d(in_filters)
        self.normalization2 = nn.InstanceNorm1d(out_filters)
        self.normalization = normalization

    def forward(self, x): #x and out have dims (N,C,T) where C is the number of channels/filters
        if self.normalization:
            x = self.normalization1(x)
        out = self.left_zero_padding(x)
        out = self.conv1(out)
        out = self.relu(out)
        if self.normalization:
            out = self.normalization2(out)
        out = self.left_zero_padding(out)
        out = self.conv2(out)
        out = self.relu(out)
        if self.residual_connection:
          out = out + x.repeat(1,int(self.out_filters/self.in_filters),1)
        return out

class CNNAlphaModel(AlphaModelBase):
  def __init__(self,
                logdir,
                random_seed = 0,
                device = "cpu", # other options for device are e.g. "cuda:0"
                normalization_conv = True,
                filter_numbers = [1,8],
                attention_heads = 4,
                use_convolution = True,
                hidden_units = 2*8,
                hidden_units_factor = 2,
                dropout = 0.25,
                filter_size = 2,
                use_transformer = True,
                residual_connection=True):

      super(CNNAlphaModel, self).__init__()
      if hidden_units and hidden_units_factor and hidden_units != hidden_units_factor * filter_numbers[-1]:
          raise Exception(f"`hidden_units` conflicts with `hidden_units_factor`; provide one or the other, but not both.")
      if hidden_units_factor:
          hidden_units = hidden_units_factor * filter_numbers[-1]
      self.logdir = logdir
      self.random_seed = random_seed
      torch.manual_seed(self.random_seed)
      self.device = torch.device(device)
      self.filter_numbers = filter_numbers
      self.use_transformer = use_transformer
      self.use_convolution = use_convolution and len(filter_numbers) > 0
      self.is_trainable = True

      self.convBlocks = nn.ModuleList()
      for i in range(len(filter_numbers)-1):
          self.convBlocks.append(
              CNN_Block(filter_numbers[i],filter_numbers[i+1],normalization=normalization_conv,filter_size=filter_size,residual_connection=residual_connection))
      #self.encoder = nn.TransformerEncoderLayer(d_model=filter_numbers[-1], nhead=attention_heads, dim_feedforward=hidden_units, dropout=dropout)
      self.linear = nn.Linear(filter_numbers[-1],1)
      #self.softmax = nn.Sequential(nn.Linear(filter_numbers[-1],num_classes))#,nn.Softmax(dim=1))

  def forward(self,x): #x has dimension (N,T)
      N,T = x.shape
      x = x.reshape((N,1,T))  #(N,1,T)
      if self.use_convolution:
          for i in range(len(self.filter_numbers)-1):
              x = self.convBlocks[i](x) #(N,C,T), C is the number of channels/features
      x = x.permute(2,0,1)

      return self.linear(x[-1,:,:]).squeeze() #this outputs the weights #self.softmax(x[-1,:,:]) #(N,num_classes)

class FullyConnectedNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=64,
                 hidden_layers=2,
                 output_dim=1,
                 dropout=0.0,
                 activation=nn.LeakyReLU,#nn.ReLU,
                 #activation=nn.Tanh,#nn.ReLU,
                 logdir=None,
                 random_seed=1,
                 device="cpu"):
        super(FullyConnectedNN, self).__init__()
        torch.manual_seed(random_seed)

        self.device = torch.device(device)
        self.is_trainable = True

        layers = []
        for i in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_units  # update input for next layer

        layers.append(nn.Linear(hidden_units, output_dim))
        self.network = nn.Sequential(*layers)
        for module in self.network:
          if isinstance(module, nn.Linear):
              nn.init.zeros_(module.bias)
    def forward(self, x):
        for layer in self.network:
            x = layer(x)
            #print(f"{layer}: {x}")
        return x.squeeze()
# ==============================
# Trainer Base
# ==============================
class AlphaTrainerBase(ABC):
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    @abstractmethod
    def train(self, dataset, epochs, batch_size): ...


# ==============================
# Sharpe Ratio Trainer
# ==============================
class SharpeTrainer(AlphaTrainerBase):
    def __init__(self, model, optimizer_cls=torch.optim.Adam, optimizer_kwargs=None, device="cpu", tc_coef=0.0,train_cost=0.0):
        super().__init__(model, device)
        self.tc_coef = tc_coef
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 1e-3}
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        self.cost=tc_coef
        self.train_cost = train_cost
        self.prev_wts = None
        self.loss_train_hist =[]
        self.loss_test_hist =[]

    def train(self, dataset, epochs=100, batch_size=64):
        loss_train_hist,loss_test_hist = [0]*epochs,[0]*epochs
        rets_full = []
        volume_full = []
        short_proportion =[]
        turnover =[]
        history = []
        wts_full_df = []
        loss_full=[]
        for epoch in range(epochs):
            self.model.train()
            self.prev_wts = None
            batch_losses = []
            for i in range(dataset.num_train_batches(batch_size)):
              x, y ,t_obs,N= dataset.get_train_batch(batch_size, i)
              x= torch.tensor(x).to(self.device)
              wts = self.model(x)
              wts =wts.reshape(t_obs,N)

              wts = wts / (torch.sum(torch.abs(wts), dim=1, keepdim=True) + 1e-6)
              wts = wts - wts.mean(dim=1, keepdim=True)
              volume_diff = wts[1:, :] - wts[:-1, :]
              volume_abs_diff = torch.abs(volume_diff)
              volume_first_step = torch.abs(wts[0, :]).unsqueeze(0)
              volume = torch.cat((volume_first_step, volume_abs_diff), dim=0)
              short_proportion_batch=(wts<0).sum(axis=1).detach().cpu().numpy()
              rets_por = torch.sum(wts*torch.tensor(y,device=device),axis=1) -torch.sum(volume*self.train_cost,axis=1)
              #rets_por = torch.clamp(rets_por, min=-100 * torch.std(rets_por), max=3 * torch.std(rets_por))
              #self.prev_wts = wts.detach()
              #print(rets_por.shape)
              #rets_por = rets_por - cost
              pnl_mean=torch.mean(rets_por)
              pnl_std=torch.std(rets_por)
              loss = -pnl_mean/pnl_std
              loss_full.append(loss.detach().cpu().item())
              self.optimizer.zero_grad()
              loss.backward()
              self.optimizer.step()
              rets_full.append(rets_por.detach().cpu().numpy())
              volume_full.append(torch.sum(volume).detach().cpu().numpy())
              short_proportion.append(short_proportion_batch)

            mean_full = np.mean(np.concat(rets_full,axis=0))
            sum_full = np.sum(np.concat(rets_full,axis=0))
            vol_full = np.sum(volume_full)
            std_full =np.std(np.concat(rets_full,axis=0))
            sharpe_full = mean_full/std_full*252**0.5
            loss_train_hist[epoch]=sharpe_full
            out = self.evaluate(model, dataset)
            ret_out = (out['pnl']-out['volume']*trainer.cost)
            sharpe_out = ret_out.mean()/ret_out.std()*252**0.5
            loss_test_hist[epoch]=sharpe_out
            #print(sum_full)
            #print(vol_full)

            if epoch%100==0:
              #print(loss_full)
              print(f'epoch {epoch} | Sharpe {sharpe_full:.4f}| SharpeOOS {sharpe_out:.4f} |loss {np.mean(loss_full):.4f} |pnl {sum_full/vol_full*1e4:.4f} bps')
        self.loss_train_hist =loss_train_hist
        self.loss_test_hist =loss_test_hist
        return loss_train_hist

    def evaluate(self,model, dataset, device="cpu",type_='test',batch_s=3000000):
        model.eval()
        with torch.no_grad():
          if type_=='train':
            x, y ,t_obs,N= dataset.get_train_batch(batch_s, 0)
            idx=dataset.train_idx
          else:
            x, y ,t_obs,N= dataset.get_test_batch(batch_s, 0)
            idx=dataset.test_idx
          x= torch.tensor(x).to(self.device)
          wts = model(x)
          wts =wts.reshape(t_obs,N)

          wts = wts / (torch.sum(torch.abs(wts), dim=1, keepdim=True) + 1e-6)
          wts = wts - wts.mean(dim=1, keepdim=True)
          volume_diff = wts[1:, :] - wts[:-1, :]
          volume_abs_diff = torch.abs(volume_diff)
          volume_first_step = torch.abs(wts[0, :]).unsqueeze(0)
          volume = torch.cat((volume_first_step, volume_abs_diff), dim=0)
          short_proportion_batch=(wts<0).sum(axis=1).detach().cpu().numpy()
          rets_por = torch.sum(wts*torch.tensor(y,device=device),axis=1)
          rets_out = rets_por.detach().cpu().numpy()
          volume_out = torch.sum(volume,axis=1).detach().cpu().numpy()
          pnl =pd.DataFrame(rets_out,index = idx)
          vol=pd.DataFrame(volume_out,index = idx)
          out = pd.concat([pnl,vol],axis=1)
          out.columns = ['pnl','volume']
          return out
class MomTrainer(AlphaTrainerBase):
    def __init__(self, model, optimizer_cls=torch.optim.Adam, optimizer_kwargs=None, device="cpu", tc_coef=0.0,train_cost=0.0):
        super().__init__(model, device)
        self.tc_coef = tc_coef
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 1e-3}
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        self.cost=tc_coef
        self.train_cost = train_cost
        self.prev_wts = None
        self.loss_train_hist =[]
        self.loss_test_hist =[]

    def train(self, dataset, epochs=100, batch_size=64):
        loss_train_hist,loss_test_hist = [0]*epochs,[0]*epochs
        rets_full = []
        volume_full = []
        short_proportion =[]
        turnover =[]
        history = []
        wts_full_df = []
        loss_full=[]
        for epoch in range(epochs):
            self.model.train()
            self.prev_wts = None
            batch_losses = []
            for i in range(dataset.num_train_batches(batch_size)):
              x, y ,t_obs,N= dataset.get_train_batch(batch_size, i)
              x= torch.tensor(x).to(self.device)
              wts = self.model(x)
              wts =wts.reshape(t_obs,N)

              wts = wts / (torch.sum(torch.abs(wts), dim=1, keepdim=True) + 1e-6)
              #wts = wts - wts.mean(dim=1, keepdim=True)
              volume_diff = wts[1:, :] - wts[:-1, :]
              volume_abs_diff = torch.abs(volume_diff)
              volume_first_step = torch.abs(wts[0, :]).unsqueeze(0)
              volume = torch.cat((volume_first_step, volume_abs_diff), dim=0)
              short_proportion_batch=(wts<0).sum(axis=1).detach().cpu().numpy()
              rets_por = torch.sum(wts*torch.tensor(y,device=device),axis=1) -torch.sum(volume*self.train_cost,axis=1)
              #rets_por = torch.clamp(rets_por, min=-100 * torch.std(rets_por), max=3 * torch.std(rets_por))
              #self.prev_wts = wts.detach()
              #print(rets_por.shape)
              #rets_por = rets_por - cost
              pnl_mean=torch.mean(rets_por)
              pnl_std=torch.std(rets_por)
              loss = -pnl_mean/pnl_std
              loss_full.append(loss.detach().cpu().item())
              self.optimizer.zero_grad()
              loss.backward()
              self.optimizer.step()
              rets_full.append(rets_por.detach().cpu().numpy())
              volume_full.append(torch.sum(volume).detach().cpu().numpy())
              short_proportion.append(short_proportion_batch)

            mean_full = np.mean(np.concat(rets_full,axis=0))
            sum_full = np.sum(np.concat(rets_full,axis=0))
            vol_full = np.sum(volume_full)
            std_full =np.std(np.concat(rets_full,axis=0))
            sharpe_full = mean_full/std_full*252**0.5
            loss_train_hist[epoch]=sharpe_full
            out = self.evaluate(model, dataset)
            ret_out = (out['pnl']-out['volume']*trainer.cost)
            sharpe_out = ret_out.mean()/ret_out.std()*252**0.5
            loss_test_hist[epoch]=sharpe_out
            #print(sum_full)
            #print(vol_full)

            if epoch%100==0:
              #print(loss_full)
              print(f'epoch {epoch} | Sharpe {sharpe_full:.4f}| SharpeOOS {sharpe_out:.4f} |loss {np.mean(loss_full):.4f} |pnl {sum_full/vol_full*1e4:.4f} bps')
        self.loss_train_hist =loss_train_hist
        self.loss_test_hist =loss_test_hist
        return loss_train_hist

    def evaluate(self,model, dataset, device="cpu",type_='test',batch_s=3000000):
        model.eval()
        with torch.no_grad():
          if type_=='train':
            x, y ,t_obs,N= dataset.get_train_batch(batch_s, 0)
            idx=dataset.train_idx
          else:
            x, y ,t_obs,N= dataset.get_test_batch(batch_s, 0)
            idx=dataset.test_idx
          x= torch.tensor(x).to(self.device)
          wts = model(x)
          wts =wts.reshape(t_obs,N)

          wts = wts / (torch.sum(torch.abs(wts), dim=1, keepdim=True) + 1e-6)
          wts = wts - wts.mean(dim=1, keepdim=True)
          volume_diff = wts[1:, :] - wts[:-1, :]
          volume_abs_diff = torch.abs(volume_diff)
          volume_first_step = torch.abs(wts[0, :]).unsqueeze(0)
          volume = torch.cat((volume_first_step, volume_abs_diff), dim=0)
          short_proportion_batch=(wts<0).sum(axis=1).detach().cpu().numpy()
          rets_por = torch.sum(wts*torch.tensor(y,device=device),axis=1)
          rets_out = rets_por.detach().cpu().numpy()
          volume_out = torch.sum(volume,axis=1).detach().cpu().numpy()
          pnl =pd.DataFrame(rets_out,index = idx)
          vol=pd.DataFrame(volume_out,index = idx)
          out = pd.concat([pnl,vol],axis=1)
          out.columns = ['pnl','volume']
          return out

def preprocess_cumsum(data, lookback):
    """
    Preprocess TxN numpy ndarray `data` into cumulative sum windows of integer length `lookback`.

    For use with residual returns time series, to create residual cumulative returns time series.
    """
    signal_length = lookback
    T,N = data.shape
    cumsums = np.cumsum(data, axis=0)
    windows = np.zeros((T-lookback, N, signal_length), dtype=np.float32)
    idxs_selected = np.zeros((T-lookback,N), dtype=bool)
    for t in range(lookback,T):
        # chooses stocks which have no missing returns in the t-th lookback window
        idxs_selected[t-lookback,:] = ~np.any(data[(t-lookback):t,:] == 0, axis = 0).ravel()
        idxs = idxs_selected[t-lookback,:]

        if t == lookback:
            windows[t-lookback,idxs,:] = cumsums[t-lookback:t,idxs].T
        else:
            # Probably unnecessary given the conv normalization, but is just to have the same setting as in the OU case
            windows[t-lookback,idxs,:] = cumsums[t-lookback:t,idxs].T - cumsums[t-lookback-1,idxs].reshape(int(sum(idxs)),1)
    idxs_selected = torch.as_tensor(idxs_selected)
    return windows, idxs_selected
#############run
device = "cuda" if torch.cuda.is_available() else "cpu"
device_ids = [0] # Adjust if you have multiple GPUs and want to use them
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")



DEVICE =device
BATCH_SIZE =5000
EPOCH_NUM = 1000
optimizer_opts = {"lr":0.001}

model = FullyConnectedNN(logdir=None,
                         hidden_units=1,
                         hidden_layers=1,
                         dropout=0.4,
                         input_dim=dataset.train_inputs.shape[-1],
                random_seed = 0)
trainer = SharpeTrainer(model, device="cpu",tc_coef=0e-4,train_cost=1e-4)
display(summary(model, input_size=(1, dataset.lookback), device=DEVICE))
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")
loss_train_hist = trainer.train(dataset, epochs=EPOCH_NUM, batch_size=BATCH_SIZE)



plt.plot(pd.DataFrame(trainer.loss_train_hist))
plt.plot(pd.DataFrame(trainer.loss_test_hist))
plt.show()

out = trainer.evaluate(model,dataset)
(out['pnl']-out['volume']*trainer.cost).cumsum().plot()
pnl_trade=(out['pnl'].sum()-out['volume'].sum()*0e-4)/out['volume'].sum()
print(pnl_trade*1e4)
plt.show()
out = trainer.evaluate(model,dataset,type_='train')
plt.plot((out['pnl']-out['volume']*trainer.cost).cumsum().values)
plt.show()
pnl_trade=(out['pnl'].sum()-out['volume'].sum()*0e-4)/out['volume'].sum()
print(pnl_trade*1e4)
