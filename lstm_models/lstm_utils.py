import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.ndimage import gaussian_filter


class EarthquakeDataset(Dataset):
    def __init__(
        self, 
        path, 
        target_path,
        baseline_history_days=5*365,
        sigma=0.0, 
        seq_len=30, 
        stride=30,
        history_horizon=1,
        pred_horizon_min=10, 
        pred_horizon_max=50,
        dataset_type="train",
        n=160,
        m=180,
        x_min=None,
        x_max=None,
    ):
        """
        Args:
            path: Path to input data tensor of shape [total_timesteps, H, W]
            target_path: Path to target data tensor of shape [total_timesteps, H, W]
            baseline_history_days: Days to aggregate for baseline map
            seq_len: Input sequence length
            stride: Sliding window step
            pred_horizon_min/max: Range of days to predict (binary target)
            dataset_type: "train" or "test"
            x_mean/x_std: Precomputed mean/std (for test set)
        """
        assert dataset_type in ["train", "test"]
    
        self.data = torch.load(path).to(torch.float)  # [T, H, W]
        self.target = torch.load(target_path).to(torch.float)  # [T, H, W]
        if dataset_type == "train":
            self.data = self.data[:-1000]
            self.target = self.target[:-1000]
        elif dataset_type == "test":
            self.data = self.data[-(baseline_history_days+1000):]
            self.target = self.target[-(baseline_history_days+1000):]
            
        self.seq_len = seq_len
        self.stride = stride
        self.n = n
        self.m = m
        self.history_horizon = history_horizon
        self.baseline_history_days = baseline_history_days
        self.pred_horizon_min = pred_horizon_min
        self.pred_horizon_max = pred_horizon_max
        self.dataset_type = dataset_type
        self.sigma = sigma

        # Compute valid clips (avoid out-of-bounds)
        self.total_clips = (
            (len(self.data) - baseline_history_days - seq_len - pred_horizon_max - history_horizon) 
            // stride + 1
        )
        assert self.total_clips > 0, "Dataset is too short for given seq_len/pred_horizon!"

        # Normalization
        if dataset_type == "train":
            self.x_min, self.x_max = self._compute_min_max()
        else:
            assert x_min is not None and x_max is not None, "Test set requires precomputed min/max!"
            self.x_min, self.x_max = x_min, x_max

    def _compute_min_max(self):
        all_sequences = []

        for idx in range(self.total_clips):
            start = self.baseline_history_days + idx * self.stride
            window_starts = torch.arange(
                start,
                start + self.seq_len * self.stride,
                self.stride,
                device=self.data.device
            )
            window_ends = window_starts + self.history_horizon
            x = torch.stack([self.data[s:e].sum(0) for s, e in zip(window_starts, window_ends)])
            all_sequences.append(x)

        all_sequences = torch.stack(all_sequences)
        x_min = torch.amin(all_sequences, dim=(0, 1))
        x_max = torch.amax(all_sequences, dim=(0, 1))
        
        return x_min, x_max

    def min_max_scale(self, x):
        if self.x_min is not None and self.x_max is not None:
            mask = x > 0
            x_min_expanded = self.x_min.unsqueeze(0).expand_as(x)
            x_max_expanded = self.x_max.unsqueeze(0).expand_as(x)
            x[mask] = (x[mask] - x_min_expanded[mask]) / (x_max_expanded[mask] - x_min_expanded[mask] + 1e-8)
            x[mask] = x[mask].clamp(0, 1)
        
        return x

    def __len__(self):
        return self.total_clips

    def __getitem__(self, idx):
        start = self.baseline_history_days + idx*self.stride + self.history_horizon # do not take first baseline_history_days
        
        # 1. Input sequence
        window_starts = torch.arange(
            start,
            start + self.seq_len * self.stride,
            self.stride,
            device=self.data.device
        )
        window_ends = window_starts + self.history_horizon
        x = torch.stack([self.data[s:e].sum(0) for s, e in zip(window_starts, window_ends)])
        if self.x_min is not None:
            x = self.min_max_scale(x) 

        # 2. Baseline calculation
        baseline_map = self.target[start-self.baseline_history_days:start].sum(0)
        baseline_map = gaussian_filter(baseline_map, sigma=self.sigma)
        if baseline_map.max() > 0:
            baseline_map /= baseline_map.max() + 1e-8 
        baseline_map = torch.tensor(baseline_map+ 1e-8, dtype=torch.float32)

        # 3. Concatenate
        x = torch.cat([x, baseline_map.unsqueeze(0)]) # [seq_len+1, H, W]

        # 4. Binary target (did any event occur in the horizon?)
        last_x_idx = window_ends[-1].numpy()
        y = (
            self.target[
                last_x_idx + self.pred_horizon_min:
                last_x_idx + self.pred_horizon_max
            ].sum(dim=0) > 0
        )  # [H, W], binary

        return x.to(torch.float16), y.to(torch.float16)
    
    
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

    
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()
        
        self._check_kernel_size_consistency(kernel_size)
        
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, _, _, h, w = input_tensor.size()
        
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            hidden_state = hidden_state
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    
class EarthquakePredictionModel(nn.Module):
    def __init__(self, seq_len, input_dim=1, hidden_dim=[64], kernel_size=(3, 3), num_layers=1):
        super(EarthquakePredictionModel, self).__init__()
        
        self.hidden_state = None
        self.convlstm = ConvLSTM(input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 num_layers=num_layers,
                                 batch_first=True)
        
        self.layer_norm_lstm = nn.LayerNorm([hidden_dim[-1], 160, 180])
        self.conv1x1 = nn.Conv2d(in_channels=hidden_dim[-1] + 1, out_channels=1, kernel_size=(1, 1))
        self.layer_norm_conv = nn.LayerNorm([1, 160, 180])
   
        self.w1 = nn.Parameter(torch.ones(160, 180))  
        self.w2 = nn.Parameter(torch.ones(160, 180))  
        
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        seq_data = x[:, :-1, :, :]  # [batch, seq_len, H, W]
        baseline_map = x[:, -1, :, :].unsqueeze(1)  # [batch, 1, H, W]
        
        seq_data = seq_data.unsqueeze(2)  # [batch, seq_len, 1, H, W]

        # Forward pass through ConvLSTM
        lstm_out, _ = self.convlstm(seq_data)
        lstm_out_last = lstm_out[-1][:, -1, :, :, :]  # Get the last output of the last layer
        lstm_out_last = self.layer_norm_lstm(lstm_out_last)
        
        # Concatenate LSTM output with baseline map
        concat = torch.cat([lstm_out_last, baseline_map], dim=1)
        
        # Apply 1x1 convolution
        correction = self.conv1x1(concat)
        correction = self.layer_norm_conv(correction)
        correction = self.activation(correction)

        # linear combination of baseline and correction
        prediction = self.w1 * baseline_map.squeeze(1) + self.w2 * correction.squeeze(1)
    
        return prediction
    