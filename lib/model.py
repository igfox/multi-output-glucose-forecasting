import numpy as np
import joblib

import torch
from torch import nn
from torch.autograd import Variable


class ForecastRNN(nn.Module):
    """
    Helper for pytorch reimplementation
    Uses variable sized/depth GRU with linear layer to get output right
    """
    def __init__(self, input_dim, output_dim, hidden_size, depth, output_len=-1, cuda=False):
        super(ForecastRNN, self).__init__()
        self.cuda = cuda
        self.rnn = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_size,
                          num_layers=depth,
                          dropout=False,
                          bidirectional=False,  # would bidirectional help forecasting?
                          batch_first=True)
        self.sm = nn.LogSoftmax(dim=1)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_len = output_len
        if self.cuda:
            self.rnn = self.rnn.cuda()
            self.sm = self.sm.cuda()
            self.float = torch.cuda.FloatTensor  # not sure I need this
        else:
            self.float = torch.FloatTensor

    @staticmethod
    def _dist_to_bins(dist):
        return torch.max(dist, dim=-1)[1]

    @staticmethod
    def _get_sequence_info(seq):
        """
        gets info on fed sequence
        """
        if type(seq) == torch.nn.utils.rnn.PackedSequence:
            pack = True
            batch_size = seq.batch_sizes[0]
            sequence_length = len(seq.batch_sizes)
        else:
            pack = False
            batch_size = seq.size(0)
            sequence_length = seq.size(1)
        return pack, batch_size, sequence_length
    
    def _rnn_forward(self, seq, pack, batch_size):
        """
        Helper function for forward that computes up to output layer
        """
        h = Variable(torch.zeros(self.rnn.num_layers, 
                                 batch_size, # not sure if need to reshape for batch_first
                                 self.rnn.hidden_size).type(self.float), 
                         requires_grad=False)
        # predict within the sequence
        out, h = self.rnn.forward(seq, h)
        if pack:
            out, lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, padding_value=-1)
        else:
            lens = None
        # out has dim (batch_size, sequence_length, hidden_size)
        out_flat = out.contiguous().view(-1, self.rnn.hidden_size)
        return out_flat, h, lens
    
    def _extract_final_dist(self, pack, batch_size, y, lens):
        """
        Given y (possibly with padding), get distribution
        for final prediction at t+1
        prediction must be of size (batch_size, 1[, output_len], output_length)
        """
        if type(self) is RecursiveRNN:
            output_len = 1
        else:
            output_len = self.decoding_steps
        single_view = 1, 1, output_len, self.output_dim
        batch_view = batch_size, 1, output_len, self.output_dim
        if pack:
            # need to handle uneven lengths
            final_dist = []
            for i in range(batch_size):
                final_dist.append(y[i, lens[i]-1].view(single_view))
            final_dist = torch.cat(final_dist).view(batch_view)
        else:
            final_dist = y[:, -1].contiguous().view(batch_view)
        return final_dist
    
    def forward(self, seq, glucose_dat, pred_len=0):
        raise NotImplementedError
        

class RecursiveRNN(ForecastRNN):
    """
    Designed to handle uneven batch sizes
    """
    def __init__(self, input_dim, output_dim, hidden_size, depth, cuda):
        super(RecursiveRNN, self).__init__(input_dim=input_dim, 
                                           output_dim=output_dim, 
                                           hidden_size=hidden_size,
                                           depth=depth, 
                                           cuda=cuda)
        self.output = nn.Linear(hidden_size, output_dim)
        if self.cuda:
            self.output = self.output.cuda()
    
    def _hidden_state_to_output(self, out_flat, batch_size, sequence_length):
        """
        Given output from RNN layer, translate to output
        """
        return self.sm(self.output(out_flat)).contiguous().view(batch_size, sequence_length, 1, self.output_dim)
    
    def forward(self, seq, glucose_dat, pred_len=0, **kwargs):
        """
        pred_len is number of recursive forecasts to make
        Note: there is padding in form of -1, need to remove for
        accurate loss
        bins reverse probability predictions to real values
        
        returns:
        curr_dist: (batch_size, sequence_length-1, 1[output_len], output_dim)
        curr_pred: (batch_size, sequence_length-1, 1[pred_dim])
        future_dist: (batch_size, 1[tiled preds], pred_len+1, output_dim)
        future_pred: (batch_size, 1[tiled preds], pred_len+1)
        """
        pack, batch_size, sequence_length = self._get_sequence_info(seq)
        out_flat, h, lens = self._rnn_forward(seq, pack, batch_size)
        
        y = self._hidden_state_to_output(out_flat, batch_size, sequence_length)

        final_dist = self._extract_final_dist(pack, batch_size, y, lens)
         
        if y.data.shape[1] == 1:
            # only 1 input, no within series predictions
            curr_dist = None
        else:
            curr_dist = y[:, :-1]
        curr_pred = self._dist_to_bins(curr_dist)
        
        future_dist = [final_dist]

        future_pred = [self._dist_to_bins(future_dist[-1])]
        
        for i in range(pred_len):
            if self.cuda:
                pred_vals = glucose_dat.bins_to_values(future_pred[-1].data.cpu().numpy())
            else:
                pred_vals = glucose_dat.bins_to_values(future_pred[-1].data.numpy())
            out, h = self.rnn.forward(Variable(torch.from_numpy(pred_vals).type(self.float)), h)
            out_flat = out.contiguous().view(-1, self.rnn.hidden_size)
            y_f = self._hidden_state_to_output(out_flat, batch_size, 1)
            future_dist.append(y_f)
            future_pred.append(self._dist_to_bins(future_dist[-1]))
        return curr_dist, curr_pred, torch.cat(future_dist, dim=2), torch.cat(future_pred, dim=2)


class MultiOutputRNN(ForecastRNN):
    """
    Designed to handle uneven batch sizes
    """
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 output_len, 
                 hidden_size, 
                 depth, 
                 cuda, 
                 autoregressive=False,
                 sequence=False,
                 polynomial=False, 
                 degree=2):
        super(MultiOutputRNN, self).__init__(input_dim=input_dim, 
                                             output_dim=output_dim, 
                                             hidden_size=hidden_size, 
                                             depth=depth, 
                                             output_len=output_len,
                                             cuda=cuda)
        self.ar = autoregressive
        self.seq = sequence
        self.polynomial = polynomial
        self.degree = degree
        if self.polynomial:
            self.decoding_steps = self.degree+1
            self.polyval_layer = nn.Linear(self.decoding_steps*output_dim, output_len*output_dim)
        else:
            self.decoding_steps = self.output_len
        if self.seq:
            self.decoder = nn.GRU(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  num_layers=1,
                                  dropout=False,
                                  bidirectional=False,
                                  batch_first=False)
            self.decoder.cuda()
            self.output = nn.Linear(hidden_size, output_dim)
        elif self.ar:
            output = [nn.Linear(hidden_size, output_dim)]
            for i in range(self.decoding_steps-1):
                output.append(nn.Linear(hidden_size + output_dim, output_dim))
            self.output = nn.ModuleList(output)
        else:
            output = [nn.Linear(hidden_size, output_dim) for i in range(self.decoding_steps)]
            self.output = nn.ModuleList(output)
        if self.cuda:
            self.output = self.output.cuda()

    def _hidden_state_to_output(self, out_flat, batch_size, sequence_length):
        """
        Given output from RNN layer, translate to output
        y has size (batch_size, sequence_length, output_len, output_dim)
        might want to change
        """
        if self.seq:
            y = []
            encoded = out_flat[None, :]
            hidden = Variable(torch.zeros(encoded.data.shape)).cuda()
            for i in range(self.decoding_steps):
                encoded, hidden = self.decoder(encoded, hidden)
                pred = self.sm(self.output(encoded[0])).contiguous()
                y.append(pred.view(batch_size,
                                   sequence_length,
                                   1,
                                   self.output_dim))
            return torch.cat(y, dim=2)
        else:
            y = []
            for i in range(len(self.output)):
                if self.ar:
                    if i == 0:
                        pred = self.sm(self.output[0](out_flat)).contiguous()
                        y.append(pred.view(batch_size,
                                           sequence_length,
                                           1,
                                           self.output_dim))
                    else:
                        fused_state = torch.cat((out_flat, pred), dim=1)
                        pred = self.sm(self.output[i](fused_state)).contiguous()
                        y.append(pred.view(batch_size,
                                           sequence_length,
                                           1,
                                           self.output_dim))
                else:
                    y.append(self.sm(self.output[i](out_flat)).contiguous().view(batch_size,
                                                                                 sequence_length,
                                                                                 1,
                                                                                 self.output_dim))
            return torch.cat(y, dim=2)

    def poly_to_val(self, poly):
        return poly

    def forward(self, seq, glucose_dat, **kwargs):
        """
        prediction into future is based on output size
        Note: there is padding in form of -1, need to remove for
        accurate loss
        bins reverse probability predictions to real values
        """
        pack, batch_size, sequence_length = self._get_sequence_info(seq)
        out_flat, h, lens = self._rnn_forward(seq, pack, batch_size)
        
        y = self._hidden_state_to_output(out_flat, batch_size, sequence_length)

        final_dist = self._extract_final_dist(pack, batch_size, y, lens)

        if y.data.shape[1] <= self.output_len:
            # curr_dist contains dists ENTIRELY within signal
            # note that this reduces training size
            curr_dist = None
        else:
            curr_dist = y[:, :-self.output_len]
        curr_pred = self._dist_to_bins(curr_dist)
        
        future_dist = [final_dist]
        future_pred = self._dist_to_bins(future_dist[-1])
        if self.polynomial:
            curr_real_pred = self.poly_to_val(curr_pred)
            future_real_pred = self.poly_to_val(future_pred)
        return (curr_dist, 
                curr_pred, 
                torch.cat(future_dist, dim=0), 
                future_pred)

def sort_batch(batch_x, batch_y, batch_y_real, lens):
    """
    Sorts minibatch by length in decreasing order
    to accomodate pack_padded_sequence 
    """
    dat_x, dat_y, dat_y_real, dat_l = batch_x.numpy(), batch_y.numpy(), batch_y_real.numpy(), lens.numpy()
    sort_x = dat_x[(dat_l*-1).argsort()]  # -1 to get descending order
    sort_y = dat_y[(dat_l*-1).argsort()]
    sort_y_real = dat_y_real[(dat_l*-1).argsort()]
    sort_l = dat_l[(dat_l*-1).argsort()]
    return sort_x, sort_y, sort_y_real, sort_l


def convert_batch(batch_x, batch_y, batch_y_real, batch_l, cuda, real_values=False):
    """
    Given batches in numpy form, 
    convert to proper type for model input
    """
    if cuda:
        float_type = torch.cuda.FloatTensor
        long_type = torch.cuda.LongTensor
    else:
        float_type = torch.FloatTensor
        long_type = torch.LongTensor
    new_batch_x = Variable(torch.from_numpy(batch_x).type(float_type), requires_grad=False)
    if real_values:
        new_batch_y = Variable(torch.from_numpy(batch_y).type(float_type), requires_grad=False)
        new_batch_y_real = new_batch_y
    else:
        new_batch_y = Variable(torch.from_numpy(batch_y).type(long_type), requires_grad=False)
        new_batch_y_real = Variable(torch.from_numpy(batch_y_real).type(long_type), requires_grad=False)
    new_batch_l = list(batch_l)
    return new_batch_x, new_batch_y, new_batch_y_real, new_batch_l


def remove_prediction_padding(prediction_distribution,
                              target_value,
                              loss_weight,
                              target_real_value):
    """
    Masks prediction for artificial targets and flattens
    """
    # assuming target value will have all -1 or no -1
    missing_indicator = torch.min(target_value, dim=2)[0] != -1

    prediction_nopad = torch.masked_select(
        prediction_distribution,
        missing_indicator[:, :, None, None]).view(-1, prediction_distribution.shape[-1])
    target_nopad = torch.masked_select(
        target_value,
        missing_indicator[:, :, None])
    target_real_nopad = torch.masked_select(
        target_real_value,
        missing_indicator[:, :, None])
    loss_weight_nopad = torch.masked_select(
        loss_weight,
        missing_indicator[:, :, None])
    return prediction_nopad, target_nopad, target_real_nopad, loss_weight_nopad


def remove_prediction_padding_old(prediction_distribution,
                              target_value,
                              loss_weight,
                              target_real_value):
    """
    Masks prediction for artificial targets
    """
    prediction_distribution = prediction_distribution.contiguous().view(-1, 361)
    target_value = target_value.contiguous().view(-1)
    loss_weight = loss_weight.contiguous().view(-1)
    inter = (target_value != -1).view(-1, 1)
    mask = inter.expand(prediction_distribution.size(0), prediction_distribution.size(1))
    ret = [prediction_distribution[mask].view(-1, prediction_distribution.size(1)),
           target_value[(target_value != -1)],
           None]
    if loss_weight is not None:
        ret.append(loss_weight[(target_value != -1)])
    else:
        ret.append(None)
    return ret


def get_loss(inp,
             out,
             out_real,
             lens,
             cuda,
             gn,
             glucose_dat,
             criterion,
             base=1,
             value_weight=0,
             value_ratio=0):
    """
    Simple helper function that calculates model loss.
    Basically to save some space
    """
    batch_size_val = inp.size(0)
    output_dim = gn.output_dim

    weight_vec = torch.Tensor([base ** i for i in reversed(range(out.size(-1)))])
    weight_vec = (weight_vec/weight_vec.sum()) * weight_vec.numel()  # consistent weighting on output length
    loss_weight = weight_vec.expand(out.shape)

    inp_s, out_s, out_real_s, lens_s = sort_batch(inp, out, out_real, lens)
    inp_s, out_s, out_real_s, lens_s = convert_batch(batch_x=inp_s,
                                                     batch_y=out_s,
                                                     batch_y_real=out_real_s,
                                                     batch_l=lens_s,
                                                     cuda=cuda,
                                                     real_values=glucose_dat.real_values)
    x = nn.utils.rnn.pack_padded_sequence(inp_s.view(batch_size_val, 
                                                     glucose_dat.max_pad,
                                                     1), 
                                          list(np.array(lens_s)), 
                                          batch_first=True)
    if glucose_dat.real_values:
        yd_p, y_p, yd_f, y_f = gn(x, pred_len=0)
        y_p_flat = y_p.contiguous().view(-1, output_dim)
        (y_p_nopad,
         y_nopad,
         y_real_nopad,
         loss_weight_nopad) = remove_prediction_padding(prediction_distribution=y_p_flat,
                                                        target_value=out_s.view(-1),
                                                        loss_weight=Variable(loss_weight.cuda()),
                                                        target_real_value=out_real_s)
        try:
            loss = criterion(y_p_nopad, y_nopad)
        except:
            print(type(y_nopad.data))
            print(type(out_s.data))
            print(type(out))
            raise
            
    else:
        yd_p, y_p, yd_f, y_f = gn(x, glucose_dat, pred_len=out.shape[-1])
        (yd_p_nopad,
         y_nopad,
         y_real_nopad,
         loss_weight_nopad) = remove_prediction_padding(prediction_distribution=yd_p,
                                                        target_value=out_s,
                                                        loss_weight=Variable(loss_weight.cuda()),
                                                        target_real_value=out_real_s)
        if glucose_dat.polynomial:
            # include MSE
            real_criterion = torch.nn.MSELoss()
            coeffs = get_coeffs(yd_p_nopad.view(-1, len(glucose_dat.bins), yd_p_nopad.shape[-1]), glucose_dat.bins)
            real_values = coeffs_to_values(coeffs)
            loss_real = real_criterion(real_values.view(-1), y_real_nopad.float()) * value_weight
            loss_dist = criterion(yd_p_nopad, y_nopad) * loss_weight_nopad
            loss = (1-value_ratio) * loss_dist + value_ratio * loss_real
            if np.isnan(loss.data[0]):
                raise ValueError('Got NaN loss')
        else:
            loss = criterion(yd_p_nopad, y_nopad) * loss_weight_nopad
            if np.isnan(loss.data[0]):
                raise ValueError('Got NaN loss')
            if torch.min(y_nopad.data) == -1:
                print('trouble ahead')
    return loss.mean(), y_p


def get_coeffs(dist, bins):
    prob = torch.exp(dist)
    bin_vals = Variable(torch.from_numpy(np.array(bins)).float().cuda()).expand_as(prob).transpose(1, 2)
    coeffs = torch.bmm(prob, bin_vals)  # includes false off-diag coeffs
    real_coeffs = coeffs[torch.eye(len(bins)).expand_as(coeffs).byte().cuda()].view(-1, len(bins))  # extract diagonals
    return real_coeffs


def coeffs_to_values(coeffs):
    degree = coeffs.shape[-1]
    basis = Variable(torch.stack([torch.arange(0, 6) ** i for i in range(degree)]).cuda())
    return coeffs.view(-1, degree) @ basis


def get_predictions(inp,
                    out,
                    lens,
                    cuda,
                    gn,
                    glucose_dat):
    """
    Gets predictions
    """
    batch_size_val = inp.size(0)
    output_dim = gn.output_dim

    inp_s, out_s, lens_s = sort_batch(inp, out, lens)
    inp_s, out_s, lens_s = convert_batch(inp_s,
                                         out_s,
                                         lens_s,
                                         cuda,
                                         glucose_dat.real_values)
    x = nn.utils.rnn.pack_padded_sequence(inp_s.view(batch_size_val,
                                                     glucose_dat.max_pad,
                                                     1),
                                          list(np.array(lens_s)),
                                          batch_first=True)
    if glucose_dat.real_values:
        yd_p, y_p, yd_f, y_f = gn(x, pred_len=0)
        y_p_flat = y_p.contiguous().view(-1, output_dim)
        y_p_nopad, y_nopad = remove_prediction_padding(y_p_flat,
                                                       out_s.view(-1))

    else:
        yd_p, y_p, yd_f, y_f = gn(x, glucose_dat, pred_len=out.shape[-1])
        yd_p_flat = yd_p.contiguous().view(-1, output_dim)
        yd_p_nopad, y_nopad = remove_prediction_padding(yd_p_flat,
                                                        out_s.view(-1))
    return yd_p, y_p, yd_f, y_f


def make_model(config):
    """
    A poor man's factory method.
    """
    if config.model_type == 'recursive':
        gn = RecursiveRNN(input_dim=config.input_dim,
                          output_dim=config.output_dim,
                          hidden_size=config.hidden_size,
                          depth=config.depth,
                          cuda=True)
    else:
        assert config.output_len == config.pred_len # could relax
        gn = MultiOutputRNN(input_dim=config.input_dim,
                            output_dim=config.output_dim,
                            hidden_size=config.hidden_size,
                            output_len=config.output_len,
                            depth=config.depth,
                            cuda=True,
                            autoregressive=config.autoregressive,
                            sequence=config.sequence,
                            polynomial=config.polynomial,
                            degree=config.degree)
    return gn
