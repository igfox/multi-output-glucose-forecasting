"""
Implements training schemes with logging
"""
import numpy as np
import os
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
import joblib

import model as forecast_model


class ExperimentTrainer:
    """
    Simple training scheme
    """

    def __init__(self, model, optimizer, criterion, name, model_dir, log_dir,
                 load=False, load_epoch=None):
        """
        :param model: initialized model for training
        :param optimizer: initialized training optimizer
        :param name: string to save trainer results under
        :param load: whether or not to load results from previous train if they exist
        :param epoch: which epoch results to load, if None then the best found
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.name = name
        self.model_dir = model_dir
        self.log_dir = log_dir

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.log_dir)
        else:
            if load:
                if load_epoch is None:
                    self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'bsf_sup.pt')))
                else:
                    self.model.load_state_dict(torch.load(os.path.join(self.model_dir, '{}_sup.pt'.format(load_epoch))))

            else:
                print('Warning: directory already exists')
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train_sup(self, epoch_lim, data, valid_data, early_stopping_lim,
                  batch_size, num_workers, track_embeddings, validation_rate, loss_weight_base=1,
                  value_weight=0, value_ratio=0):
        """
        Training loop
        :param epoch_lim: total number of training epochs
        :param data: training data
        :param valid_data: validation data
        :param early_stopping_lim: Number of epochs to run without validation improvement before stopping
        if None, never stop early
        :param batch_size: training batch_size
        :param num_workers: number of CPU workers to use for data loading
        :param track_embeddings: Save out embedding information at end of run
        :param validation_rate: Check validation performance every validation_rate training epochs
        :param loss_weight_base: A constant between 0 and 1 used to interpolate between Single (=0) and Multi (=1) Step forecasting.
        :param value_weight: A constant multiplier for the real-value loss, set to 0 in the paper
        :param value_ratio: The proportion of loss used for the MSE loss term (as opposed for the cross-entropy loss), set to 0 in the paper
        :return loss array, model:
        """
        if early_stopping_lim is None:
            early_stopping_lim = epoch_lim
        train_sampler = sampler.RandomSampler(np.arange(len(data)))
        data_train = DataLoader(data,
                                batch_size=batch_size,
                                sampler=train_sampler,
                                drop_last=True)

        valid_sampler = sampler.SequentialSampler(np.arange(len(valid_data)))
        data_valid = DataLoader(valid_data,
                                batch_size=batch_size,
                                sampler=valid_sampler)
        step = 0

        bsf_loss = np.inf
        epochs_without_improvement = 0
        improvements = []
        for epoch in range(epoch_lim):
            if epochs_without_improvement > early_stopping_lim:
                print('Exceeded early stopping limit, stopping')
                break
            if epoch % validation_rate == 0:
                valid_loss = self.validation(data_valid=data_valid,
                                             step=step,
                                             data=data,
                                             loss_weight_base=loss_weight_base,
                                             value_weight=value_weight, value_ratio=value_ratio)
                (bsf_loss,
                 epochs_without_improvement,
                 improvements) = self.manage_early_stopping(bsf_loss=bsf_loss,
                                                            early_stopping_lim=early_stopping_lim,
                                                            epochs_without_improvement=epochs_without_improvement,
                                                            valid_loss=valid_loss, validation_rate=validation_rate,
                                                            improvements=improvements)
            running_train_loss = 0
            for inp, out, out_real, lens in tqdm(data_train):
                loss, y_p = forecast_model.get_loss(inp=inp,
                                                    out=out,
                                                    lens=lens,
                                                    cuda=True,
                                                    gn=self.model,
                                                    glucose_dat=data,
                                                    criterion=self.criterion,
                                                    base=loss_weight_base,
                                                    out_real=out_real,
                                                    value_weight=value_weight,
                                                    value_ratio=value_ratio)
                step += 1
                running_train_loss += loss.data.cpu().numpy()[0]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            running_train_loss = running_train_loss/len(data_train)
            self.writer.add_scalar(tag='train_loss',
                                   scalar_value=running_train_loss,
                                   global_step=step)
        torch.save(self.model.state_dict(), '{}/final_sup.pt'.format(self.model_dir))
        if track_embeddings:
            self.embed(data_valid, step, embed_batch=100)
        return improvements

    def manage_early_stopping(self, bsf_loss, early_stopping_lim, epochs_without_improvement, valid_loss,
                              validation_rate, improvements):
        if valid_loss < bsf_loss:
            print('improved validation loss from {:.3f} to {:.3f}'.format(bsf_loss, valid_loss))
            bsf_loss = valid_loss
            improvements.append(epochs_without_improvement)
            epochs_without_improvement = 0
            torch.save(self.model.state_dict(),
                       '{}/bsf_sup.pt'.format(self.model_dir))
        else:
            epochs_without_improvement += validation_rate
            print('Validation loss of {} did not improve on {}'.format(valid_loss, bsf_loss))
            print('Early stopping at {}/{}'.format(epochs_without_improvement, early_stopping_lim))
        return bsf_loss, epochs_without_improvement, improvements

    def validation(self, data_valid, step, data, loss_weight_base, value_weight, value_ratio):
        self.model.eval()
        running_valid_loss = 0
        for inp, out, out_real, lens in data_valid:
            loss, y_p = forecast_model.get_loss(inp=inp,
                                                out=out,
                                                lens=lens,
                                                cuda=True,
                                                gn=self.model,
                                                glucose_dat=data,
                                                criterion=self.criterion,
                                                base=loss_weight_base,
                                                out_real=out_real,
                                                value_weight=value_weight,
                                                value_ratio=value_ratio)
            step += 1
            running_valid_loss += loss.data.cpu().numpy()[0]
        running_valid_loss = running_valid_loss / len(data_valid)
        print('validation loss: {:.3f}'.format(running_valid_loss))
        self.writer.add_scalar(tag='valid_total_loss',
                               scalar_value=running_valid_loss,
                               global_step=step)
        self.model.train()
        return running_valid_loss

    def embed(self, dataloader, step, embed_batch=5):
        print('embed')
        embeddings = None
        metadata = []
        i = 0
        for dat, dat_past, dat_future, init, label in dataloader:
            x = Variable(dat.float().cuda())
            e = self.model.embed(x).data
            metadata += np.round(label.numpy(), 2).tolist()
            if embeddings is None:
                embeddings = e
            else:
                embeddings = torch.cat((embeddings, e))
            if i > embed_batch:
                break
            i += 1
        print(len(metadata))
        self.writer.add_embedding(mat=embeddings,
                                  metadata=metadata,
                                  global_step=step)

    def get_predictions(self, dataloader):
        self.model.eval()
        data = None
        data_past = None
        data_future = None
        y = None
        pred_pres = None
        pred_past = None
        pred_future = None
        pred = None
        for dat, dat_past, dat_future,  init, label in dataloader:
            print('evaluation batch')
            window_data = []
            window_data_past = []
            window_data_future = []
            window_y = []
            window_pred = []
            window_pred_pres = []
            window_pred_past = []
            window_pred_future = []
            if not self.window:
                dat = [dat]
                dat_past = [dat_past]
                dat_future = [dat_future]
            for window in range(len(dat)):
                x = Variable(dat[window].float().cuda())
                y_pred, x_pres, x_past, x_future = self.model.forward(x)
                y_pred = y_pred.data.cpu().numpy()
                if self.decode_present:
                    x_pres = x_pres.data.cpu().numpy()
                if self.decode_past:
                    x_past = x_past.data.cpu().numpy()
                if self.decode_future:
                    x_future = x_future.data.cpu().numpy()
                yt = label.numpy()
                xt_pres = dat[window].numpy()
                xt_past = dat_past[window].numpy()
                xt_future = dat_future[window].numpy()
                window_data.append(xt_pres)
                window_data_past.append(xt_past)
                window_data_future.append(xt_future)
                window_y.append(yt)
                window_pred.append(y_pred)
                window_pred_pres.append(x_pres)
                window_pred_past.append(x_past)
                window_pred_future.append(x_future)
            if data is None:
                data = [window_data]
                data_past = [window_data_past]
                data_future = [window_data_future]
                y = [window_y]
                pred_pres = [window_pred_pres]
                pred_past = [window_pred_past]
                pred_future = [window_pred_future]
                pred = [window_pred]
            else:
                data.append(window_data)
                data_past.append(window_data_past)
                data_future.append(window_data_future)
                y.append(window_y)
                if self.decode_present:
                    pred_pres.append(window_pred_pres)
                if self.decode_past:
                    pred_past.append(window_pred_past)
                if self.decode_future:
                    pred_future.append(window_pred_future)
                pred.append(window_pred)
        print('done getting predictions')
        return (data, data_past, data_future, y,
                pred_pres, pred_past, pred_future, pred)
