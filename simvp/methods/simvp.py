import pdb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from timm.utils import AverageMeter

from simvp.models import SimVP_Model
from .base_method import Base_method


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, config):
        return SimVP_Model(**config).to(self.device)

    def _predict(self, batch_x):
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def adjust_learning_rate(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.args.warmup_epoch:
            lr = self.args.lr * epoch / self.args.warmup_epoch 
        else:
            lr = self.args.min_lr + (self.args.lr - self.args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.args.warmup_epoch) / (self.args.epoch - self.args.warmup_epoch)))
        for param_group in self.model_optim.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, loss_mean, **kwargs):
        losses_m = AverageMeter()
        self.model.train()
        # if self.by_epoch:
        #     self.scheduler.step(epoch)
        self.model_optim.zero_grad()
        it = 0
        train_pbar = tqdm(train_loader)
        for batch_x, batch_y in train_pbar:
            # per iteration lr scheduler
            if it % self.args.accum_iter == 0:
                self.adjust_learning_rate(it / len(train_loader) + epoch)
            # self.model_optim.zero_grad()
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')
            pred_y = self._predict(batch_x)

            # only consider the last frame's prediction
            loss1 = self.criterion(pred_y[:, :-1], batch_y[:, :-1]) * 0.2
            loss = self.criterion(pred_y[:, -1], batch_y[:, -1]) + loss1
            if (it + 1) % self.args.accum_iter == 0:
                loss.backward()
                self.model_optim.step()
                self.model_optim.zero_grad()

            # self.model_optim.step()
            # if not self.by_epoch:
            #     self.scheduler.step()
            it += 1
            num_updates += 1
            loss_mean += loss.item()
            losses_m.update(loss.item(), batch_x.size(0))
            runner.call_hook('after_train_iter')
            runner._iter += 1

            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, loss_mean

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_val_iter')
            pred_y = self._predict(batch_x)
            # only consider the last frame
            loss = self.criterion(pred_y[:, -1], batch_y[:, -1])

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()
                                                  ), [pred_y[:, -1], batch_y[:, -1]], [preds_lst, trues_lst]))
            runner.call_hook('after_val_iter')
            if i * batch_x.shape[0] > 1000:
                break

            vali_pbar.set_description('vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())
        
        total_loss = np.average(total_loss)

        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        return preds, trues, total_loss

    def test_one_epoch(self, runner, test_loader, **kwargs):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        test_pbar = tqdm(test_loader)
        for batch_x, batch_y in test_pbar:
            runner.call_hook('before_val_iter')
            pred_y = self._predict(batch_x.to(self.device))

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))
            runner.call_hook('after_val_iter')

        inputs, trues, preds = map(
            lambda data: np.concatenate(data, axis=0), [inputs_lst, trues_lst, preds_lst])
        return inputs, trues, preds
