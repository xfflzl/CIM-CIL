import os
import math
import logging
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from rational.torch import Rational

from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8


class DER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = DERNet(args['convnet_type'], False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self.save_checkpoint()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for m in self._network.convnets[i].modules():
                    if not isinstance(m, Rational):
                        for p in m.parameters(): p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train', 
                                                 appendent=self._get_memory(), randombbxblur_class_list=np.arange(self._known_classes, self._total_classes), 
                                                 bbxblur_class_list=np.arange(self._known_classes))
        self.train_loader = DataLoader(train_dataset, batch_size=self._args['batch_size'], shuffle=True, num_workers=self._args['num_workers'])
        trainval_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train', 
                                                 appendent=self._get_memory(), bbxblur_class_list=np.arange(self._known_classes))
        self.trainval_loader = DataLoader(trainval_dataset, batch_size=self._args['meta_batch_size'], shuffle=True, num_workers=self._args['num_workers'])
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self._args['batch_size'], shuffle=False, num_workers=self._args['num_workers'])
        cam_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='cam', ret_side_information=True)
        self.cam_loader = DataLoader(cam_dataset, batch_size=self._args['cam_batch_size'], shuffle=False, num_workers=self._args['num_workers'])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(data_manager, self.train_loader, self.trainval_loader, self.test_loader)
        self._gen_bbx(data_manager, self.cam_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module

    def train(self):
        self._network.train()
        if isinstance(self._network, nn.DataParallel):
            self._network.module.convnets[-1].train()
        else:
            self._network.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                if isinstance(self._network, nn.DataParallel):
                    self._network.module.convnets[i].eval()
                else:
                    self._network.convnets[i].eval()

    def _train(self, data_manager, train_loader, trainval_loader, test_loader):
        self._network.to(self._device)
        if os.path.exists(os.path.join(self._args['logdir'], '{}.pkl').format(self._cur_task)):
            logging.info('Skip the training of {}-th task'.format(self._cur_task))
            self._network.load_state_dict(torch.load(os.path.join(self._args['logdir'], '{}.pkl').format(self._cur_task))['model_state_dict'])
            return
        theta_parameters = OrderedDict((n, p) for (n, p) in self._network.named_parameters() if p.requires_grad and 'ratl' not in n)
        phi_parameters = OrderedDict((n, p) for (n, p) in self._network.named_parameters() if p.requires_grad and 'ratl' in n)
        if self._cur_task == 0:
            theta_optimizer = optim.SGD([p for (n, p) in theta_parameters.items()], lr=self._args['init_theta_lr'], momentum=self._args['momentum'], weight_decay=self._args['init_weight_decay'])
            theta_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=theta_optimizer, milestones=self._args['init_milestones'], gamma=self._args['lr_decay'])
            phi_optimizer = optim.Adam([p for (n, p) in phi_parameters.items()], lr=self._args['init_phi_lr'], weight_decay=self._args['init_weight_decay'])
            phi_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=phi_optimizer, milestones=self._args['init_milestones'], gamma=self._args['lr_decay'])
            self._init_train(data_manager, train_loader, trainval_loader, test_loader, theta_optimizer, theta_scheduler, phi_optimizer, phi_scheduler)
        else:
            theta_optimizer = optim.SGD([p for (n, p) in theta_parameters.items()], lr=self._args['incre_theta_lr'], momentum=self._args['momentum'], weight_decay=self._args['incre_weight_decay'])
            theta_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=theta_optimizer, milestones=self._args['incre_milestones'], gamma=self._args['lr_decay'])
            phi_optimizer = optim.Adam([p for (n, p) in phi_parameters.items()], lr=self._args['incre_phi_lr'], weight_decay=self._args['incre_weight_decay'])
            phi_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=phi_optimizer, milestones=self._args['incre_milestones'], gamma=self._args['lr_decay'])
            self._update_representation(data_manager, train_loader, trainval_loader, test_loader, theta_optimizer, theta_scheduler, phi_optimizer, phi_scheduler)
            if isinstance(self._network, nn.DataParallel):
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)
    
    def _init_train(self, data_manager, train_loader, trainval_loader, test_loader, theta_optimizer, theta_scheduler, phi_optimizer, phi_scheduler):
        prog_bar = tqdm(range(self._args['init_epochs']))
        for _, epoch in enumerate(prog_bar):
            if epoch % self._args['cam_update_interval'] == 0 and epoch != 0 and epoch != (self._args['init_epochs'] - 1):
                self._gen_bbx(data_manager, self.cam_loader)
                train_loader.dataset.update_randombbxblur_prob((epoch // self._args['cam_update_interval']) * self._args['randombbxblur_prob_incre'])
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, _) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)
                theta_optimizer.zero_grad()
                phi_optimizer.zero_grad()
                loss.backward()
                theta_optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            self._network.eval()
            inner_losses, outer_losses = 0.0, 0.0
            fc_weights = self._network.module.fc.weight.data if isinstance(self._network, nn.DataParallel) else self._network.fc.weight.data
            trainval_iter = iter(trainval_loader)
            for i in range(self._args['num_meta_iters']):
                _, inputs, targets, _ = trainval_iter.next()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                bz, NC, H, W = inputs.shape
                H_compressed, W_compressed = int(H / math.sqrt(self._args['compress_ratio'])), int(W / math.sqrt(self._args['compress_ratio']))
                ratl_outputs = self._network(inputs, activation='ratl')
                fmaps, logits = ratl_outputs['fmaps'], ratl_outputs['logits']
                div_loss = F.cross_entropy(logits, targets)
                _, nc, h, w = fmaps.shape
                cams = torch.bmm(fc_weights[targets].detach().unsqueeze(1), fmaps.reshape(bz, nc, h * w))
                cams = cams.reshape(bz, 1, h, w)
                cams = cams - cams.min(3, keepdim=True)[0].min(2, keepdim=True)[0]
                cams = cams / cams.max(3, keepdim=True)[0].max(2, keepdim=True)[0]
                cams_resized = F.interpolate(cams, size=(H, W), mode='bilinear')
                cams_resized = cams_resized.expand(-1, NC, -1, -1)
                inputs_compressed = F.interpolate(inputs, size=(H_compressed, W_compressed), mode='nearest')
                inputs_compressed = F.interpolate(inputs_compressed, size=(H, W), mode='nearest')
                hybrid_inputs = cams_resized * inputs + (1.0 - cams_resized) * inputs_compressed
                inner_loss = F.cross_entropy(self._network(hybrid_inputs)['logits'], targets)
                inner_losses += inner_loss.item()
                theta_optimizer.zero_grad()
                phi_optimizer.zero_grad()
                fast_weights = OrderedDict((n, p) for (n, p) in self._network.named_parameters() if p.requires_grad and 'ratl' not in n and 'fc' not in n)
                grads = torch.autograd.grad(inner_loss, [p for (n, p) in fast_weights.items()], create_graph=True)
                fast_weights = OrderedDict((n, p - theta_scheduler.get_lr()[0] * g) for ((n, p), g) in zip(fast_weights.items(), grads))
                _, inputs, targets, _ = trainval_iter.next()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outer_loss = F.cross_entropy(self._network(inputs, fast_weights)['logits'], targets) + self._args['div_coef'] * div_loss
                outer_losses += outer_loss.item()
                outer_loss.backward()
                phi_optimizer.step()
            theta_scheduler.step()
            phi_scheduler.step()

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Clf_loss {:.3f}, Inner_loss {:.3f}, Outer_loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self._args['init_epochs'], losses / len(train_loader), inner_losses / self._args['num_meta_iters'], outer_losses / self._args['num_meta_iters'], train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Clf_loss {:.3f}, Inner_loss {:.3f}, Outer_loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self._args['init_epochs'], losses / len(train_loader), inner_losses / self._args['num_meta_iters'], outer_losses / self._args['num_meta_iters'], train_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, data_manager, train_loader, trainval_loader, test_loader, theta_optimizer, theta_scheduler, phi_optimizer, phi_scheduler):
        prog_bar = tqdm(range(self._args['incre_epochs']))
        for _, epoch in enumerate(prog_bar):
            if epoch % self._args['cam_update_interval'] == 0 and epoch != 0 and epoch != (self._args['incre_epochs'] - 1):
                self._gen_bbx(data_manager, self.cam_loader)
                train_loader.dataset.update_randombbxblur_prob((epoch // self._args['cam_update_interval']) * self._args['randombbxblur_prob_incre'])
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, _) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0, aux_targets - self._known_classes + 1, 0)
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                loss = loss_clf + loss_aux
                theta_optimizer.zero_grad()
                phi_optimizer.zero_grad()
                loss.backward()
                theta_optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            self._network.eval()
            inner_losses, outer_losses = 0.0, 0.0
            fc_weights = self._network.module.fc.weight.data if isinstance(self._network, nn.DataParallel) else self._network.fc.weight.data
            trainval_iter = iter(trainval_loader)
            for i in range(self._args['num_meta_iters']):
                _, inputs, targets, _ = trainval_iter.next()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                bz, NC, H, W = inputs.shape
                H_compressed, W_compressed = int(H / math.sqrt(self._args['compress_ratio'])), int(W / math.sqrt(self._args['compress_ratio']))
                ratl_outputs = self._network(inputs, activation='ratl')
                fmaps, logits = ratl_outputs['fmaps'], ratl_outputs['logits']
                div_loss = F.cross_entropy(logits, targets)
                _, nc, h, w = fmaps.shape
                cams = torch.bmm(fc_weights[targets].detach().unsqueeze(1), fmaps.reshape(bz, nc, h * w))
                cams = cams.reshape(bz, 1, h, w)
                cams = cams - cams.min(3, keepdim=True)[0].min(2, keepdim=True)[0]
                cams = cams / cams.max(3, keepdim=True)[0].max(2, keepdim=True)[0]
                cams_resized = F.interpolate(cams, size=(H, W), mode='bilinear')
                cams_resized = cams_resized.expand(-1, NC, -1, -1)
                inputs_compressed = F.interpolate(inputs, size=(H_compressed, W_compressed), mode='nearest')
                inputs_compressed = F.interpolate(inputs_compressed, size=(H, W), mode='nearest')
                hybrid_inputs = cams_resized * inputs + (1.0 - cams_resized) * inputs_compressed
                inner_outputs = self._network(hybrid_inputs)
                inner_loss_clf = F.cross_entropy(inner_outputs['logits'], targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0, aux_targets - self._known_classes + 1, 0)
                inner_loss_aux = F.cross_entropy(inner_outputs['aux_logits'], aux_targets)
                inner_loss = inner_loss_clf + inner_loss_aux
                inner_losses += inner_loss.item()
                theta_optimizer.zero_grad()
                phi_optimizer.zero_grad()
                fast_weights = OrderedDict((n, p) for (n, p) in self._network.named_parameters() if p.requires_grad and 'ratl' not in n and 'fc' not in n)
                grads = torch.autograd.grad(inner_loss, [p for (n, p) in fast_weights.items()], create_graph=True)
                fast_weights = OrderedDict((n, p - theta_scheduler.get_lr()[0] * g) for ((n, p), g) in zip(fast_weights.items(), grads))
                _, inputs, targets, _ = trainval_iter.next()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outer_loss = F.cross_entropy(self._network(inputs, fast_weights)['logits'], targets) + self._args['div_coef'] * div_loss
                outer_losses += outer_loss.item()
                outer_loss.backward()
                phi_optimizer.step()
            theta_scheduler.step()
            phi_scheduler.step()
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Clf_loss {:.3f}, Inner_loss {:.3f}, Outer_loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self._args['incre_epochs'], losses / len(train_loader), inner_losses / self._args['num_meta_iters'], outer_losses / self._args['num_meta_iters'], train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Clf_loss {:.3f}, Inner_loss {:.3f}, Outer_loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self._args['incre_epochs'], losses / len(train_loader), inner_losses / self._args['num_meta_iters'], outer_losses / self._args['num_meta_iters'], train_acc)
            prog_bar.set_description(info)
        logging.info(info)
