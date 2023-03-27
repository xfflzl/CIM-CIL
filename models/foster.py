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
from utils.inc_net import FOSTERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8

class FOSTER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = FOSTERNet(args['convnet_type'], False)
        self._snet = None
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.per_cls_weights = None
        self.is_teacher_wa = args["is_teacher_wa"]
        self.is_student_wa = args["is_student_wa"]
        self.lambda_okd = args["lambda_okd"]
        self.wa_value = args["wa_value"]
        self.oofc = args["oofc"].lower()

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self.save_checkpoint()
        if self._cur_task > 0: self.save_snet_checkpoint()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for m in self._network.convnets[0].modules():
                 if not isinstance(m, Rational):
                     for p in m.parameters(): p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(
            count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train', 
                                                 appendent=self._get_memory(), randombbxblur_class_list=np.arange(self._known_classes, self._total_classes), 
                                                 bbxblur_class_list=np.arange(self._known_classes), ret_noother_information=True)
        self.train_loader = DataLoader(train_dataset, batch_size=self._args['batch_size'], shuffle=True, num_workers=self._args['num_workers'])
        trainval_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train', 
                                                 appendent=self._get_memory(), bbxblur_class_list=np.arange(self._known_classes), ret_noother_information=True)
        self.trainval_loader = DataLoader(trainval_dataset, batch_size=self._args['meta_batch_size'], shuffle=True, num_workers=self._args['num_workers'])
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test', ret_noother_information=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self._args['batch_size'], shuffle=False, num_workers=self._args['num_workers'])
        cam_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='cam', ret_side_information=True)
        self.cam_loader = DataLoader(cam_dataset, batch_size=self._args['cam_batch_size'], shuffle=False, num_workers=self._args['num_workers'])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(data_manager, self.train_loader, self.trainval_loader, self.test_loader)
        self._gen_bbx(data_manager, self.cam_loader, show_info=True)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network_module_ptr.train()
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.convnets[0].eval()

    def _train(self, data_manager, train_loader, trainval_loader, test_loader):
        self._network.to(self._device)
        if os.path.exists(os.path.join(self._args['logdir'], '{}.pkl').format(self._cur_task)):
            logging.info('Skip the base training of {}-th task'.format(self._cur_task))
            self._network.load_state_dict(torch.load(os.path.join(self._args['logdir'], '{}.pkl').format(self._cur_task))['model_state_dict'])
            if self._cur_task > 0: self._feature_compression(train_loader, test_loader)
            return
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        theta_parameters = OrderedDict((n, p) for (n, p) in self._network.named_parameters() if p.requires_grad and 'ratl' not in n)
        phi_parameters = OrderedDict((n, p) for (n, p) in self._network.named_parameters() if p.requires_grad and 'ratl' in n)
        if self._cur_task == 0:
            theta_optimizer = optim.SGD([p for (n, p) in theta_parameters.items()], lr=self._args['init_theta_lr'], momentum=self._args['momentum'], weight_decay=self._args['init_weight_decay'])
            theta_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=theta_optimizer, T_max=self.args["init_epochs"])
            phi_optimizer = optim.Adam([p for (n, p) in phi_parameters.items()], lr=self._args['init_phi_lr'], weight_decay=self._args['init_weight_decay'])
            phi_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=phi_optimizer, T_max=self._args['init_epochs'])
            self._init_train(data_manager, train_loader, trainval_loader, test_loader, theta_optimizer, theta_scheduler, phi_optimizer, phi_scheduler)
        else:
            cls_num_list = [self.samples_old_class] * self._known_classes + [self.samples_new_class(data_manager, i) for i in range(self._known_classes, self._total_classes)]

            effective_num = 1.0 - np.power(self.beta1, cls_num_list)
            per_cls_weights = (1.0 - self.beta1) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)

            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)

            theta_optimizer = optim.SGD([p for (n, p) in theta_parameters.items()], lr=self._args['incre_theta_lr'], momentum=self._args['momentum'], weight_decay=self._args['incre_weight_decay'])
            theta_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=theta_optimizer, T_max=self.args["incre_epochs"])
            phi_optimizer = optim.Adam([p for (n, p) in phi_parameters.items()], lr=self._args['incre_phi_lr'], weight_decay=self._args['incre_weight_decay'])
            phi_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=phi_optimizer, T_max=self._args['incre_epochs'])
            
            self._feature_boosting(data_manager, train_loader, trainval_loader, test_loader, theta_optimizer, theta_scheduler, phi_optimizer, phi_scheduler)
            if self.is_teacher_wa:
                self._network_module_ptr.weight_align(self._known_classes, self._total_classes-self._known_classes, self.wa_value)
            else:
                logging.info("do not weight align teacher!")

            effective_num = 1.0 - np.power(self.beta2, cls_num_list)
            per_cls_weights = (1.0 - self.beta2) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)
            self._feature_compression(train_loader, test_loader)

    def _init_train(self, data_manager, train_loader, trainval_loader, test_loader, theta_optimizer, theta_scheduler, phi_optimizer, phi_scheduler):
        prog_bar = tqdm(range(self.args["init_epochs"]))
        num_meta_iters = int((self._total_classes - self._known_classes) * self._args['num_meta_iters_per_cls'])
        for _, epoch in enumerate(prog_bar):
            if epoch % self._args['cam_update_interval'] == 0 and epoch != 0 and epoch != (self._args['init_epochs'] - 1):
                self._gen_bbx(data_manager, self.cam_loader)
                train_loader.dataset.update_randombbxblur_prob((epoch // self._args['cam_update_interval']) * self._args['randombbxblur_prob_incre'])
            self.train()
            losses = 0.
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
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
            div_losses, reg_losses = 0.0, 0.0
            fc_weights = self._network.module.fc.weight.data if isinstance(self._network, nn.DataParallel) else self._network.fc.weight.data
            trainval_iter = iter(trainval_loader)
            for i in range(num_meta_iters):
                inputs, targets = trainval_iter.next()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                bz, NC, H, W = inputs.shape
                H_compressed, W_compressed = int(H / math.sqrt(self._args['compress_ratio'])), int(W / math.sqrt(self._args['compress_ratio']))
                ratl_outputs = self._network(inputs, activation='ratl')
                fmaps, logits = ratl_outputs['fmaps'], ratl_outputs['logits']
                div_loss = F.cross_entropy(logits, targets)
                div_losses += div_loss.item()
                _, nc, h, w = fmaps.shape
                cams = torch.bmm(fc_weights[targets].detach().unsqueeze(1), fmaps.reshape(bz, nc, h * w))
                cams = cams.reshape(bz, 1, h, w)
                cams = cams - cams.min(3, keepdim=True)[0].min(2, keepdim=True)[0]
                cams = cams / cams.max(3, keepdim=True)[0].max(2, keepdim=True)[0]
                reg_loss = F.mse_loss(cams, torch.zeros_like(cams))
                reg_losses += reg_loss.item()
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
                fast_weights = OrderedDict((n, p - theta_scheduler.get_lr()[0] * self._args['lr_ratio'] * g) for ((n, p), g) in zip(fast_weights.items(), grads))
                inputs, targets = trainval_iter.next()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outer_loss = F.cross_entropy(self._network(inputs, fast_weights)['logits'], targets) + self._args['div_coef'] * div_loss + self._args['reg_coef'] * reg_loss
                outer_losses += outer_loss.item()
                outer_loss.backward()
                nn.utils.clip_grad_norm_(phi_optimizer.param_groups[0]['params'], self._args['max_phi_grad_norm'])
                phi_optimizer.step()
            phi_scheduler.step()
            theta_scheduler.step()
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Clf_loss {:.3f}, Inner_loss {:.3f}, Outer_loss {:.3f}, Div_loss {:.3f}, Reg_loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self._args['init_epochs'], losses / len(train_loader), inner_losses / num_meta_iters if num_meta_iters > 0 else 0.0, outer_losses / num_meta_iters if num_meta_iters > 0 else 0.0, div_losses / num_meta_iters if num_meta_iters > 0 else 0.0, reg_losses / num_meta_iters if num_meta_iters > 0 else 0.0, train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Clf_loss {:.3f}, Inner_loss {:.3f}, Outer_loss {:.3f}, Div_loss {:.3f}, Reg_loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self._args['init_epochs'], losses / len(train_loader), inner_losses / num_meta_iters if num_meta_iters > 0 else 0.0, outer_losses / num_meta_iters if num_meta_iters > 0 else 0.0, div_losses / num_meta_iters if num_meta_iters > 0 else 0.0, reg_losses / num_meta_iters if num_meta_iters > 0 else 0.0, train_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def _feature_boosting(self, data_manager, train_loader, trainval_loader, test_loader, theta_optimizer, theta_scheduler, phi_optimizer, phi_scheduler):
        prog_bar = tqdm(range(self._args["incre_epochs"]))
        num_meta_iters = int((self._total_classes - self._known_classes) * self._args['num_meta_iters_per_cls'])
        for _, epoch in enumerate(prog_bar):
            if epoch % self._args['cam_update_interval'] == 0 and epoch != 0 and epoch != (self._args['init_epochs'] - 1):
                self._gen_bbx(data_manager, self.cam_loader)
                train_loader.dataset.update_randombbxblur_prob((epoch // self._args['cam_update_interval']) * self._args['randombbxblur_prob_incre'])
            self.train()
            losses = 0.
            losses_clf = 0.
            losses_fe = 0.
            losses_kd = 0.
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                outputs = self._network(inputs)
                logits, fe_logits, old_logits = outputs["logits"], outputs["fe_logits"], outputs["old_logits"].detach()
                loss_clf = F.cross_entropy(logits / self.per_cls_weights, targets)
                loss_fe = F.cross_entropy(fe_logits, targets)
                loss_kd = self.lambda_okd * _KD_loss(logits[:, :self._known_classes], old_logits, self._args["T"])
                loss = loss_clf + loss_fe + loss_kd
                theta_optimizer.zero_grad()
                phi_optimizer.zero_grad()
                loss.backward()
                theta_optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                losses_kd += (self._known_classes / self._total_classes) * loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            
            self._network.eval()
            inner_losses, outer_losses = 0.0, 0.0
            div_losses, reg_losses = 0.0, 0.0
            fc_weights = self._network.module.fc.weight.data if isinstance(self._network, nn.DataParallel) else self._network.fc.weight.data
            trainval_iter = iter(trainval_loader)
            for i in range(num_meta_iters):
                inputs, targets = trainval_iter.next()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                bz, NC, H, W = inputs.shape
                H_compressed, W_compressed = int(H / math.sqrt(self._args['compress_ratio'])), int(W / math.sqrt(self._args['compress_ratio']))
                ratl_outputs = self._network(inputs, activation='ratl')
                fmaps, logits = ratl_outputs['fmaps'], ratl_outputs['logits']
                div_loss = F.cross_entropy(logits, targets)
                div_losses += div_loss.item()
                _, nc, h, w = fmaps.shape
                cams = torch.bmm(fc_weights[targets].detach().unsqueeze(1), fmaps.reshape(bz, nc, h * w))
                cams = cams.reshape(bz, 1, h, w)
                cams = cams - cams.min(3, keepdim=True)[0].min(2, keepdim=True)[0]
                cams = cams / cams.max(3, keepdim=True)[0].max(2, keepdim=True)[0]
                reg_loss = F.mse_loss(cams, torch.zeros_like(cams))
                reg_losses += reg_loss.item()
                cams_resized = F.interpolate(cams, size=(H, W), mode='bilinear')
                cams_resized = cams_resized.expand(-1, NC, -1, -1)
                inputs_compressed = F.interpolate(inputs, size=(H_compressed, W_compressed), mode='nearest')
                inputs_compressed = F.interpolate(inputs_compressed, size=(H, W), mode='nearest')
                hybrid_inputs = cams_resized * inputs + (1.0 - cams_resized) * inputs_compressed
                inner_outputs = self._network(hybrid_inputs)
                inner_loss_clf = F.cross_entropy(inner_outputs['logits'] / self.per_cls_weights, targets)
                inner_loss_fe = F.cross_entropy(inner_outputs["fe_logits"], targets)
                inner_loss_kd = self.lambda_okd * _KD_loss(inner_outputs['logits'][:, :self._known_classes], inner_outputs['old_logits'].detach(), self._args["T"])
                inner_loss = inner_loss_clf + inner_loss_fe + inner_loss_kd
                inner_losses += inner_loss.item()
                theta_optimizer.zero_grad()
                phi_optimizer.zero_grad()
                fast_weights = OrderedDict((n, p) for (n, p) in self._network.named_parameters() if p.requires_grad and 'ratl' not in n and 'fc' not in n)
                grads = torch.autograd.grad(inner_loss, [p for (n, p) in fast_weights.items()], create_graph=True)
                fast_weights = OrderedDict((n, p - theta_scheduler.get_lr()[0] * self._args['lr_ratio'] * g) for ((n, p), g) in zip(fast_weights.items(), grads))
                inputs, targets = trainval_iter.next()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outer_loss = F.cross_entropy(self._network(inputs, fast_weights)['logits'], targets) + self._args['div_coef'] * div_loss + self._args['reg_coef'] * reg_loss
                outer_losses += outer_loss.item()
                outer_loss.backward()
                nn.utils.clip_grad_norm_(phi_optimizer.param_groups[0]['params'], math.sqrt(2.0) * self._args['max_phi_grad_norm'])
                phi_optimizer.step()
            phi_scheduler.step()
            theta_scheduler.step()
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Clf_loss {:.3f}, Inner_loss {:.3f}, Outer_loss {:.3f}, Div_loss {:.3f}, Reg_loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self._args['incre_epochs'], losses / len(train_loader), inner_losses / num_meta_iters if num_meta_iters > 0 else 0.0, outer_losses / num_meta_iters if num_meta_iters > 0 else 0.0, div_losses / num_meta_iters if num_meta_iters > 0 else 0.0, reg_losses / num_meta_iters if num_meta_iters > 0 else 0.0, train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Clf_loss {:.3f}, Inner_loss {:.3f}, Outer_loss {:.3f}, Div_loss {:.3f}, Reg_loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self._args['incre_epochs'], losses / len(train_loader), inner_losses / num_meta_iters if num_meta_iters > 0 else 0.0, outer_losses / num_meta_iters if num_meta_iters > 0 else 0.0, div_losses / num_meta_iters if num_meta_iters > 0 else 0.0, reg_losses / num_meta_iters if num_meta_iters > 0 else 0.0, train_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def _feature_compression(self, train_loader, test_loader):
        self._snet = FOSTERNet(self._args['convnet_type'], False)
        self._snet.update_fc(self._total_classes)
        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet, self._multiple_gpus)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)
        self._snet_module_ptr.convnets[0].load_state_dict(self._network_module_ptr.convnets[0].state_dict())
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)
        if os.path.exists(os.path.join(self._args['logdir'], 'snet_{}.pkl').format(self._cur_task)):
            logging.info('Skip the snet training of {}-th task'.format(self._cur_task))
            self._snet.load_state_dict(torch.load(os.path.join(self._args['logdir'], 'snet_{}.pkl').format(self._cur_task))['model_state_dict'])
        else:
            theta_parameters = OrderedDict((n, p) for (n, p) in self._snet.named_parameters() if p.requires_grad and 'ratl' not in n)
            optimizer = optim.SGD([p for (n, p) in theta_parameters.items()], lr=self._args["incre_theta_lr"], momentum=self._args["momentum"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["compression_epochs"])
            self._network.eval()
            prog_bar = tqdm(range(self._args["compression_epochs"]))
            for _, epoch in enumerate(prog_bar):
                self._snet.train()
                losses = 0.
                correct, total = 0, 0
                for i, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                    dark_logits = self._snet(inputs)["logits"]
                    with torch.no_grad():
                        outputs = self._network(inputs)
                        logits, old_logits, fe_logits = outputs["logits"], outputs["old_logits"], outputs["fe_logits"]
                    loss_dark = self.BKD(dark_logits, logits, self._args["T"])
                    loss = loss_dark
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()
                    _, preds = torch.max(dark_logits[:targets.shape[0]], dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)
                scheduler.step()
                train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
                if epoch % 5 == 0:
                    test_acc = self._compute_accuracy(self._snet, test_loader)
                    info = 'SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}'.format(
                        self._cur_task, epoch+1, self._args["compression_epochs"], losses/len(train_loader), train_acc, test_acc)
                else:
                    info = 'SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}'.format(
                        self._cur_task, epoch+1, self._args["compression_epochs"], losses/len(train_loader),  train_acc)
                prog_bar.set_description(info)
            logging.info(info)
            if len(self._multiple_gpus) > 1:
                self._snet = self._snet.module
            if self.is_student_wa:
                self._snet.weight_align(
                    self._known_classes, self._total_classes-self._known_classes, self.wa_value)
            else:
                logging.info("do not weight align student!")

        self._snet.eval()
        y_pred, y_true = [], []
        for _, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._snet(inputs)['logits']
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cnn_accy = self._evaluate(y_pred, y_true)
        logging.info("darknet eval: ")
        logging.info('CNN top1 curve: {}'.format(cnn_accy['top1']))
        logging.info('CNN top5 curve: {}'.format(cnn_accy['top5']))

    @property
    def samples_old_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, 'Total classes is 0'
            return (self._memory_size // self._known_classes)

    def samples_new_class(self, data_manager, index):
        return data_manager.getlen(index)

    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred/T, dim=1)
        soft = torch.softmax(soft/T, dim=1)
        soft = soft*self.per_cls_weights
        soft = soft / soft.sum(1)[:, None]
        return -1*torch.mul(soft, pred).sum()/pred.shape[0]
        
    def save_snet_checkpoint(self):
        self._snet.cpu()
        snet_dict = {
            'tasks': self._cur_task,
            'model_state_dict': self._snet.state_dict(),
        }
        torch.save(snet_dict, os.path.join(self._args['logdir'], 'snet_{}.pkl'.format(self._cur_task)))


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1*torch.mul(soft, pred).sum()/pred.shape[0]
