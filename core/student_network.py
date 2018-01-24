from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from misc.utils import to_var


class StudentNetwork(nn.Module):
    def __init__(self, configs):
        '''
        Descriptions:
            Some details of the paper:
            1. Use ResNet as the CNN student model, or use LSTM as the RNN student model.
            2. SGD for CNN, Adam for RNN.
            3.

        :param configs:
        '''
        super(StudentNetwork, self).__init__()
        self.base_model = configs['base_model']
        self.evaluator = configs['evaluator']

    def forward(self, data, configs):
        inputs, labels = data['inputs'], data['labels']
        # import pdb
        # pdb.set_trace()
        predicts = self.base_model(inputs)
        eval_res = self.evaluator(predicts, labels)
        return predicts, eval_res

    def fit(self, configs):
        self.base_model.train()
        dataloader, optimizer = configs['dataloader'], configs['optimizer']
        try:
            flag = True
            total_steps = len(dataloader)
        except:
            flag = False
            total_steps = 1
        current_epoch = configs['current_epoch']
        total_epochs = configs['total_epochs']
        teacher_updates = configs.get('policy_step', -1)
        logger = configs['logger']

        all_correct = 0
        all_samples = 0
        loss_average = 0

        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            if flag:
                inputs = to_var(inputs)
                labels = to_var(labels)
            predicts = self.base_model(inputs)

            eval_res = self.evaluator(predicts, labels)
            num_correct = eval_res['num_correct']
            num_samples = eval_res['num_samples']
            # logger.info('num_samples %d, num_correct %d'%(num_samples, num_correct))
            loss = eval_res['loss']
            all_correct += num_correct
            all_samples += num_samples
            loss.backward()
            optimizer.step()
            logger.info('Policy Steps: [%d] Train: ----- Iteration [%d], loss: %5.4f, accuracy: %5.4f(%5.4f)' % (
                teacher_updates, current_epoch+1, loss.cpu().data[0], num_correct/num_samples, all_correct/all_samples))
            loss_average += loss.cpu().data[0]
        return loss_average/total_steps

    def val(self, configs):
        self.base_model.eval()
        dataloader = configs['dataloader']
        total_steps = len(dataloader)

        all_correct = 0
        all_samples = 0
        loss_average = 0
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = to_var(inputs)
            labels = to_var(labels)
            predicts = self.base_model(inputs)
            eval_res = self.evaluator(predicts, labels)
            num_correct = eval_res['num_correct']
            num_samples = eval_res['num_samples']
            all_correct += num_correct
            all_samples += num_samples
            # logger.info('Eval: Epoch [%d/%d], Iteration [%d/%d], accuracy: %5.4f(%5.4f)' % (
            #    current_epoch, total_epochs, idx, total_steps, num_correct/num_samples, all_correct/all_samples))
            loss_average += eval_res['loss'].cpu().data[0]

        return all_correct/all_samples, loss_average/total_steps




