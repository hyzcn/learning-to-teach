from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import math
import torch.nn as nn


from core.student_network import StudentNetwork
from core.teacher_network import TeacherNetwork

# ================== helper function ===============
def to_generator(data):
    yield data

# ==================================================

class TeacherStudentModel(nn.Module):

    def __init__(self, configs):
        super(TeacherStudentModel, self).__init__()
        self.student_net = StudentNetwork(configs['student_configs'])
        self.teacher_net = TeacherNetwork(configs['teacher_configs'])

    def forward(self, data, configs):
        pass

    def fit(self, configs):
        teacher = self.teacher_net
        student = self.student_net
        max_t = configs.get('max_t', 100000)
        tau = configs.get('tau', 0.8)
        threshold = configs.get('threshold', 0.5)
        M = configs.get('M', 128)

        teacher_dataloader = configs['dataloader']['teacher']
        student_dataloader = configs['dataloader']['student']
        dev_dataloader = configs['dataloader']['dev']
        test_dataloader = configs['dataloader']['test']

        teacher_optimizer = configs['optimizer']['teacher']
        student_optimizer = configs['optimizer']['student']

        current_epoch = configs['current_epoch']
        total_epochs = configs['total_epochs']

        logger = configs['logger']

        rewards = []
        non_increasing_steps = 0
        max_non_increasing_steps = 100
        while True:
            i_tau = 0
            actions = []

            while i_tau < max_t:
                i_tau += 1
                count = 0
                input_pool = []
                label_pool = []
                for idx, (inputs, labels) in teacher_dataloader:
                    # TODO: features for the teacher network
                    predicts = teacher(inputs, None)
                    indices = torch.nonzero(predicts >= threshold)
                    if len(indices) == 0:
                        continue
                    count += len(indices)
                    selected_inputs = torch.gather(inputs, 0, indices.squeeze()).view(-1, inputs.size(1))
                    selected_labels = torch.gather(labels, 0, indices.squeeze()).view(-1, 1)
                    input_pool.append(selected_inputs)
                    label_pool.append(selected_labels)
                    actions.append(torch.log(predicts))
                    if count >= M:
                        break
                inputs = torch.cat(input_pool, 0)
                labels = torch.cat(label_pool, 0)
                st_configs = {
                    'dataloader': to_generator([inputs, labels]),
                    'optimizer': student_optimizer,
                    'current_epoch': current_epoch,
                    'total_epochs': total_epochs,
                    'logger': logger
                }
                student.fit(st_configs)
                st_configs['dataloader'] = dev_dataloader
                acc = student.val(st_configs)
                if acc >= tau:
                    teacher_optimizer.zero_grad()
                    reward = -math.log(i_tau/max_t)
                    baseline = 0 if len(rewards) == 0 else sum(rewards)/len(rewards)
                    last_reward = 0 if len(rewards) == 0 else rewards[-1]
                    if last_reward >= reward:
                        non_increasing_steps += 1
                    loss = -sum([torch.sum(_) for _ in actions])*(reward - baseline)
                    logger.info('Policy: Epoch [%d/%d], Iteration [%d/%d], loss: %5.4f, reward: %5.4f(%5.4f)'
                                %(current_epoch, total_epochs, i_tau, max_t, loss.cpu().data[0], reward, baseline))
                    rewards.append(reward)
                    loss.backward()
                    teacher_optimizer.step()

                    if non_increasing_steps >= max_non_increasing_steps:
                        break



    def val(self, configs):
        pass