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

    # def forward(self, data, configs):
    #     pass

    def fit_teacher(self, configs):
        teacher = self.teacher_net
        student = self.student_net
        # ==================== fetch configs ==================
        max_t = configs.get('max_t', 100000)
        tau = configs.get('tau', 0.8)
        threshold = configs.get('threshold', 0.5)
        M = configs.get('M', 128)
        max_non_increasing_steps = 100
        num_classes = 10
        state_func = configs['state_func']
        teacher_dataloader = configs['dataloader']['teacher']
        student_dataloader = configs['dataloader']['student']
        dev_dataloader = configs['dataloader']['dev']
        test_dataloader = configs['dataloader']['test']
        teacher_optimizer = configs['optimizer']['teacher']
        student_optimizer = configs['optimizer']['student']
        current_epoch = configs['current_epoch']
        total_epochs = configs['total_epochs']
        logger = configs['logger']

        # ================== init tracking history =============
        rewards = []
        training_loss_history = []
        val_loss_history = []

        non_increasing_steps = 0

        while True:
            i_tau = 0
            actions = []

            while i_tau < max_t:
                i_tau += 1
                count = 0
                input_pool = []
                label_pool = []
                # ================== collect training batch ============
                for idx, (inputs, labels) in teacher_dataloader:
                    # TODO: features for the teacher network
                    state_configs = {
                        'num_classes': num_classes,
                        'labels': labels,
                        'inputs': inputs,
                        'student': student,
                        'current_iter': i_tau,
                        'max_iter': max_t,
                        'train_loss_history': training_loss_history,
                        'val_loss_history': val_loss_history
                    }
                    states = state_func(state_configs) # TODO: implement the function for computing state
                    predicts = teacher(states, None)
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

                # ================== prepare training data ============
                inputs = torch.cat(input_pool, 0)
                labels = torch.cat(label_pool, 0)
                st_configs = {
                    'dataloader': to_generator([inputs, labels]),
                    'optimizer': student_optimizer,
                    'current_epoch': current_epoch,
                    'total_epochs': total_epochs,
                    'logger': logger
                }
                # ================= feed the selected batch ============
                train_loss = student.fit(st_configs)
                training_loss_history.append(train_loss)

                # ================ test on dev set =====================
                st_configs['dataloader'] = dev_dataloader
                acc, val_loss = student.val(st_configs)
                val_loss_history.append(val_loss)
                # ============== check if reach the expected accuracy ==
                if acc >= tau:
                    teacher_optimizer.zero_grad()
                    reward = -math.log(i_tau/max_t)
                    baseline = 0 if len(rewards) == 0 else sum(rewards)/len(rewards)
                    last_reward = 0 if len(rewards) == 0 else rewards[-1]
                    if last_reward >= reward:
                        non_increasing_steps += 1
                    loss = -sum([torch.sum(_) for _ in actions])*(reward - baseline)
                    logger.info('Policy: Epoch [%d/%d], stops at %d/%d to achieve %5.4f, loss: %5.4f, reward: %5.4f(%5.4f)'
                                %(current_epoch, total_epochs, i_tau, max_t, acc, loss.cpu().data[0], reward, baseline))
                    rewards.append(reward)
                    loss.backward()
                    teacher_optimizer.step()
                    # ========== break for next batch =========================
                    break

            # ==================== policy converged (stopping criteria) ==========================
            if non_increasing_steps >= max_non_increasing_steps:
                break

    def val_teacher(self, configs):
        # TODO: test for the policy. Plotting the curve of #effective_samples-test_accuracy
        pass