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
        self.configs = configs
        self.student_net = StudentNetwork(configs['student_configs'])
        self.teacher_net = TeacherNetwork(configs['teacher_configs'])

    # def forward(self, data, configs):
    #     pass

    def fit_teacher(self, configs):
        '''
        :param configs:
            Required:
                state_func: [function] used to compute the state vector

                dataloader: [dict]
                    teacher: teacher training data loader
                    student: student training data loader
                    dev: for testing the student model so as to compute reward for the teacher
                    test: student testing data loader

                optimizer: [dict]
                    teacher: the optimizer for teacher
                    student: the optimizer for student

                lr_scheduler: [dict]
                    teahcer: the learning rate scheduler for the teacher model
                    student: the learning rate scheduler for the student model

                <del>current_epoch: [int] the current epoch</del>
                <del>total_epochs: the max number of epochs to train the model</del>
                logger: the logger

            Optional:
                max_t: [int] [50,000]
                    the maximum number iterations before stopping the teaching
                    , and once reach this number, return a reward 0.
                tau: [float32] [0.8]
                    the expected accuracy of the student model on dev set
                threshold: [float32] [0.5]
                    the probability threshold for choosing a sample.
                M: [int] [128]
                    the required batch-size for training the student model.
                max_non_increasing_steps: [int] [10]
                    The maximum number of iterations of the reward not increasing.
                    If exceeds it, stop training the teacher model.
                num_classes: [int] [10]
                    the number of classes in the training set.
        :return:
        '''
        teacher = self.teacher_net
        student = self.student_net
        # ==================== fetch configs [optional] ===============
        max_t = configs.get('max_t', 50000)
        tau = configs.get('tau', 0.8)
        threshold = configs.get('threshold', 0.5)
        M = configs.get('M', 128)
        max_non_increasing_steps = configs.get('max_non_increasing_steps', 10)
        num_classes = configs.get('num_classes', 10)

        # =================== fetch configs [required] ================
        state_func = configs['state_func']
        teacher_dataloader = configs['dataloader']['teacher']
        # student_dataloader = configs['dataloader']['student']
        dev_dataloader = configs['dataloader']['dev']
        # test_dataloader = configs['dataloader']['test']
        teacher_optimizer = configs['optimizer']['teacher']
        student_optimizer = configs['optimizer']['student']
        teacher_lr_scheduler = configs['lr_scheduler']['teacher']
        student_lr_scheduler = configs['lr_scheduler']['student']
        logger = configs['logger']

        # ================== init tracking history ====================
        rewards = []
        training_loss_history = []
        val_loss_history = []

        non_increasing_steps = 0
        student_updates = 0
        teacher_updates = 0
        best_acc_on_dev = 0
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
                    selected_inputs = torch.gather(inputs, 0, indices.squeeze()).view(len(indices),
                                                                                      *inputs.size()[1:])
                    selected_labels = torch.gather(labels, 0, indices.squeeze()).view(-1, 1)
                    input_pool.append(selected_inputs)
                    label_pool.append(selected_labels)
                    actions.append(torch.log(predicts))
                    if count >= M:
                        break

                # ================== prepare training data =============
                inputs = torch.cat(input_pool, 0)
                labels = torch.cat(label_pool, 0)
                st_configs = {
                    'dataloader': to_generator([inputs, labels]),
                    'optimizer': student_optimizer,
                    'current_epoch': student_updates,
                    'total_epochs': -1,
                    'logger': logger
                }
                # ================= feed the selected batch ============
                train_loss = student.fit(st_configs)
                training_loss_history.append(train_loss)
                student_updates += 1
                student_lr_scheduler(student_optimizer, student_updates)

                # ================ test on dev set =====================
                st_configs['dataloader'] = dev_dataloader
                acc, val_loss = student.val(st_configs)
                best_acc_on_dev = acc if best_acc_on_dev < acc else best_acc_on_dev
                logger.info('Test on Dev: Iteration [%d], accuracy: %5.4f, best: %5.4f'%(student_updates,
                                                                                         acc, best_acc_on_dev))
                val_loss_history.append(val_loss)

                # ============== check if reach the expected accuracy ==
                # ============== or exceeds the max_t ==================
                if acc >= tau or i_tau == max_t:
                    teacher_optimizer.zero_grad()

                    reward = -math.log(i_tau/max_t)
                    baseline = 0 if len(rewards) == 0 else sum(rewards)/len(rewards)
                    last_reward = 0 if len(rewards) == 0 else rewards[-1]
                    if last_reward >= reward:
                        non_increasing_steps += 1
                    loss = -sum([torch.sum(_) for _ in actions])*(reward - baseline)
                    logger.info('Policy: Iterations [%d], stops at %d/%d to achieve %5.4f, loss: %5.4f, reward: %5.4f(%5.4f)'
                                %(teacher_updates, i_tau, max_t, acc, loss.cpu().data[0], reward, baseline))
                    rewards.append(reward)
                    loss.backward()
                    teacher_optimizer.step()
                    teacher_updates += 1
                    teacher_lr_scheduler(teacher_optimizer, teacher_updates)

                    # ========= reinitialize the student network =========
                    self.student_net.init_weights()
                    # ========== break for next batch ====================
                    break

            # ==================== policy converged (stopping criteria) ==
            if non_increasing_steps >= max_non_increasing_steps:
                return

    def val_teacher(self, configs):
        # TODO: test for the policy. Plotting the curve of #effective_samples-test_accuracy
        '''
        :param configs:
            Required:
                state_func
                dataloader: student/dev/test
                optimizer: student
                lr_scheduler: student
                logger
            Optional:
                threshold
                M
                num_classes
                max_t
                (Note: should be consistent with training)
        :return:
        '''
        teacher = self.teacher_net
        # ==================== train student from scratch ============
        self.student_net.init_weights()
        student = self.student_net
        # ==================== fetch configs [optional] ===============
        threshold = configs.get('threshold', 0.5)
        M = configs.get('M', 128)
        num_classes = configs.get('num_classes', 10)
        max_t = configs.get('max_t', 50000)
        # =================== fetch configs [required] ================
        state_func = configs['state_func']
        student_dataloader = configs['dataloader']['student']
        dev_dataloader = configs['dataloader']['dev']
        test_dataloader = configs['dataloader']['test']
        student_optimizer = configs['optimizer']['student']
        student_lr_scheduler = configs['lr_scheduler']['student']
        logger = configs['logger']

        # ================== init tracking history ====================
        training_loss_history = []
        val_loss_history = []

        student_updates = 0
        best_acc_on_dev = 0
        best_acc_on_test = 0
        i_tau = 0
        effective_num = 0
        effnum_acc_curves = []

        while i_tau < max_t:
            i_tau += 1
            count = 0
            input_pool = []
            label_pool = []
            # ================== collect training batch ============
            for idx, (inputs, labels) in student_dataloader:
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
                states = state_func(state_configs)  # TODO: implement the function for computing state
                predicts = teacher(states, None)
                indices = torch.nonzero(predicts >= threshold)
                if len(indices) == 0:
                    continue
                count += len(indices)
                selected_inputs = torch.gather(inputs, 0, indices.squeeze()).view(len(indices),
                                                                                  *inputs.size()[1:])
                selected_labels = torch.gather(labels, 0, indices.squeeze()).view(-1, 1)
                input_pool.append(selected_inputs)
                label_pool.append(selected_labels)
                if count >= M:
                    effective_num += count
                    break

            # ================== prepare training data =============
            inputs = torch.cat(input_pool, 0)
            labels = torch.cat(label_pool, 0)
            st_configs = {
                'dataloader': to_generator([inputs, labels]),
                'optimizer': student_optimizer,
                'current_epoch': student_updates,
                'total_epochs': -1,
                'logger': logger
            }
            # ================= feed the selected batch ============
            train_loss = student.fit(st_configs)
            training_loss_history.append(train_loss)
            student_updates += 1
            student_lr_scheduler(student_optimizer, student_updates)

            # ================ test on dev set =====================
            st_configs['dataloader'] = dev_dataloader
            acc, val_loss = student.val(st_configs)
            best_acc_on_dev = acc if best_acc_on_dev < acc else best_acc_on_dev
            logger.info('Test on Dev: Iteration [%d], accuracy: %5.4f, best: %5.4f' % (student_updates,
                                                                                       acc, best_acc_on_dev))
            val_loss_history.append(val_loss)

            # =============== test on test set ======================
            st_configs['dataloader'] = test_dataloader
            acc, test_loss = student.val(configs)
            best_acc_on_test = acc if best_acc_on_test < acc else best_acc_on_test
            logger.info('Testing Set: Iteration [%d], accuracy: %5.4f, best: %5.4f' % (student_updates,
                                                                                       acc, best_acc_on_test))
            effnum_acc_curves.append((effective_num, acc))
        return effnum_acc_curves
