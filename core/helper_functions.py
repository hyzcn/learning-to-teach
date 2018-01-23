from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable


def to_var(x, volatile=False, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile, requires_grad=requires_grad)


def state_func(configs):
    '''
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
    '''
    num_classes = configs['num_classes']
    labels = configs['labels']
    inputs = configs['inputs']
    student = configs['student']
    current_iter = configs['current_iter']
    max_iter = configs['max_iter']
    train_loss_history = configs['train_loss_history']
    val_loss_history = configs['val_loss_history']

    predicts, _ = student(inputs, None) # predicts are logits
    predicts = nn.Softmax()(predicts)

    n_samples = inputs.size(0)
    data_features = to_var(torch.zeros(n_samples, num_classes))
    data_features[range(n_samples), labels] = 1

    model_features = to_var(torch.zeros(n_samples, 3))
    model_features[:, 0] = current_iter / max_iter # current iteration number
    model_features[:, 1] = sum(train_loss_history)/len(train_loss_history) # averaged training loss
    model_features[:, 2] = min(val_loss_history)

    combined_features = to_var(torch.zeros(n_samples, 12))
    combined_features[:, :10] = predicts
    combined_features[:, 10:11] = -torch.log(predicts[range(n_samples), labels])
    mask = to_var(torch.ones(n_samples, num_classes))
    mask[range(n_samples), labels] = 0
    combined_features[:, 11:12] = predicts[range(n_samples), labels] - torch.max(mask*predicts, 1)

    states = torch.cat([data_features, model_features, combined_features], 1)
    return states


def evaluator(predicts, labels):
    criterion = nn.CrossEntropyLoss()
    num_correct = torch.sum(torch.max(predicts, 1)[1] == labels)
    num_samples = predicts.size(0)
    loss = criterion(predicts, labels)
    return {'num_correct': num_correct, 'num_samples':num_samples, 'loss':loss}