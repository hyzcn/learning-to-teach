from __future__ import print_function, division, absolute_import
import torch
import math
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

    _inputs = {'inputs':inputs, 'labels':labels}
    predicts, _ = student(_inputs, None) # predicts are logits
    predicts = nn.Softmax()(predicts)

    n_samples = inputs.size(0)
    data_features = to_var(torch.zeros(n_samples, num_classes))
    # import pdb
    # print ('0000000')
    # pdb.set_trace()
    data_features[range(n_samples), labels.data] = 1

    def sigmoid(x):
        return 1.0/(1.0 + math.exp(-x))

    model_features = to_var(torch.zeros(n_samples, 3))
    model_features[:, 0] = current_iter / max_iter # current iteration number
    model_features[:, 1] = 0 if len(train_loss_history) == 0 else sigmoid(sum(train_loss_history)/len(train_loss_history)) # averaged training loss
    model_features[:, 2] = 1 if len(val_loss_history) == 0 else sigmoid(min(val_loss_history))
    # import pdb
    # print('11111111')
    # pdb.set_trace()
    combined_features = to_var(torch.zeros(n_samples, 12))
    combined_features[:, :10] = predicts
    # import pdb
    # print('22222222')
    # pdb.set_trace()
    combined_features[:, 10:11] = -torch.log(predicts[range(n_samples), labels.data])
    # import pdb
    # print('33333333')
    # pdb.set_trace()
    mask = to_var(torch.ones(n_samples, num_classes))
    # import pdb
    # print('44444444')
    # pdb.set_trace()
    mask[range(n_samples), labels.data] = 0
    combined_features[:, 11:12] = predicts[range(n_samples), labels.data] - torch.max(mask*predicts, 1)[0]

    states = torch.cat([data_features, model_features, combined_features], 1)
    return states


def evaluator(predicts, labels):
    labels = labels.squeeze()
    criterion = nn.CrossEntropyLoss()
    num_correct = torch.sum(torch.max(predicts, 1)[1] == labels).cpu().data[0]
    num_samples = predicts.size(0)
    loss = criterion(predicts, labels)
    # print ('num_correct:', num_correct, 'num_samples:', num_samples)
    return {'num_correct': num_correct, 'num_samples':num_samples, 'loss':loss}