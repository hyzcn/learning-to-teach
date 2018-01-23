import torch
import torch.nn as nn


class TeacherNetwork(nn.Module):

    def __init__(self, configs):
        '''
        Descriptions:
            The teacher network is used for data scheduling, and it is a
            3-layer MLP, using tanh activation. (The same as it is in the L2T paper)
            The detailed architecture is: fc1(input_dim, 12), tanh, fc2(12, 1), sigmoid -> (0,1)
            The input feature consists of three parts: data feature, model feature and combined feature.
            e.g. for cifar-10 dataset:
                                    a). Data feature: 10-dim, one hot encoding.
                                    b). Model feature: 3-dim, [current_iter_num, avg_train_loss, best_val_loss].
                                        Note: All three signals are respectively divided by pre-defined maximum number
                                        to constrain their values in the interval [0,1].
                                    c). Combined feature: 12-dim, [P, -logP_y, P(y|x)-max_{y' \neq y}P(y'|x)]
                                        P: 10-dim, and the other two are 1-dim respectively.
            In section 7.3.2: the authors studied the importance of different features, surprisingly, the Model feature
            and combined feature are most important.

            Teaching Strategies: collect M samples before updating the base neural network.
        :param configs:
            1. input_dim: int (for cifar-10&mnist, d = 25 (10(data feature) + 3(model feature) + 12(combined feature)))
        '''
        super(TeacherNetwork, self).__init__()
        self.input_dim = configs.get('input_dim', 25)
        self.fc0 = nn.Linear(self.input_dim, 12) # 12 is selected by the paper.
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(12, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        scale = 0.01
        self.fc0.weight.data.uniform_(-scale, scale)
        self.fc0.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.weight.bias.fill_(2)
        # for not filtering too much data in the early stage
        # refer to section 7.2 for details.

    def forward(self, data, configs):
        '''
        :param data: tensor, [batch_size, input_dim]
        :param configs:
        :return:
        '''
        x = data['input']
        out = self.fc0(x)
        out = self.tanh(out)
        out = self.fc1(out)
        out = self.sigmoid(out)
        return out



