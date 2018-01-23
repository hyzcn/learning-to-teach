import torch.optim as optim


def get_optimizer(configs):
    optimizer = configs['optimizer']
    base_lr = configs['base_lr']
    model = configs['model']
    built_in = {'SGD': optim.SGD, 'Adam': optim.Adam}
    momentum = configs.get('momentum', 0)
    if momentum != 0:
        return built_in[optimizer](lr=base_lr, parameters=model.parameters(), momentum=momentum)
    else:
        return built_in[optimizer](lr=base_lr, parameters=model.parameters())