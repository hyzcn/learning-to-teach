from .hparams import HParams
from .register import register
from core.models.resnet import ResNet34
from core.helper_functions import evaluator
import torchvision.transforms as transforms
import pickle

#TODO add experiment config
seed = 666


@register("cifar10_l2t")
def cifar10_l2t(extra_info):
    global seed
    root = './'
    dataset = 'cifar10'
    splits = ['teacher_train', 'student_train', 'dev', 'test']
    teacher_configs = {
        'input_dim':25
    }
    student_configs = {
        'base_model': ResNet34(),
        'evaluator': evaluator
    }
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = {
        'teacher_train':
            {
                'dataset': dataset,
                'split': splits[0],
                'root': root,
                'transform':transform_train,
                'batch_size': 32,
                'shuffle': True
            },
        'student_train':
            {
                'dataset': dataset,
                'split': splits[1],
                'root': root,
                'transform': transform_train,
                'batch_size': 32,
                'shuffle': True
            },
        'dev':
            {
                'dataset': dataset,
                'split': splits[2],
                'root': root,
                'transform': transform_test,
                'batch_size': 64,
                'shuffle': False
            },
        'test':
            {
                'dataset': dataset,
                'split': splits[3],
                'root': root,
                'transform': transform_test,
                'batch_size': 64,
                'shuffle': False
            }
    }
    models = {
        'teacher_configs': teacher_configs,
        'student_configs': student_configs
    }

    optimizer = {
        'teacher_configs':
            {
                'base_lr': 1e-4,
                'optimizer': 'Adam',
            },
        'student_configs':
            {
                'base_lr': 0.1,
                'optimizer': 'SGD',
                'momentum': 0.9
            }
    }

    optional = dict()
    logger_configs = {
        'output_path': './log',
        'cfg_name': '%s-l2t'%dataset
    }

    hparams = HParams(
        dataloader=dataloader,
        models=models,
        optimizer=optimizer,
        optional=optional,
        logger_configs=logger_configs,
        seed=seed
    )
    return hparams