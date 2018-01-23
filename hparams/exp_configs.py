from .hparams import HParams
from .register import register
from core.models.resnet import ResNet34
from core.helper_functions import evaluator
import pickle

#TODO add experiment config
seed = 666


@register("cifar10_l2t")
def cifar10_l2t(extra_info):
    global seed
    dataset = 'cifar10'
    splits = ['teacher_train', 'student_train', 'dev', 'test']
    teacher_configs = {
        'input_dim':25
    }
    student_configs = {
        'base_model': ResNet34(),
        'evaluator': evaluator
    }

    dataloader = {
        'teacher_train':
            {
                'dataset': dataset,
                'split': splits[0]
            },
        'student_train':
            {
                'dataset': dataset,
                'split': splits[1]
            },
        'dev':
            {
                'dataset': dataset,
                'split': splits[2]
            },
        'test':
            {
                'dataset': dataset,
                'split': splits[3]
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