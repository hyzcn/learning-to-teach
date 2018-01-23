from .hparams import HParams
from .register import register
from core.models.resnet import ResNet34
from core.helper_functions import evaluator
import pickle

#TODO add experiment config
seed = 666


@register("cifar10_l2t")
def cifar10_l2t(extra_info):
    dataset = 'cifar10'
    splits = ['teacher_train', 'dev', '']
    teacher_configs = {
        'input_dim':25
    }
    student_configs = {
        'base_model': ResNet34(),
        'evaluator': evaluator
    }

    dataloader_configs = {
        'student':
            {
                'dataset':
            }
    }

    hparams = HParams( trainloader_info=trainloader_info,
                       valloader_info=valloader_info,
                       testloader_info=testloader_info,
                       model_info=model_info,
                       criterion_info=criterion_info,
                       optimizer_info=optimizer_info,
                       main_info=main_info,
                       seed=seed,
                       )
    return hparams