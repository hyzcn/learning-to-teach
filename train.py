from __future__ import print_function, division, absolute_import

import argparse

import torch
import torch.nn as nn

from core.student_network import StudentNetwork
from core.teacher_network import TeacherNetwork
from core.teacher_student import TeacherStudentModel

from misc.logger import create_logger
from misc.saver import Saver

from hparams.register import get_hparams

# ================= define global parameters ===============
logger = None
saver = None
evaluator = None

def make_global_parameters(hparams):
    global logger
    logger_configs = hparams.logger_configs
    logger = create_logger(logger_configs['output_path'], logger_configs['cfg_name'])

    global saver
    saver_configs = hparams.saver_configs
    saver = Saver(saver_configs['init_best_metric'], saver_configs['metric_name'], hparams, saver_configs['output_path'])`


def main(hparams):
    # teacher train set 45%
    # teacher dev set 5%
    # student train set 50%
    # student test set 100%





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Main Module')
    parser.add_argument('--hparams', type=str, help='choose hyper parameter set')

    args = parser.parse_args()
    extra_info = None
    hparams = get_hparams(args.hparams)(extra_info)

    make_global_parameters(hparams)
    main(hparams)

