# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging, subprocess, sys

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    try:
        GIT_BRANCH = subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD'])
    except:
        GIT_BRANCH = "Unknown"

    try:
        GIT_REVISION = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except:
        GIT_REVISION = "Unknown"

    logger.info('COMMAND: [ nohup python %s & ], GIT_REVISION: [%s] [%s]'
                % (' '.join(sys.argv), GIT_BRANCH.strip(), GIT_REVISION.strip()))

    return logger
