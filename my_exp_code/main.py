#!/usr/bin/env python

import argparse
import logging
import random
from contextlib import redirect_stdout, redirect_stderr

from mlflow import log_metric, log_param
from orion.client import report_results
from yaml import load

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='log to this file (otherwise log to screen)')
    parser.add_argument('--config', help='config file with hyper parameters - in yaml format')
    parser.add_argument('--hyper_param1', default='1')
    parser.add_argument('--hyper_param2', default='1')
    parser.add_argument('--saved_model', default='model.pt')
    parser.add_argument('--max_epoch', type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # redirect stdout and stderr to a log file - if provided (useful for orion on cluster)
    if args.log is not None:
        with open(args.log, 'w') as out:
            with redirect_stdout(out):
                with redirect_stderr(out):
                    run(args)
    else:
        run(args)


def do_training():
    return random.random()


def eval_on_dev():
    return random.random()


def save_model():
    # your model is safe..
    pass


def run(args):
    with open(args.config, 'r') as stream:
        hyper_params = load(stream)
    hp1 = hyper_params['hyper_param1']
    log_param("hyper_param1", hp1)
    hp2 = hyper_params['hyper_param2']
    log_param("hyper_param2", hp2)

    logger.info('hp1: {} / hp2: {}'.format(hp1, hp2))

    patience = 10
    not_improving_since = 0
    best_dev_metric = None
    for e in range(args.max_epoch):
        loss = do_training()
        dev_metric = eval_on_dev()

        log_metric("loss", loss, step=e)
        log_metric("dev_metric", dev_metric, step=e)

        if best_dev_metric is None or dev_metric > best_dev_metric:
            best_dev_metric = dev_metric
            not_improving_since = 0
            save_model()
        else:
            not_improving_since += 1

        logger.info('\ndone epoch {} => loss {} - dev metric {} (not improving'
                    ' since {} epoch)'.format(e, loss, dev_metric, not_improving_since))

        if not_improving_since >= patience:
            logger.info('done! best dev metric is {}'.format(best_dev_metric))
            break
    report_results([dict(
        name='dev_metric',
        type='objective',
        # note the minus - cause orion is always trying to minimize (cit. from the guide)
        value=-best_dev_metric)])


if __name__ == '__main__':
    main()
