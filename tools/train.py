# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, wrap_fp16_model
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from mmcv.utils import TORCH_VERSION, digit_version


def apply_epoch_schedule(cfg, num_train, logger=None):
    """Derive the IterBasedRunner's iteration counts from an epoch schedule.

    This repo trains with ``IterBasedRunner`` (its infinite batch sampler needs it), so
    epochs, evaluation and checkpoint intervals are iteration counts that depend on the
    train-set size and the batch size. A config may instead carry::

        epoch_schedule = dict(num_epochs=16, eval_every=2, ckpt_every=1,
                              effective_batch=16, warmup_samples=8000)

    and this function, called once the train set is built, sets with
    ``bs = data.samples_per_gpu * number of GPUs`` and ``cum = effective_batch // bs``::

        iters/epoch              = ceil(num_train / effective_batch) * cum
        runner.max_iters         = num_epochs * iters/epoch
        evaluation.interval      = eval_every * iters/epoch
        checkpoint_config.interval = ckpt_every * iters/epoch
        lr_config.warmup_iters   = ceil(warmup_samples / bs)
        optimizer_config         -> GradientCumulativeOptimizerHook(cumulative_iters=cum)
                                    when cum > 1 (grad_clip kept)

    so ``--cfg-options epoch_schedule.num_epochs=E data.samples_per_gpu=B`` keeps the
    schedule and the effective batch consistent. Configs without ``epoch_schedule`` are
    untouched. Returns the derived values (or None).
    """
    import math
    es = cfg.get('epoch_schedule', None)
    if not es:
        return None
    if cfg.runner['type'] != 'IterBasedRunner':
        raise ValueError('epoch_schedule is for IterBasedRunner configs')
    bs = int(cfg.data.samples_per_gpu) * max(1, len(cfg.gpu_ids))
    eff = int(es.get('effective_batch', bs))
    if eff % bs:
        raise ValueError(
            f'epoch_schedule.effective_batch={eff} is not a multiple of the batch per '
            f'iteration ({cfg.data.samples_per_gpu} per GPU x {len(cfg.gpu_ids)} GPU(s) = {bs})')
    cum = eff // bs
    ipe = int(math.ceil(num_train / float(eff))) * cum
    num_epochs = int(es['num_epochs'])
    cfg.runner.max_iters = num_epochs * ipe
    if cfg.get('evaluation', None) is not None:
        cfg.evaluation.interval = int(es.get('eval_every', 1)) * ipe
    if cfg.get('checkpoint_config', None) is not None:
        cfg.checkpoint_config.interval = int(es.get('ckpt_every', 1)) * ipe
    if es.get('warmup_samples', None) is not None and cfg.get('lr_config', None) is not None:
        cfg.lr_config.warmup_iters = max(1, int(math.ceil(es['warmup_samples'] / float(bs))))
    # gradient accumulation follows the batch actually used, whatever the config file wrote
    oc = dict(cfg.get('optimizer_config', None) or {})
    if cum > 1:
        oc['type'] = 'GradientCumulativeOptimizerHook'
        oc['cumulative_iters'] = cum
    elif oc.get('type', None) == 'GradientCumulativeOptimizerHook':
        oc.pop('type')
        oc.pop('cumulative_iters', None)
    cfg.optimizer_config = oc
    out = dict(num_train=int(num_train), batch_per_iter=bs, cumulative_iters=cum,
               effective_batch=eff, iters_per_epoch=ipe, max_iters=cfg.runner.max_iters,
               eval_interval=cfg.get('evaluation', {}).get('interval', None),
               ckpt_interval=(cfg.get('checkpoint_config', None) or {}).get('interval', None),
               warmup_iters=(cfg.get('lr_config', None) or {}).get('warmup_iters', None))
    msg = f'epoch_schedule {dict(es)} -> {out}'
    logger.info(msg) if logger is not None else print(msg)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description='Train MapDiffusion')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # torch>=2.0's torch.distributed.launch/torchrun passes the hyphenated
    # spelling, so accepting only '--local_rank' makes dist_train.sh fail
    # with "unrecognized arguments: --local-rank=0" before anything runs.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # The streaming BEV memory holds one slot per sample in the batch, so
    # model.streaming_cfg.batch_size must always equal data.samples_per_gpu.
    # The configs set both from one `batch_size` variable, but --cfg-options
    # merges after the config is evaluated, so overriding data.samples_per_gpu
    # alone (the natural thing to do from a job script) would leave the buffer
    # sized for the old batch. Derive it here instead of asking callers to
    # remember to set both.
    streaming_cfg = cfg.model.get('streaming_cfg') if 'model' in cfg else None
    if streaming_cfg is not None and 'batch_size' in streaming_cfg:
        samples_per_gpu = cfg.data.get('samples_per_gpu')
        if samples_per_gpu is not None and \
                streaming_cfg['batch_size'] != samples_per_gpu:
            print(f'[cfg] syncing model.streaming_cfg.batch_size '
                  f'{streaming_cfg["batch_size"]} -> {samples_per_gpu} to '
                  'match data.samples_per_gpu')
            streaming_cfg['batch_size'] = samples_per_gpu

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # import modules from plguin/xx, registry will be updated
    import sys
    sys.path.append(os.path.abspath('.'))    
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                def import_path(plugin_dir):
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)

                plugin_dirs = cfg.plugin_dir
                if not isinstance(plugin_dirs,list):
                    plugin_dirs = [plugin_dirs,]
                for plugin_dir in plugin_dirs:
                    import_path(plugin_dir)
                
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    model.init_weights()

    if cfg.get('SyncBN', False):
        import torch.nn as nn
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Using SyncBN")
        
    logger.info(f'Model:\n{model}')
    cfg.data.train.work_dir = cfg.work_dir
    cfg.data.val.work_dir = cfg.work_dir
    datasets = [build_dataset(cfg.data.train)]
    # epochs -> iterations, once the train-set size is known (no-op without
    # cfg.epoch_schedule); the dumped config is refreshed with the result
    if apply_epoch_schedule(cfg, len(datasets[0]), logger) is not None:
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
        meta['config'] = cfg.pretty_text
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=None,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES

    # compute noise coef
    total_steps = cfg.get("total_steps", None)
    print(f"Diffusion Total Steps: {total_steps}")
    scheduler = cfg.get("scheduler", None)
    print(f"Diffusion Noise Scheduler: {scheduler}")
    if total_steps is not None and scheduler is not None:
        from plugin.models.utils.coef import compute_ddpm_coef
        coef = compute_ddpm_coef(total_steps, scheduler)
    else:
        coef = None

    from plugin.core.apis import custom_train_model_diffuse
    custom_train_model_diffuse(
        model,
        coef,
        total_steps,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
