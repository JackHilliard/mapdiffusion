
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import os.path as osp

import mmcv
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import DistEvalHook


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHookDiffuse(BaseDistEvalHook):

    def __init__(self, val_dataloader, coef, total_steps, *args, dynamic_intervals=None,  **kwargs):
        super(CustomDistEvalHookDiffuse, self).__init__(val_dataloader, *args, **kwargs)
        self.coef = coef
        self.total_steps = total_steps
        self.eta = float(kwargs['eval_diffusion_eta']) if 'eval_diffusion_eta' in kwargs.keys() else None
        self.sampling_timesteps = int(kwargs['eval_diffusion_sampling_timesteps']) if 'eval_diffusion_sampling_timesteps' in kwargs.keys() else None
        self.query_threshold = float(kwargs['eval_diffusion_query_threshold']) if 'eval_diffusion_query_threshold' in kwargs.keys() else None
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from ..apis.test import custom_multi_gpu_test_diffuse # to solve circlur  import

        results = custom_multi_gpu_test_diffuse(
            runner.model,
            self.dataloader,
            self.total_steps,
            self.coef,
            self.eta,
            self.sampling_timesteps,
            self.query_threshold,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)


class CustomEvalHookDiffuse(BaseEvalHook):
    """Single-GPU counterpart of :class:`CustomDistEvalHookDiffuse`.

    ``custom_train_detector_diffuse`` used to fall back to mmdet's stock
    ``EvalHook`` whenever ``distributed`` was False, but that class knows
    nothing about the diffusion parameters this model needs, so the
    ``eval_hook(val_dataloader, coef, total_steps, **eval_cfg)`` call landed
    ``coef`` in ``start`` and ``total_steps`` in ``interval`` and then died
    with ``TypeError: EvalHook.__init__() got multiple values for argument
    'interval'``. That made training with validation impossible under
    ``--launcher none``.

    Everything here mirrors the distributed hook except the test function
    and the absence of rank/BN-buffer handling.
    """

    def __init__(self, val_dataloader, coef, total_steps, *args,
                 dynamic_intervals=None, **kwargs):
        super(CustomEvalHookDiffuse, self).__init__(val_dataloader, *args,
                                                    **kwargs)
        self.coef = coef
        self.total_steps = total_steps
        self.eta = float(kwargs['eval_diffusion_eta']) if 'eval_diffusion_eta' in kwargs.keys() else None
        self.sampling_timesteps = int(kwargs['eval_diffusion_sampling_timesteps']) if 'eval_diffusion_sampling_timesteps' in kwargs.keys() else None
        self.query_threshold = float(kwargs['eval_diffusion_query_threshold']) if 'eval_diffusion_query_threshold' in kwargs.keys() else None
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from ..apis.test import custom_single_gpu_test_diffuse  # circular import

        results = custom_single_gpu_test_diffuse(
            runner.model,
            self.dataloader,
            self.total_steps,
            self.coef,
            self.eta,
            self.sampling_timesteps,
            self.query_threshold)
        print('\n')
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

        key_score = self.evaluate(runner, results)

        if self.save_best:
            self._save_ckpt(runner, key_score)

