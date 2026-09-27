# -*- coding: utf-8 -*-

import os
# Reduce CUDA caching-allocator fragmentation so more of the 16 GB is usable
# (must be set before torch initializes the allocator).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time

import torch
import argparse
import os
import numpy as np
import warnings
# warnings.filterwarnings("ignore")
from utility.logger import *
from comet_ml import Experiment

import open3d as o3d
from Dataset.dataset_coupled import DatasetCoupled,DatasetCoupled_slice_wise
from utility.read_confs import *
device="cuda" if torch.cuda.is_available() else "cpu"
from utility import misc
from models.AdaPoinTr import  AdaPoinTr
import cpuinfo
from utility.AverageMeter import AverageMeter
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
print("CPU:", cpuinfo.get_cpu_info()['brand_raw'])
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())
from pathlib import Path
from models.PoseFromRebuildFeature import *
from extensions.chamfer_dist import ChamferDistanceL1,ChamferDistanceL1_one_side
from utility.converter import *

import random
import contextlib
def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility.

    Args:
        seed: Random seed.
        deterministic: If True, make PyTorch/CUDA behavior as deterministic as possible.
                       This may reduce performance and may raise errors if a deterministic
                       implementation is unavailable for some ops.
    """
    # 1) Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 2) PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # 3) Reproducibility / determinism flags
    if deterministic:
        # cuDNN / matmul determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enforce deterministic algorithms where possible
        torch.use_deterministic_algorithms(True)

        # Needed for some CUDA ops (cublas) to be deterministic
        # (set before running; safe to set here as well)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # Faster, but non-deterministic
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        # Allow TF32 on Ampere+ (RTX 5080): sizeable matmul/conv speedup at no
        # extra memory, with negligible precision loss for this regression task.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@contextlib.contextmanager
def fixed_seed(seed):
    """Seed NumPy and PyTorch inside the block (or decorated function), then restore the
    previous RNG states, so the caller's random stream is left untouched."""
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class Runner_ShapeCompletion:
    def __init__(self,conf,experiment):

        self.experiment=experiment
        self.conf=conf
        self.model_shapeCompletion = AdaPoinTr(conf['model_shape_completion']).to(device)

        self.optimizer=self.build_optimizer(self.model_shapeCompletion,conf.train.optimizer)
        self.model_PoseFromRebuildFeature = PoseFromRebuildFeature(in_dim=384, hidden_dim=self.conf.model_pose_estimator.hidden_dim,
                                                                   pooling='mean',dropout=self.conf.model_pose_estimator.dropout).to(device)
        self.optimizer_pose_estimator = self.build_optimizer(self.model_PoseFromRebuildFeature,conf.train.optimizer_pose_estimator)
        self.epoch=0

        # ---- Joint-training settings -------------------------------------
        # Shape completion and registration are trained jointly from the start
        # (no warmup). A single combined loss drives both modules, with the
        # registration term dynamically weighted each step so that its value
        # stays `reg_loss_ratio` x the completion loss (default 0.1 => the
        # registration loss is 10x smaller), guaranteeing shape completion
        # dominates the optimization.
        self.reg_loss_ratio = float(self.conf.train.get('reg_loss_ratio', 0.1))


        self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'],f"{conf['dataset']['specimen_id']}_{conf['dataset']['anatomy']}")
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = DatasetCoupled_slice_wise(conf['dataset'])
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.conf.dataset.bs, shuffle=True, drop_last=True,
                                                 num_workers=4, worker_init_fn=worker_init_fn, persistent_workers=True,
                                                 multiprocessing_context="spawn", pin_memory=True, prefetch_factor=4)




        self.loss_func_registration=ChamferDistanceL1_one_side()
        log_name = "ShapeCompletion"
        self.logger = get_logger(log_name)
        if 'ckp_file_path' in conf.keys():
            self.load_checkpoint(conf.ckp_file_path)
        self.scheduler = self.build_scheduler(self.model_shapeCompletion, self.optimizer, self.conf.train.scheduler,
                                              self.conf.train.bnmscheduler, last_epoch=self.epoch - 1)

        self.scheduler_pose_estimator = self.build_scheduler(self.model_PoseFromRebuildFeature, self.optimizer_pose_estimator, self.conf.train.scheduler_pose_estimator,
                                              self.conf.train.bnmscheduler_pose_estimator, last_epoch=self.epoch - 1)

    def build_optimizer(self,model, opti_config):
        if opti_config.type == 'AdamW':
            def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
                decay = []
                no_decay = []
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue  # frozen weights
                    if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                        no_decay.append(param)
                    else:
                        decay.append(param)
                return [
                    {'params': no_decay, 'weight_decay': 0.},
                    {'params': decay, 'weight_decay': weight_decay}]

            param_groups = add_weight_decay(model, weight_decay=opti_config.kwargs.weight_decay)
            optimizer = torch.optim.AdamW(param_groups, **opti_config.kwargs)
        elif opti_config.type == 'Adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **opti_config.kwargs)
        elif opti_config.type == 'SGD':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), **opti_config.kwargs)
        else:
            raise NotImplementedError()

        return optimizer

    def build_scheduler(self,base_model, optimizer, scheduler_config,bnsche_config, last_epoch=-1):
        if scheduler_config.type == 'LambdaLR':
            scheduler = misc.build_lambda_sche(optimizer, scheduler_config.kwargs, last_epoch=last_epoch)  # misc.py
        elif scheduler_config.type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **scheduler_config.kwargs)
        else:
            raise NotImplementedError()

        if bnsche_config.type == 'Lambda':
            bnscheduler = misc.build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]

        return scheduler

    def checkDataset(self):
        for idx, gt in enumerate(self.dataloader):
            gt = gt.to("cuda")
            partial, _ = misc.seprate_point_cloud(gt, gt.shape[1],
                                                  [int(gt.shape[1] * 1 / 4), int(gt.shape[1] * 3 / 4)],
                                                  fixed_points=None)
            ret = self.model_shapeCompletion(partial)
            for i in range(5):
                gt_o3d = o3d.geometry.PointCloud()
                gt_o3d.points = o3d.utility.Vector3dVector(gt[i].cpu().numpy())
                partial_o3d = o3d.geometry.PointCloud()
                partial_o3d.points = o3d.utility.Vector3dVector(partial[i].cpu().numpy())
                o3d.visualization.draw_geometries([gt_o3d.paint_uniform_color([1, 0, 0])])
                o3d.visualization.draw_geometries([partial_o3d.paint_uniform_color([0, 1, 0])])
                o3d.visualization.draw_geometries(
                    [gt_o3d.paint_uniform_color([1, 0, 0]), partial_o3d.paint_uniform_color([0, 1, 0])])

    def save_checkpoint(self,metrics, best_metrics,prefix):
        ckp_folder=os.path.join(self.conf.general.base_exp_dir,f"{self.conf.dataset.specimen_id}_{self.conf.dataset.anatomy}","checkpoints","ShapeCompletion")
        os.makedirs(ckp_folder,exist_ok=True)
        ckp_file=os.path.join(ckp_folder, prefix + '.pth')
        torch.save({
            'base_model': self.model_shapeCompletion.state_dict() ,
            'model_PoseFromRebuildFeature': self.model_PoseFromRebuildFeature.state_dict() ,
            'optimizer': self.optimizer.state_dict(),
            'optimizer_model_PoseFromRebuildFeature': self.optimizer_pose_estimator.state_dict(),
            'epoch': self.epoch,
            'metrics': metrics,
            'best_metrics': best_metrics,
        }, ckp_file)
        print_log(f"Save checkpoint at {ckp_file}", logger=self.logger)

    def load_checkpoint(self,ckpt_path):
        if not os.path.exists(ckpt_path):
            print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger=self.logger)
            return 0, 0
        print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger=self.logger)

        # load state dict
        state_dict = torch.load(ckpt_path, map_location=device,weights_only=False)
        # parameter resume of base model
        # if args.local_rank == 0:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
        self.model_shapeCompletion.load_state_dict(base_ckpt)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.model_PoseFromRebuildFeature.load_state_dict({k.replace("module.", ""): v for k, v in state_dict['model_PoseFromRebuildFeature'].items()})
        self.optimizer_pose_estimator.load_state_dict(state_dict['optimizer_model_PoseFromRebuildFeature'])
        # parameter
        self.epoch = state_dict['epoch'] + 1
        # best_metrics = state_dict['best_metrics']
        # if not isinstance(best_metrics, dict):
        #     best_metrics = best_metrics.state_dict()
        # print(best_metrics)

        print_log(f'[RESUME INFO] resume ckpts @ {self.epoch - 1} epoch)',
                  logger=self.logger)

    def _registration_loss(self, ret, rebuild_feature, rebuild_feature_gt_untransformed,
                           mean_diff, gt_untransformed, T_gt, bs_reg, epoch, partial=None):
        """Per-batch registration loss + a transform for logging/metrics.

        `partial` is the raw partial input cloud; the base ignores it, but the
        DiffICP variant can use it as the ICP source (see diff_icp.icp_src).

        Base behavior: split-gradient coarse-only pose regression -- the rotation
        term uses NON-detached features (couples into the encoder, generalizes
        sim->real), the translation term uses DETACHED features (the pose head
        learns t but cannot reshape the encoder to memorize the simulated centroid).
        Subclasses (e.g. DiffICP) override this to add a refinement term.
        Returns (loss_registration, T_estimated).
        """
        self.model_PoseFromRebuildFeature.train()

        # Rotation: couples into the encoder (features NOT detached).
        T_est_coupled = self.model_PoseFromRebuildFeature(
            rebuild_feature[:bs_reg],
            rebuild_feature_gt_untransformed[:bs_reg],
            mean_diff[:bs_reg])
        loss_rot = rte_rre_loss_batch(
            T_est_coupled, T_gt[:bs_reg],
            w_rte=0.0, w_rre=1.0, translation_scale=self.dataset.shape_scale)

        # Translation: detached so it cannot sculpt the encoder.
        T_est_detached = self.model_PoseFromRebuildFeature(
            rebuild_feature[:bs_reg].detach(),
            rebuild_feature_gt_untransformed[:bs_reg],
            mean_diff[:bs_reg].detach())
        loss_trans = rte_rre_loss_batch(
            T_est_detached, T_gt[:bs_reg],
            w_rte=1.0, w_rre=0.0, translation_scale=self.dataset.shape_scale)

        loss_registration = loss_rot + loss_trans
        # Optional refined transform, logged as RTE_ICP/RRE_ICP. None for the base
        # (coarse-only); DiffICP variants stash the ICP-refined transform here.
        self._reg_T_icp = None
        return loss_registration, T_est_detached

    def _eval_pose(self, rebuild_feature, rebuild_feature_gt_untransformed,
                   mean_diff, outputs, gt, B, partial=None):
        """Pose estimate used at validation. Base: the coarse MLP output.
        Subclasses (e.g. DiffICP) override to refine it and stash _eval_T_icp.
        `partial` is the raw input cloud, usable as the ICP source by DiffICP."""
        self._eval_T_icp = None
        return self.model_PoseFromRebuildFeature(
            rebuild_feature.detach(),
            rebuild_feature_gt_untransformed.repeat((B, 1)).detach(),
            mean_diff.detach())

    def train(self):


        # from IPython import embed; embed()

        # parameter setting
        best_metrics = 1e5
        metrics = None

        # --- Optional one-shot profiler: set PROFILE_STEPS=N to profile N real
        #     training steps (after a short warmup), print an op-level breakdown,
        #     and exit. Used to locate the speed bottleneck. ---
        _profile_steps = int(os.environ.get("PROFILE_STEPS", "0"))
        _prof = None
        _prof_seen = 0
        _prof_wait, _prof_warmup = 3, 3
        _prof_active_t0 = None
        if _profile_steps > 0:
            from torch.profiler import profile as _tprofile, ProfilerActivity, schedule as _tsched
            _prof = _tprofile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=_tsched(wait=_prof_wait, warmup=_prof_warmup, active=_profile_steps, repeat=1),
                record_shapes=True, with_stack=False, profile_memory=True)
            _prof.start()
            print_log(f"[PROFILE] on: wait={_prof_wait} warmup={_prof_warmup} active={_profile_steps}", logger=self.logger)

        self.model_shapeCompletion.zero_grad()
        for epoch in range(self.epoch, self.conf.train.max_epoch + 1):
            self.epoch=epoch
            self.model_shapeCompletion.train()
            self.model_PoseFromRebuildFeature.train()
            # Start each epoch with clean gradients on both models so the
            # warmup -> joint transition never accumulates stale gradients.
            self.model_shapeCompletion.zero_grad()
            self.model_PoseFromRebuildFeature.zero_grad()

            epoch_start_time = time.time()
            batch_start_time = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter(['SparseLoss', 'DenseLoss'])
            losses_pose_estimation=AverageMeter()
            RTE_average_meter=AverageMeter()
            RRE_averge_meter=AverageMeter()
            RTE_ICP_average_meter=AverageMeter()   # refined (DiffICP) metrics, if any
            RRE_ICP_average_meter=AverageMeter()
            epoch_has_icp=False

            num_iter = 0

            # GT point cloud is constant — cache its features once per epoch so
            # we don't run a second full forward pass every batch.
            with torch.no_grad():
                _gt_fps = misc.fps(
                    self.dataset.pcd_gt.unsqueeze(0).to(device).float(), self.dataset.npoints_input)
                _, rebuild_feature_gt_untransformed_cached = self.model_shapeCompletion(_gt_fps)

            n_batches = len(self.dataloader)
            for batch_idx, (gt,partial,gt_untransformed,Transformation_matrix_disturbance_inv) in enumerate(self.dataloader):
                data_time.update(time.time() - batch_start_time)
                gt = gt.to(device, non_blocking=True)
                partial=partial.to(device, non_blocking=True)
                gt_untransformed=gt_untransformed.to(device, non_blocking=True)
                Transformation_matrix_disturbance_inv=Transformation_matrix_disturbance_inv.to(device, non_blocking=True)
                ret,rebuild_feature = self.model_shapeCompletion(partial)
                rebuild_feature_gt_untransformed = rebuild_feature_gt_untransformed_cached.expand(gt.shape[0], -1)
                mean_diff=ret[3].mean(dim=1)-gt_untransformed.mean(dim=1)


                num_iter += 1

                sparse_loss, dense_loss = self.model_shapeCompletion.get_loss(ret, gt, epoch)
                loss_completion = sparse_loss + dense_loss

                # Registration loss + a transform for logging. Factored into
                # _registration_loss so variants (e.g. DiffICP refinement) can
                # override it without touching the training loop. Base behavior is
                # the split-gradient coarse-only pose regression.
                bs_reg = self.conf.dataset.bs  # registration uses the full batch
                loss_registration, T_estimated = self._registration_loss(
                    ret, rebuild_feature, rebuild_feature_gt_untransformed, mean_diff,
                    gt_untransformed, Transformation_matrix_disturbance_inv, bs_reg, epoch,
                    partial=partial)

                # Dynamic, DETACHED weight: makes the weighted registration term
                # equal to reg_loss_ratio x the completion loss in value (default
                # 0.1 => 10x smaller), so shape completion always dominates.
                w_reg = self.reg_loss_ratio * loss_completion.detach() / (loss_registration.detach() + 1e-8)
                total_loss = loss_completion + w_reg * loss_registration
                # Normalize by accumulation steps so the effective batch
                # (bs * step_per_update) emulates a true larger-batch average.
                (total_loss / self.conf.train.step_per_update).backward()

                # Joint optimizer step (supports gradient accumulation via
                # step_per_update): both models are stepped from the single
                # combined backward.
                if num_iter == self.conf.train.step_per_update:
                    torch.nn.utils.clip_grad_norm_(self.model_shapeCompletion.parameters(),
                                                   getattr(self.conf, 'grad_norm_clip', 10), norm_type=2)
                    num_iter = 0
                    self.optimizer.step()
                    self.model_shapeCompletion.zero_grad()
                    torch.nn.utils.clip_grad_norm_(self.model_PoseFromRebuildFeature.parameters(),
                                                   getattr(self.conf, 'grad_norm_clip', 10), norm_type=2)
                    self.optimizer_pose_estimator.step()
                    self.model_PoseFromRebuildFeature.zero_grad()

                n_itr = epoch * n_batches + batch_idx

                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()

                if _prof is not None:
                    _prof.step()
                    _prof_seen += 1
                    if _prof_seen == _prof_wait + _prof_warmup:
                        torch.cuda.synchronize()
                        _prof_active_t0 = time.time()
                    if _prof_seen >= _prof_wait + _prof_warmup + _profile_steps:
                        torch.cuda.synchronize()
                        _active_wall = time.time() - _prof_active_t0
                        _prof.stop()
                        ka = _prof.key_averages()
                        print("\n[PROFILE] ===== Top ops by SELF CUDA time (where the GPU spends time) =====")
                        print(ka.table(sort_by="self_cuda_time_total", row_limit=25))
                        print("\n[PROFILE] ===== Top ops by SELF CPU time (launch/python overhead) =====")
                        print(ka.table(sort_by="self_cpu_time_total", row_limit=20))
                        _ms = _active_wall / _profile_steps * 1000
                        print(f"\n[PROFILE] active window: {_active_wall:.3f}s / {_profile_steps} steps = {_ms:.1f} ms/step")
                        print("[PROFILE] If summed Self CUDA time << wall time, the loop is launch/sync-bound (small batch).")
                        import sys; sys.exit(0)

                # Optional clean early-stop for perf measurement (MAX_STEPS=N).
                _max_steps = int(os.environ.get("MAX_STEPS", "0"))
                if _max_steps > 0 and batch_idx + 1 >= _max_steps:
                    print(f"[MAXSTEPS] reached {batch_idx + 1} steps; last BatchTime={batch_time.val():.4f}s avg(last)={batch_time.avg():.4f}s")
                    import sys; sys.exit(0)

                if batch_idx % 10 == 0:
                    # All .item() and .cpu().numpy() calls deferred here to avoid per-batch GPU sync
                    losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
                    losses_pose_estimation.update(loss_registration.item())
                    rte, rre = compute_mean_RTE_RRE_batch(
                        Transformation_matrix_disturbance_inv[:bs_reg].cpu().numpy(),
                        T_estimated.detach().cpu().numpy(), self.dataset.shape_scale.item())
                    RTE_average_meter.update(rte)
                    RRE_averge_meter.update(rre)
                    # ICP-refined transform (set by _registration_loss in DiffICP
                    # variants); logged separately as RTE_ICP / RRE_ICP.
                    T_icp = getattr(self, '_reg_T_icp', None)
                    if T_icp is not None:
                        rte_icp, rre_icp = compute_mean_RTE_RRE_batch(
                            Transformation_matrix_disturbance_inv[:bs_reg].cpu().numpy(),
                            T_icp.detach().cpu().numpy(), self.dataset.shape_scale.item())
                        RTE_ICP_average_meter.update(rte_icp)
                        RRE_ICP_average_meter.update(rre_icp)
                        epoch_has_icp = True
                    # Per-batch line reports THIS batch's mean (not meter.val(), which
                    # is the last sample); the meters keep the full per-sample arrays so
                    # the epoch .avg() stays an exact per-sample mean.
                    _icp_str = ((' | RTE_ICP = %.2f, RRE_ICP = %.2f' %
                                 (rte_icp.mean(), rre_icp.mean()))
                                if T_icp is not None else '')
                    if epoch >= 0:
                        print_log(
                            '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s loss_reg = %s, RTE = %.2f, RRE = %.2f%s, lr = %.6f, lr_pose = %.6f' %
                            (epoch, self.conf.train.max_epoch, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                             ['%.4f' % l for l in losses.val()],
                             losses_pose_estimation.val(), rte.mean(), rre.mean(), _icp_str,
                             self.optimizer.param_groups[0]['lr'],
                             self.optimizer_pose_estimator.param_groups[0]['lr']),
                            logger=self.logger)
                    else:
                        print_log(
                            '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, self.conf.train.max_epoch, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                             ['%.4f' % l for l in losses.val()], self.optimizer.param_groups[0]['lr']), logger=self.logger)



                if self.conf.train.scheduler.type == 'GradualWarmup':
                    if n_itr < self.conf.train.scheduler.kwargs_2.total_epoch:
                        self.scheduler.step()
                        self.scheduler_pose_estimator.step()

            if isinstance(self.scheduler, list):
                for item in self.scheduler:
                    item.step()
                for item in self.scheduler_pose_estimator:
                    item.step()
            else:
                self.scheduler.step()
                self.scheduler_pose_estimator.step()
            epoch_end_time = time.time()

            # if train_writer is not None:
            #     train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            #     train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
            _icp_epoch = ((' RTE_ICP = %.2f, RRE_ICP = %.2f,' %
                           (RTE_ICP_average_meter.avg(), RRE_ICP_average_meter.avg())) if epoch_has_icp else '')
            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s, loss_reg = %s, RTE = %.2f, RRE = %.2f,%s' %
                      (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                       losses_pose_estimation.avg(), RTE_average_meter.avg(), RRE_averge_meter.avg(), _icp_epoch), logger=self.logger)
            if self.experiment:
                _metrics = {'Sparse Loss': losses.avg()[0], 'Dense Loss': losses.avg()[1], 'Reg Loss':losses_pose_estimation.avg(),
                            'RTE':RTE_average_meter.avg(),'RRE':RRE_averge_meter.avg()}
                if epoch_has_icp:
                    _metrics['RTE_ICP'] = RTE_ICP_average_meter.avg()
                    _metrics['RRE_ICP'] = RRE_ICP_average_meter.avg()
                self.experiment.log_metrics(_metrics, epoch=epoch)

            if epoch % 10 == 0 and self.experiment:
                metrics = self.validate()
                self.experiment.log_metrics({'Validation CD': metrics}, epoch=epoch)

                # Save ckeckpoints
                # if  metrics<best_metrics:
                #     best_metrics = metrics
                #     self.save_checkpoint(metrics, best_metrics, prefix='ckpt-best')
            # Periodic checkpoints (always-joint training; no warmup boundary).
            if epoch >= 100 and epoch % 50 == 0:
                self.save_checkpoint(metrics, best_metrics, prefix=f'ckpt-{epoch}')


    @fixed_seed(1234)   # same disturbances at every validation, comparable across epochs
    def validate(self):
        self.model_PoseFromRebuildFeature.eval()
        #self.model_shapeCompletion.eval()
        # normalize intraoperative data, mean based on intra, but scale based on CT
        # intra_data_raw = self.dataset.intra_pcd_merged_mm.cpu()  # B*N*3
        intra_data_raw = self.dataset.intra_pcds_mm  # B*N*3
        intra_data_raw = (intra_data_raw - self.dataset.shape_center) / self.dataset.shape_scale
        intra_data_raw = intra_data_raw.float().to(device)
        # intra_data_raw = intra_data_raw.repeat((3, 1, 1))

        gt = self.dataset.pcd_gt
        gt = gt.to(device).unsqueeze(0).float()
        gt = misc.fps(gt, self.dataset.npoints_input)
        gt_o3d = o3d.geometry.PointCloud()
        gt_o3d.points = o3d.utility.Vector3dVector(gt[0].cpu().numpy())
        with torch.no_grad():
            _, rebuild_feature_gt_untransformed = self.model_shapeCompletion(gt)

        AverageMeter_CD = AverageMeter()

        AverageMeter_HD95 = AverageMeter()
        AverageMeter_RTE = AverageMeter()
        AverageMeter_RRE = AverageMeter()
        AverageMeter_RTE_ICP = AverageMeter()
        AverageMeter_RRE_ICP = AverageMeter()
        _val_has_icp = False



        for run_idx in range(20):
            idx = np.random.permutation(intra_data_raw.shape[0])[:4]
            B = len(idx)

            intra_data=torch.clone(intra_data_raw[idx])
            Transformation_matrix_disturbance = np.zeros([B, 4, 4])
            Transformation_matrix_disturbance_inv = np.zeros([B, 4, 4])
            for idx in range(B):
                # Random axis-angle disturbance: +-90 deg about a random axis,
                # translation along that same axis in [-0.5, 0.5] (matches training).
                Transformation_matrix_disturbance[idx] = random_axis_angle_transform(
                    angle_deg_bound=self.dataset.max_rotation_deg, t_bound=self.dataset.max_translation)
                Transformation_matrix_disturbance_inv[idx] = invert_transformation_matrix(
                    Transformation_matrix_disturbance[idx])

            Transformation_matrix_disturbance = torch.tensor(Transformation_matrix_disturbance).float().to(device)
            Transformation_matrix_disturbance_inv = torch.tensor(Transformation_matrix_disturbance_inv).float().to(device)

            intra_data_transformed = (torch.matmul(Transformation_matrix_disturbance[:, :3, :3],
                                                   intra_data.transpose(1, 2)).transpose(1, 2) +
                                      Transformation_matrix_disturbance[:, :3, 3].unsqueeze(1))  # B*N*3
            centroid = torch.mean(intra_data_transformed, dim=1, keepdim=True)
            intra_data_transformed = (intra_data_transformed - centroid).contiguous()
            T_cenerized = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).float().to(device)
            T_cenerized[:, :3, 3] = -centroid.squeeze(1)
            Transformation_matrix_disturbance_inv = torch.matmul(Transformation_matrix_disturbance_inv,
                                                                 torch.linalg.inv(T_cenerized))


            with torch.no_grad():
                outputs, rebuild_feature = self.model_shapeCompletion(intra_data_transformed)

                mean_diff = outputs[3].mean(dim=1) - gt.mean(dim=1)
                T_estimated = self._eval_pose(rebuild_feature, rebuild_feature_gt_untransformed,
                                              mean_diff, outputs, gt, B, partial=intra_data_transformed)
                rte, rre = compute_mean_RTE_RRE_batch(Transformation_matrix_disturbance_inv.cpu().numpy(),
                                                      T_estimated.detach().cpu().numpy(), self.dataset.shape_scale.item())
                AverageMeter_RTE.update(rte)
                AverageMeter_RRE.update(rre)
                T_icp = getattr(self, '_eval_T_icp', None)
                if T_icp is not None:
                    rte_icp, rre_icp = compute_mean_RTE_RRE_batch(Transformation_matrix_disturbance_inv.cpu().numpy(),
                                                                  T_icp.detach().cpu().numpy(), self.dataset.shape_scale.item())
                    AverageMeter_RTE_ICP.update(rte_icp)
                    AverageMeter_RRE_ICP.update(rre_icp)
                    _val_has_icp = True



            for idx in range(outputs[3].shape[0]):
                input_pcd = intra_data_transformed[idx]
                input_pcd = input_pcd.cpu().detach().numpy()
                input_pcd_o3d = o3d.geometry.PointCloud()
                input_pcd_o3d.points = o3d.utility.Vector3dVector(input_pcd)

                pcd_completed = outputs[3][idx]
                pcd_completed = torch.matmul(Transformation_matrix_disturbance_inv[idx, :3, :3],
                                             pcd_completed.transpose(0, 1)).transpose(0, 1) + \
                                Transformation_matrix_disturbance_inv[idx, :3, 3].unsqueeze(0)

                # transform back, CT scale, and intraoperative centroid
                pcd_completed = pcd_completed.cpu().detach().numpy()

                pcd_completed_o3d = o3d.geometry.PointCloud()
                pcd_completed_o3d.points = o3d.utility.Vector3dVector(pcd_completed)
                # o3d.visualization.draw_geometries([input_pcd_o3d.paint_uniform_color((0,0,1)),pcd_completed_o3d.paint_uniform_color((0,1,0))])
                # o3d.visualization.draw_geometries([gt_o3d.paint_uniform_color((1,0,0)),pcd_completed_o3d.paint_uniform_color((0,1,0))])
                dists_gt_2_pred = np.asarray(gt_o3d.compute_point_cloud_distance(pcd_completed_o3d))
                dists_pred_2_gt = np.asarray(pcd_completed_o3d.compute_point_cloud_distance(gt_o3d))
                CD = (0.5 * (dists_gt_2_pred.mean() + dists_pred_2_gt.mean())) * self.dataset.shape_scale
                HD95 = (0.5 * np.percentile(dists_gt_2_pred, 95) + 0.5 * np.percentile(dists_pred_2_gt,
                                                                                       95)) * self.dataset.shape_scale
                AverageMeter_CD.update(CD)
                AverageMeter_HD95.update(HD95)

        _icp_val = (('  RTE_ICP = %.2f, RRE_ICP = %.2f' %
                     (AverageMeter_RTE_ICP.avg(), AverageMeter_RRE_ICP.avg())) if _val_has_icp else '')
        print_log('[Validation] EPOCH: %d  CD = %.2f, HD95 = %.2f, RTE = %.2f, RRE = %.2f%s' %
                  (self.epoch, AverageMeter_CD.avg(), AverageMeter_HD95.avg(), AverageMeter_RTE.avg(), AverageMeter_RRE.avg(), _icp_val), logger=self.logger)
        if self.experiment:
            _val_metrics = {'Validation RTE': AverageMeter_RTE.avg(), 'Validation RRE': AverageMeter_RRE.avg()}
            if _val_has_icp:
                _val_metrics['Validation RTE_ICP'] = AverageMeter_RTE_ICP.avg()
                _val_metrics['Validation RRE_ICP'] = AverageMeter_RRE_ICP.avg()
            self.experiment.log_metrics(_val_metrics, epoch=self.epoch)
        file_path = Path("./quantitative_evaluation_shape_completion/slice_wise.txt")

        # with file_path.open("a", encoding="utf-8") as f:
        #     f.write(f"specimen_id: {self.conf.dataset.specimen_id}, anatomy: {self.conf.dataset.anatomy}, "
        #             f"CD_mean: {AverageMeter_CD.avg():.3f}, CD_std: {AverageMeter_CD.std():.3f}, "
        #             f"HD95_mean: {AverageMeter_HD95.avg():.3f}, HD95_std: {AverageMeter_HD95.std():.3f}\n")
        self._last_val_metrics = {
            'CD': AverageMeter_CD.avg(), 'HD95': AverageMeter_HD95.avg(),
            'RTE': AverageMeter_RTE.avg(), 'RRE': AverageMeter_RRE.avg(),
            'RTE_ICP': (AverageMeter_RTE_ICP.avg() if _val_has_icp else None),
            'RRE_ICP': (AverageMeter_RRE_ICP.avg() if _val_has_icp else None)}
        return AverageMeter_CD.avg()
        # return intra_data


    def validate_merged(self):
        self.model_PoseFromRebuildFeature.eval()
        #self.model_shapeCompletion.eval()
        # normalize intraoperative data, mean based on intra, but scale based on CT
        intra_data_raw = self.dataset.intra_pcd_merged_mm.cpu()  # B*N*3
        if len(intra_data_raw.shape)==2:
            intra_data_raw=intra_data_raw.unsqueeze(0)
        intra_data_raw = (intra_data_raw - self.dataset.shape_center) / self.dataset.shape_scale
        intra_data_raw = intra_data_raw.float().to(device)
        intra_data_raw = intra_data_raw.repeat((3, 1, 1))

        gt = self.dataset.pcd_gt
        gt = gt.to(device).unsqueeze(0).float()
        gt = misc.fps(gt, self.dataset.npoints_input)
        gt_o3d = o3d.geometry.PointCloud()
        gt_o3d.points = o3d.utility.Vector3dVector(gt[0].cpu().numpy())
        with torch.no_grad():
            _, rebuild_feature_gt_untransformed = self.model_shapeCompletion(gt)

        AverageMeter_CD = AverageMeter()

        AverageMeter_HD95 = AverageMeter()
        AverageMeter_RTE = AverageMeter()
        AverageMeter_RRE = AverageMeter()
        AverageMeter_RTE_ICP = AverageMeter()
        AverageMeter_RRE_ICP = AverageMeter()
        _val_has_icp = False



        for run_idx in range(100):
            idx = np.random.permutation(intra_data_raw.shape[0])[:4]
            B = len(idx)

            intra_data=torch.clone(intra_data_raw[idx])
            Transformation_matrix_disturbance = np.zeros([B, 4, 4])
            Transformation_matrix_disturbance_inv = np.zeros([B, 4, 4])
            for idx in range(B):
                # Random axis-angle disturbance: +-90 deg about a random axis,
                # translation along that same axis in [-0.5, 0.5] (matches training).
                Transformation_matrix_disturbance[idx] = random_axis_angle_transform(
                    angle_deg_bound=self.dataset.max_rotation_deg, t_bound=self.dataset.max_translation)
                Transformation_matrix_disturbance_inv[idx] = invert_transformation_matrix(
                    Transformation_matrix_disturbance[idx])

            Transformation_matrix_disturbance = torch.tensor(Transformation_matrix_disturbance).float().to(device)
            Transformation_matrix_disturbance_inv = torch.tensor(Transformation_matrix_disturbance_inv).float().to(device)

            intra_data_transformed = (torch.matmul(Transformation_matrix_disturbance[:, :3, :3],
                                                   intra_data.transpose(1, 2)).transpose(1, 2) +
                                      Transformation_matrix_disturbance[:, :3, 3].unsqueeze(1))  # B*N*3
            centroid = torch.mean(intra_data_transformed, dim=1, keepdim=True)
            intra_data_transformed = (intra_data_transformed - centroid).contiguous()
            T_cenerized = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).float().to(device)
            T_cenerized[:, :3, 3] = -centroid.squeeze(1)
            Transformation_matrix_disturbance_inv = torch.matmul(Transformation_matrix_disturbance_inv,
                                                                 torch.linalg.inv(T_cenerized))


            with torch.no_grad():
                outputs, rebuild_feature = self.model_shapeCompletion(intra_data_transformed)

                mean_diff = outputs[3].mean(dim=1) - gt.mean(dim=1)
                T_estimated = self._eval_pose(rebuild_feature, rebuild_feature_gt_untransformed,
                                              mean_diff, outputs, gt, B, partial=intra_data_transformed)
                rte, rre = compute_mean_RTE_RRE_batch(Transformation_matrix_disturbance_inv.cpu().numpy(),
                                                      T_estimated.detach().cpu().numpy(), self.dataset.shape_scale.item())
                AverageMeter_RTE.update(rte)
                AverageMeter_RRE.update(rre)
                T_icp = getattr(self, '_eval_T_icp', None)
                if T_icp is not None:
                    rte_icp, rre_icp = compute_mean_RTE_RRE_batch(Transformation_matrix_disturbance_inv.cpu().numpy(),
                                                                  T_icp.detach().cpu().numpy(), self.dataset.shape_scale.item())
                    AverageMeter_RTE_ICP.update(rte_icp)
                    AverageMeter_RRE_ICP.update(rre_icp)
                    _val_has_icp = True



            for idx in range(outputs[3].shape[0]):
                input_pcd = intra_data_transformed[idx]
                input_pcd = input_pcd.cpu().detach().numpy()
                input_pcd_o3d = o3d.geometry.PointCloud()
                input_pcd_o3d.points = o3d.utility.Vector3dVector(input_pcd)

                pcd_completed = outputs[3][idx]
                pcd_completed = torch.matmul(Transformation_matrix_disturbance_inv[idx, :3, :3],
                                             pcd_completed.transpose(0, 1)).transpose(0, 1) + \
                                Transformation_matrix_disturbance_inv[idx, :3, 3].unsqueeze(0)

                # transform back, CT scale, and intraoperative centroid
                pcd_completed = pcd_completed.cpu().detach().numpy()

                pcd_completed_o3d = o3d.geometry.PointCloud()
                pcd_completed_o3d.points = o3d.utility.Vector3dVector(pcd_completed)
                # o3d.visualization.draw_geometries([input_pcd_o3d.paint_uniform_color((0,0,1)),pcd_completed_o3d.paint_uniform_color((0,1,0))])
                # o3d.visualization.draw_geometries([gt_o3d.paint_uniform_color((1,0,0)),pcd_completed_o3d.paint_uniform_color((0,1,0))])
                dists_gt_2_pred = np.asarray(gt_o3d.compute_point_cloud_distance(pcd_completed_o3d))
                dists_pred_2_gt = np.asarray(pcd_completed_o3d.compute_point_cloud_distance(gt_o3d))
                CD = (0.5 * (dists_gt_2_pred.mean() + dists_pred_2_gt.mean())) * self.dataset.shape_scale
                HD95 = (0.5 * np.percentile(dists_gt_2_pred, 95) + 0.5 * np.percentile(dists_pred_2_gt,
                                                                                       95)) * self.dataset.shape_scale
                AverageMeter_CD.update(CD)
                AverageMeter_HD95.update(HD95)

        _icp_val = (('  RTE_ICP = %.2f, RRE_ICP = %.2f' %
                     (AverageMeter_RTE_ICP.avg(), AverageMeter_RRE_ICP.avg())) if _val_has_icp else '')
        print_log('[Validation] EPOCH: %d  CD = %.2f, HD95 = %.2f, RTE = %.2f, RRE = %.2f%s' %
                  (self.epoch, AverageMeter_CD.avg(), AverageMeter_HD95.avg(), AverageMeter_RTE.avg(), AverageMeter_RRE.avg(), _icp_val), logger=self.logger)
        if self.experiment:
            _val_metrics = {'Validation RTE': AverageMeter_RTE.avg(), 'Validation RRE': AverageMeter_RRE.avg()}
            if _val_has_icp:
                _val_metrics['Validation RTE_ICP'] = AverageMeter_RTE_ICP.avg()
                _val_metrics['Validation RRE_ICP'] = AverageMeter_RRE_ICP.avg()
            self.experiment.log_metrics(_val_metrics, epoch=self.epoch)
        file_path = Path("./quantitative_evaluation_shape_completion/slice_wise.txt")

        # with file_path.open("a", encoding="utf-8") as f:
        #     f.write(f"specimen_id: {self.conf.dataset.specimen_id}, anatomy: {self.conf.dataset.anatomy}, "
        #             f"CD_mean: {AverageMeter_CD.avg():.3f}, CD_std: {AverageMeter_CD.std():.3f}, "
        #             f"HD95_mean: {AverageMeter_HD95.avg():.3f}, HD95_std: {AverageMeter_HD95.std():.3f}\n")
        self._last_val_metrics = {
            'CD': AverageMeter_CD.avg(), 'HD95': AverageMeter_HD95.avg(),
            'RTE': AverageMeter_RTE.avg(), 'RRE': AverageMeter_RRE.avg(),
            'RTE_ICP': (AverageMeter_RTE_ICP.avg() if _val_has_icp else None),
            'RRE_ICP': (AverageMeter_RRE_ICP.avg() if _val_has_icp else None)}
        return AverageMeter_CD.avg()
        # return intra_data







# ============================================================================
# DiffICP variant: joint shape-completion + registration WITH a differentiable-
# ICP refinement stage (the manuscript's coarse-to-fine registration).
#
# Subclasses Runner_ShapeCompletion and overrides only the two hooks it exposes:
#   * _registration_loss : adds the fine (DiffICP-refined) pose term to the
#                          coarse split-gradient loss -> L = lc*L_coarse + lf*L_fine
#   * _eval_pose         : refines the coarse pose with DiffICP at validation
# so the training loop, accumulation, schedulers and logging are reused verbatim.
#
# Gradient routing (keeps the split that fixed validation RTE):
#   - coarse term: rotation couples into the encoder, translation is detached.
#   - DiffICP init: rotation from the coupled pose, translation from the detached
#     pose -> the encoder still only ever receives rotation-related gradient.
#   - the point clouds fed to ICP are detached, so the fine loss trains the pose
#     head (via the init), not the completion encoder via the geometry.
#
# Memory: DiffICP uses a subsampled correspondence set (n_points) and gradient
# checkpointing per iteration, so peak memory ~= one iteration regardless of
# n_iters (see models/DiffICP.py). Set diff_icp.lambda_fine = 0
# to train coarse-only while still refining with DiffICP at validation.
#
# Simulated real-trajectory US data is generated automatically on first use (the
# dataloader calls generate_partial_pcd/us_pcd_synthesis_real_trajectory.py's
# ensure_simulated_data when the .npy is missing). To pre-generate everything up
# front instead:
#     ULTRABONES100K_ROOT=/path python generate_partial_pcd/us_pcd_synthesis_real_trajectory.py
# ============================================================================
from models.DiffICP import DiffICP

CONFIG_FILE = 'confs/default.conf'

# Test-time pipeline, shared by test.py (flag defaults) and validation: `passes` network
# passes with re-orientation, then DiffICP on the partial US cloud with these settings.
# Training keeps the conf's diff_icp block.
TEST_TIME = dict(passes=3, icp_src='partial', n_iters=300, n_points=4096, sigma=0.001, trim_ratio=0.1)


def _project_so3(R):
    """Nearest rotation matrix (SVD), batched (B, 3, 3)."""
    U, _, Vh = torch.linalg.svd(R.double())
    d = torch.sign(torch.det(U @ Vh))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return (U @ D @ Vh).to(R.dtype)


def reoriented_coarse(runner, x, gt, feat_gt, T1, completed1, passes):
    """Test-time re-orientation. The pose head is least accurate for large input rotations,
    so the network is re-run on the input pre-rotated by the current estimate: pass k sees
    x_k = x R_acc^T (still centred), predicts T_k, and the estimate for x is [R_k R_acc | t_k].
    Returns (T (B,4,4), completed cloud of the last pass in x's frame (B,N,3))."""
    T, comp = T1, completed1
    B = x.shape[0]
    for _ in range(passes - 1):
        R_acc = _project_so3(T[:, :3, :3])
        out, feat = runner.model_shapeCompletion(torch.matmul(x, R_acc.transpose(1, 2)).contiguous())
        Tk = runner.model_PoseFromRebuildFeature(
            feat, feat_gt.repeat((B, 1)), out[3].mean(dim=1) - gt.mean(dim=1))
        T = Tk.clone()
        T[:, :3, :3] = torch.matmul(Tk[:, :3, :3], R_acc)
        comp = torch.matmul(out[3], R_acc)             # back to x's frame
    return T, comp


class Runner_DiffICP(Runner_ShapeCompletion):
    def __init__(self, conf, experiment=None):
        super().__init__(conf, experiment=experiment)
        dc = conf.diff_icp
        self.diff_icp = DiffICP(
            n_iters=int(dc.get('n_iters', 4)),
            sigma=float(dc.get('sigma', 0.02)),
            sigma_init=float(dc.get('sigma_init', dc.get('sigma', 0.02))),
            n_points=int(dc.get('n_points', 1024)),
            grad_checkpoint=bool(dc.get('grad_checkpoint', True)),
            hard=bool(dc.get('hard', False)),
            trim_ratio=float(dc.get('trim_ratio', 0.0)),
        )
        self.lambda_coarse = float(dc.get('lambda_coarse', 0.5))
        self.lambda_fine = float(dc.get('lambda_fine', 1.0))
        # What the DiffICP refinement aligns to the GT:
        #   'completed' -> the completed cloud (ret[3]); 'partial' -> the raw
        #   partial input cloud. The coarse pose always comes from the completed
        #   shape feature; only the geometry fed to ICP changes.
        self.icp_src = str(dc.get('icp_src', 'completed'))

    # ---- training: coarse (split-gradient) + DiffICP fine refinement ---------
    def _registration_loss(self, ret, rebuild_feature, rebuild_feature_gt_untransformed,
                           mean_diff, gt_untransformed, T_gt, bs_reg, epoch, partial=None):
        self.model_PoseFromRebuildFeature.train()
        Tgt = T_gt[:bs_reg]
        scale = self.dataset.shape_scale

        # coarse, split-gradient (rotation coupled to encoder, translation detached)
        T_coupled = self.model_PoseFromRebuildFeature(
            rebuild_feature[:bs_reg], rebuild_feature_gt_untransformed[:bs_reg], mean_diff[:bs_reg])
        loss_rot = rte_rre_loss_batch(T_coupled, Tgt, w_rte=0.0, w_rre=1.0, translation_scale=scale)
        T_detached = self.model_PoseFromRebuildFeature(
            rebuild_feature[:bs_reg].detach(), rebuild_feature_gt_untransformed[:bs_reg],
            mean_diff[:bs_reg].detach())
        loss_trans = rte_rre_loss_batch(T_detached, Tgt, w_rte=1.0, w_rre=0.0, translation_scale=scale)
        loss_coarse = loss_rot + loss_trans

        if self.lambda_fine <= 0.0:
            self._reg_T_icp = None
            return self.lambda_coarse * loss_coarse, T_detached.detach()

        # DiffICP init: rotation from the coupled pose, translation from the
        # detached pose -> the encoder only ever gets rotation-related gradient.
        R = T_coupled[:, :3, :3]
        t = T_detached[:, :3, 3]
        T_init = torch.zeros(bs_reg, 4, 4, device=R.device, dtype=R.dtype)
        T_init[:, :3, :3] = R
        T_init[:, :3, 3] = t
        T_init[:, 3, 3] = 1.0

        # detached geometry: fine loss trains the pose head via T_init, not the encoder.
        # ICP source: completed cloud (default) or the raw partial input (icp_src).
        if self.icp_src == 'partial' and partial is not None:
            src = partial[:bs_reg].detach()
        else:
            src = ret[3][:bs_reg].detach()
        tgt = gt_untransformed[:bs_reg].detach()
        T_fine = self.diff_icp(src, tgt, T_init)
        loss_fine = rte_rre_loss_batch(T_fine, Tgt, w_rte=1.0, w_rre=1.0, translation_scale=scale)

        loss_registration = self.lambda_coarse * loss_coarse + self.lambda_fine * loss_fine
        # Primary metric (RTE/RRE) = coarse pose; ICP-refined pose logged as RTE_ICP/RRE_ICP.
        self._reg_T_icp = T_fine.detach()
        return loss_registration, T_detached.detach()

    def set_icp(self, n_iters, n_points, sigma, trim_ratio):
        """Set the DiffICP settings (n_points <= 0 or None = full resolution; constant sigma)."""
        icp = self.diff_icp
        icp.n_iters, icp.trim_ratio = n_iters, trim_ratio
        icp.n_points = None if not n_points or n_points <= 0 else n_points
        icp.sigma = icp.sigma_init = sigma

    # ---- validation: the test-time pipeline (TEST_TIME), under no_grad --------
    def _eval_pose(self, rebuild_feature, rebuild_feature_gt_untransformed, mean_diff, outputs, gt, B, partial=None):
        T_coarse = self.model_PoseFromRebuildFeature(
            rebuild_feature.detach(),
            rebuild_feature_gt_untransformed.repeat((B, 1)).detach(),
            mean_diff.detach())
        T_start, completed = T_coarse, outputs[3]
        if TEST_TIME['passes'] > 1:
            T_start, completed = reoriented_coarse(self, partial, gt, rebuild_feature_gt_untransformed,
                                                   T_coarse, outputs[3], TEST_TIME['passes'])
        src = partial.detach() if TEST_TIME['icp_src'] == 'partial' else completed.detach()
        tgt = gt.detach()
        if tgt.shape[0] == 1 and B > 1:
            tgt = tgt.repeat(B, 1, 1)
        # Test-time DiffICP settings for validation only; the training settings are restored.
        icp = self.diff_icp
        train_icp = (icp.n_iters, icp.n_points, icp.sigma, icp.sigma_init, icp.trim_ratio)
        self.set_icp(TEST_TIME['n_iters'], TEST_TIME['n_points'], TEST_TIME['sigma'], TEST_TIME['trim_ratio'])
        try:
            # Primary (RTE/RRE) = pass-1 coarse; refined (RTE_ICP/RRE_ICP) stashed for logging.
            self._eval_T_icp = self.diff_icp(src, tgt, T_start)
        finally:
            icp.n_iters, icp.n_points, icp.sigma, icp.sigma_init, icp.trim_ratio = train_icp
        return T_coarse


def main_ShapeCompletion_UltraBones100k_realtraj_diffICP():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--target_specimen_id', type=int, default=3)
    parser.add_argument('--target_anatomy', type=str, default='fibula')
    # Resume training from a checkpoint of THIS experiment (same output folder).
    # If resume_epoch < 0, or no checkpoint exists at that epoch, train from scratch.
    parser.add_argument('--resume_epoch', type=int, default=100)
    args = parser.parse_args()

    conf = read_confs(CONFIG_FILE)
    for specimen_id in range(1, 15):
        if args.target_specimen_id > 0 and specimen_id != args.target_specimen_id:
            continue
        for anatomy in ["fibula", "tibia"]:
            if specimen_id==1 and anatomy=="fibula":
                continue
            if args.target_anatomy is not None and args.target_anatomy != anatomy:
                continue

            if int(os.environ.get("PROFILE_STEPS", "0")) == 0:
                experiment = Experiment(
                    api_key=os.getenv("COMET_API_KEY"),
                    project_name=os.getenv("COMET_PROJECT_NAME", "comreg"),
                    workspace=os.getenv("COMET_WORKSPACE"),
                )
                experiment.set_name(
                    f"{specimen_id}_{anatomy}_hidden[{conf.model_pose_estimator.hidden_dim}]"
                    f"_drop[{conf.model_pose_estimator.dropout}]_bs[{conf.dataset.bs}]"
                    f"_joint_regRatio[{conf.get('train.reg_loss_ratio', 0.1)}]_realtrajSim"
                    f"_diffICP[it{conf.diff_icp.n_iters}_lf{conf.diff_icp.lambda_fine}]")
                experiment.log_code(folder=".", overwrite=True)
            else:
                experiment = None

            conf.put('dataset.specimen_id', specimen_id)
            conf.put('dataset.anatomy', anatomy)

            # Resume from a checkpoint of this experiment if requested and present;
            # otherwise (resume_epoch < 0 or missing file) train from scratch.
            conf.pop('ckp_file_path', None)  # clear any stale value from a prior iteration
            if args.resume_epoch >= 0:
                ckpt = os.path.join(conf['general.base_exp_dir'], f"{specimen_id}_{anatomy}",
                                    "checkpoints", "ShapeCompletion", f"ckpt-{args.resume_epoch}.pth")
                if os.path.isfile(ckpt):
                    conf.put('ckp_file_path', ckpt)
                    print(f"[resume] resuming from {ckpt}")
                else:
                    print(f"[resume] no checkpoint at {ckpt}; training from scratch")

            runner = Runner_DiffICP(conf, experiment=experiment)
            runner.train()
        print("finish")


if __name__ == '__main__':
    seed_everything(1234, deterministic=False)
    main_ShapeCompletion_UltraBones100k_realtraj_diffICP()



