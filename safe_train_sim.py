import sys
# use line-buffering for both stdout and stderr
# sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
# sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
 
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
# from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_lowdim_policy import RobomimicLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

from diffusion_policy.policy.robomimic_lowdim_policy import RobomimicLowdimPolicy
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

# from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Sampler 

import h5py

# import mimicgen
# import mimicgen.utils.file_utils as MG_FileUtils
# import mimicgen.utils.robomimic_utils as RobomimicUtils
# from mimicgen.utils.misc_utils import add_red_border_to_frame
# from mimicgen.configs import MG_TaskSpec

OmegaConf.register_new_resolver("eval", eval, replace=True)
config_path='.'
# config_name = "image_franka_drawer_lay5_g53.yaml" 
# config_name="image_franka_image_224"

config_name = "safe_image_lift_sim_train"  # 58 demos.

with initialize(version_base=None, config_path=config_path):
    cfg_org = compose(
        config_name=config_name,
        overrides=[
            "hydra.run.dir=data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}",
            "training.seed=42",
            "training.device=cuda:0"
        ],
    )
    print(cfg_org)
    
OmegaConf.resolve(cfg_org)

print('resume: ', cfg_org.training.resume)
# last_checkpoint_dir = "/home/carl_lab/diffusion_policy/data/outputs/2024.12.13/03.05.17_train_diffusion_unet_pick_duck_image/"
last_checkpoint_dir = None 
class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
import datetime
### recreating workspace
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
output_dir = f"/Riad/diffusion_policy/data/outputs/Riad_sim_lift_mh_full_hand_view_{timestamp}"
os.mkdir(output_dir)
workspace = TrainDiffusionUnetHybridWorkspace(cfg_org, output_dir=output_dir)
self = workspace
print('output dir: ', output_dir)


cfg = copy.deepcopy(self.cfg)

# resume training
# if cfg.training.resume:
#     lastest_ckpt_path = self.get_checkpoint_path()
#     if lastest_ckpt_path.is_file():
#         print(f"Resuming from checkpoint {lastest_ckpt_path}")
#         self.load_checkpoint(path=lastest_ckpt_path)
new_config = OmegaConf.to_container(cfg.task.dataset, resolve=True )
del new_config['_target_']

if "shape_meta" not in new_config:
    raise ValueError("shape_meta is missing in dataset configuration")

dataset = RobomimicReplayImageDataset(**new_config)
len(dataset)

# new_config
cfg_dataloader = {key:value for key,value in cfg.dataloader.items()} 
print("Replay Buffer Keys:", dataset.replay_buffer.keys())
print("Action Shape:", dataset.replay_buffer['action'].shape if 'action' in dataset.replay_buffer else "Missing")

print("Lowdim Keys:", dataset.lowdim_keys)
for key in dataset.lowdim_keys:
    if key not in dataset.replay_buffer:
        print(f"Missing key in replay buffer: {key}")
    else:
        print(f"Key {key} shape: {dataset.replay_buffer[key].shape}")


train_dataloader = DataLoader(dataset, **cfg_dataloader)
normalizer = dataset.get_normalizer()

# configure validation dataset
val_dataset = dataset.get_validation_dataset()
val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

batch =  next(iter(train_dataloader))
batch.keys()
batch['action'].shape
batch['obs']['robot0_eye_in_hand_image'].shape
# batch['obs']['agentview_image'].shape
self.model.set_normalizer(normalizer)
if cfg.training.use_ema:
    self.ema_model.set_normalizer(normalizer)

# configure lr scheduler
lr_scheduler = get_scheduler(
    cfg.training.lr_scheduler,
    optimizer=self.optimizer,
    num_warmup_steps=cfg.training.lr_warmup_steps,
    num_training_steps=(
        len(train_dataloader) * cfg.training.num_epochs) \
            // cfg.training.gradient_accumulate_every,
    # pytorch assumes stepping LRScheduler every epoch
    # however huggingface diffusers steps it every batch
    last_epoch=self.global_step-1
)

# configure ema
ema: EMAModel = None
if cfg.training.use_ema:
    ema = hydra.utils.instantiate(
        cfg.ema,
        model=self.ema_model)
 
topk_manager = TopKCheckpointManager(
    save_dir=os.path.join(self.output_dir, 'checkpoints'),
    **cfg.checkpoint.topk
)

# device transfer
device = torch.device(cfg.training.device)
self.model.to(device)
if self.ema_model is not None:
    self.ema_model.to(device)
optimizer_to(self.optimizer, device)
print('output dir: ', output_dir)
train_sampling_batch = None
log_path = os.path.join(self.output_dir, 'logs.json.txt')
with JsonLogger(log_path) as json_logger:
    for local_epoch_idx in range(cfg.training.num_epochs):
        step_log = dict()
        # ========= train for this epoch ==========
        train_losses = list()
        with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                if train_sampling_batch is None:
                    train_sampling_batch = batch

                # compute loss
                raw_loss = self.model.compute_loss(batch)
                loss = raw_loss / cfg.training.gradient_accumulate_every
                loss.backward()

                # step optimizer
                if self.global_step % cfg.training.gradient_accumulate_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                
                # update ema
                if cfg.training.use_ema:
                    ema.step(self.model)

                # logging
                raw_loss_cpu = raw_loss.item()
                tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                train_losses.append(raw_loss_cpu)
                step_log = {
                    'train_loss': raw_loss_cpu,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }

                is_last_batch = (batch_idx == (len(train_dataloader)-1))
                if not is_last_batch:
                    # log of last step is combined with validation and rollout
                     
                    json_logger.log(step_log)
                    self.global_step += 1

                if (cfg.training.max_train_steps is not None) \
                    and batch_idx >= (cfg.training.max_train_steps-1):
                    break

        # at the end of each epoch
        # replace train_loss with epoch average
        train_loss = np.mean(train_losses)
        step_log['train_loss'] = train_loss

        # ========= eval for this epoch ==========
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()

 
        # run validation
        if (self.epoch % cfg.training.val_every) == 0:
            with torch.no_grad():
                val_losses = list()
                with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        loss = self.model.compute_loss(batch)
                        val_losses.append(loss)
                        if (cfg.training.max_val_steps is not None) \
                            and batch_idx >= (cfg.training.max_val_steps-1):
                            break
                if len(val_losses) > 0:
                    val_loss = torch.mean(torch.tensor(val_losses)).item()
                    # log epoch average validation loss
                    step_log['val_loss'] = val_loss

        # run diffusion sampling on a training batch
        if (self.epoch % cfg.training.sample_every) == 0:
            with torch.no_grad():
                # sample trajectory from training set, and evaluate difference
                batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                obs_dict = batch['obs']
                gt_action = batch['action']
                
                result = policy.predict_action(obs_dict)
                pred_action = result['action_pred']
                mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                step_log['train_action_mse_error'] = mse.item()
                del batch
                del obs_dict
                del gt_action
                del result
                del pred_action
                del mse
        
        # checkpoint
        if (self.epoch % 100) == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # self.save_checkpoint(tag=f'epoch_{self.epoch}') 
            checkpoint_name = f'epoch_{self.epoch}_{timestamp}'
            path_ = self.save_checkpoint(tag=checkpoint_name)
            print(path_)

            
        # ========= eval end for this epoch ==========
        policy.train()

        # end of epoch
        # log of last step is combined with validation and rollout
         
        json_logger.log(step_log)
        self.global_step += 1
        self.epoch += 1
# 879/60 
# took 1184m  = 19.13333 hours
self.save_checkpoint(tag=f"after_train_{self.epoch}_epochs")