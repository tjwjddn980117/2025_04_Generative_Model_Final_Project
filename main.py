import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch, gc
import torchvision
import pytorch_lightning as pl
import json
import pickle
import shutil
import os
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from functools import partial
from PIL import Image

import torch.distributed as dist
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins import DDPPlugin as DDPStrategy


sys.path.append("./stable_diffusion")

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from tqdm import tqdm

from torch.utils.data.dataloader import default_collate

class NanGuardCallback(Callback):
    # def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx):
    #     # 2) gradient NaN/Inf 제거
    #     for p in pl_module.parameters():
    #         if p.grad is not None:
    #             # NaN → 0, +inf→1e3, -inf→-1e3
    #             p.grad.data = torch.nan_to_num(p.grad.data,
    #                                            nan=0.0,
    #                                            posinf=1e3,
    #                                            neginf=-1e3)
    def on_after_backward(self, trainer, pl_module, *args, **kwargs):
        for p in pl_module.parameters():
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(
                    p.grad.data,
                    nan=0.0,
                    posinf=1e3,
                    neginf=-1e3
                )

def collate_impute_zero(batch):
    """
    1) default_collate 으로 묶고
    2) 텐서라면 nan/inf 를 지정 값으로 치환
    3) 딕셔너리나 리스트/튜플 안도 재귀 처리
    """
    batch = default_collate(batch)
    
    def _impute(x):
        if torch.is_tensor(x):
            # nan → 0, +inf → 1e3, -inf → -1e3 (원하는 값으로 조정)
            return torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        elif isinstance(x, dict):
            return {k: _impute(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(_impute(v) for v in x)
        else:
            return x

    return _impute(batch)

def _on_before_batch_transfer(self, batch, dataloader_idx: int):
    # GPU로 옮기기 전에 batch를 half로 통일
    def to_half(x):
        return x.half() if isinstance(x, torch.Tensor) else x
    if isinstance(batch, dict):
        return {k: to_half(v) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(to_half(x) for x in batch)
    else:
        return to_half(batch)

# LightningModule 서브클래스 전체에 적용
# pl.LightningModule.on_before_batch_transfer = _on_before_batch_transfer

def _collate_to_half(batch):
    return 
    batch = default_collate(batch)
    return {
        k: v.half() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

class TQDMProgressBarCustom(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        total_batches = len(trainer.train_dataloader)
        self.bar = tqdm(
            total=total_batches,
            desc=f"[Epoch {trainer.current_epoch+1}/{trainer.max_epochs}]",
            leave=False
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.bar.update(1)

    def on_train_epoch_end(self, trainer, pl_module):
        self.bar.close()

class DataLoaderTqdmCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        # trainer.train_dataloader (단일) vs .train_dataloaders (리스트)
        if hasattr(trainer, 'train_dataloaders'):
            orig = trainer.train_dataloaders
        else:
            orig = [trainer.train_dataloader]
        wrapped = [
            tqdm(dl, desc=f"[Data ▶ Epoch 1/{trainer.max_epochs}]", total=len(dl), leave=False)
            for dl in orig
        ]
        # 다시 할당
        if hasattr(trainer, 'train_dataloaders'):
            trainer.train_dataloaders = wrapped
        else:
            trainer.train_dataloader = wrapped[0]

    def on_validation_start(self, trainer, pl_module):
        # validation 쪽도 마찬가지
        if hasattr(trainer, 'val_dataloaders'):
            orig = trainer.val_dataloaders
        else:
            orig = [trainer.val_dataloader]
        wrapped = [
            tqdm(dl, desc="Validation", total=len(dl), leave=False)
            for dl in orig
        ]
        if hasattr(trainer, 'val_dataloaders'):
            trainer.val_dataloaders = wrapped
        else:
            trainer.val_dataloader = wrapped[0]

class ClearCacheCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        gc.collect()

    def on_validation_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        gc.collect()

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    # parser.add_argument(
    #     "--gpus",
    #     type=int,
    #     default=1,
    #     help="number of GPUs to use"
    # )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_thing",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=init_fn, persistent_workers=False, collate_fn=collate_impute_zero)
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn, persistent_workers=False, collate_fn=collate_impute_zero)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, persistent_workers=False, collate_fn=collate_impute_zero)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, persistent_workers=False, collate_fn=collate_impute_zero)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=False, collate_fn=collate_impute_zero)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            # os.makedirs(self.logdir, exist_ok=True)
            # os.makedirs(self.ckptdir, exist_ok=True)
            # os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    origin_size = None
    if not isinstance(data, torch.Tensor):
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")
    else:
        origin_size = data.size()
        tensor = data.reshape(-1)

    tensor_type = tensor.dtype

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.FloatTensor(size=(max_size,)).cuda().to(tensor_type))
    if local_size != max_size:
        padding = torch.FloatTensor(size=(max_size - local_size,)).cuda().to(tensor_type)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        if origin_size is None:
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        else:
            buffer = tensor[:size]
            data_list.append(buffer)

    if origin_size is not None:
        new_shape = [-1] + list(origin_size[1:])
        resized_list = []
        for data in data_list:
            # suppose the difference of tensor size exist in first dimension
            data = data.reshape(new_shape)
            resized_list.append(data)

        return resized_list
    else:
        return data_list

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(6, int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, prompts,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        names = {"reals": "before", "inputs": "after", "reconstruction": "before-vq", "samples": "after-gen"}
        # print(root)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=8)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(
                global_step,
                current_epoch,
                batch_idx,
                names[k])
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            # print(path)
            Image.fromarray(grid).save(path)

        filename = "gs-{:06}_e-{:06}_b-{:06}_prompt.json".format(
            global_step,
            current_epoch,
            batch_idx)
        path = os.path.join(root, filename)
        with open(path, "w") as f:
            for p in prompts:
                f.write(f"{json.dumps(p)}\n")

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0) or (split == "val" and batch_idx == 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            prompts = batch["edit"]["c_crossattn"][:self.max_images]
            prompts = [p for ps in all_gather(prompts) for p in ps]

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                images[k] = torch.cat(all_gather(images[k][:N]))
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images, prompts,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            if len(self.log_steps) > 0:
                self.log_steps.pop(0)
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value
    #os.makedirs('/home/jovyan/.cache/torch/hub/checkpoints/')
    #shutil.copy("checkpoint_liberty_with_aug.pth","/home/jovyan/.cache/torch/hub/checkpoints/")
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    assert opt.name
    cfg_fname = os.path.split(opt.base[0])[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    nowname = f"{cfg_name}_{opt.name}"
    logdir = os.path.join(opt.logdir, nowname)
    ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
    resume = False

    if os.path.isfile(ckpt):
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
        resume = True
        
    
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        if resume:
            # By default, when finetuning from Stable Diffusion, we load the EMA-only checkpoint to initialize all weights.
            # If resuming InstructPix2Pix from a finetuning checkpoint, instead load both EMA and non-EMA weights.
            config.model.params.load_ema = True

        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        # trainer_config["accelerator"] = "ddp"
        trainer_config["gpus"] = 1
        trainer_config["accelerator"] = "gpu"
        trainer_config["progress_bar_refresh_rate"] = 1
        trainer_config["precision"] = 32 
        trainer_config["gradient_clip_val"]=1.0
        trainer_config["gradient_clip_algorithm"]="value"
        # trainer_config["amp_level"] = "O1"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        # if not "gpus" in trainer_config:
        #     del trainer_config["accelerator"]
        #     cpu = True
        # else:
        #     gpuinfo = trainer_config["gpus"]
        #     print(f"Running on GPUs {gpuinfo}")
        #     cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)
        #model = model.cuda().half()

        import types
        # 1) transfer_batch_to_device 훅 정의
        def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
            # Lightning 기본 동작으로 GPU로 올리고
            batch = super(type(self), self).transfer_batch_to_device(batch, device, dataloader_idx)
            # 모든 Tensor를 half precision으로 변환
            def to_half(x):
                return x.half() if isinstance(x, torch.Tensor) else x
            if isinstance(batch, dict):
                return {k: to_half(v) for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                return type(batch)(to_half(x) for x in batch)
            else:
                return to_half(batch)

        # 2) 인스턴스에 바인딩
        #model.transfer_batch_to_device = types.MethodType(transfer_batch_to_device, model)
# 
        # from typing import Tuple
        # 
        # class SafeNormMixin:
        #     def forward(self, x):
        #         orig_dtype = x.dtype
        #         x = x.float()
        #         out = super().forward(x)
        #         return out.to(orig_dtype)
# 
        # # LayerNorm, GroupNorm 클래스 재정의
        # class SafeLayerNorm(SafeNormMixin, torch.nn.LayerNorm): pass
        # class SafeGroupNorm(SafeNormMixin, torch.nn.GroupNorm): pass
# 
        # def find_parent_module(root: torch.nn.Module, target_name: str) -> Tuple[torch.nn.Module, str]:
        #     """
        #     Given a root module and the full dotted name of a submodule (as from named_modules()),
        #     return (parent_module, attribute_name) so you can setattr on it.
        #     """
        #     parts = target_name.split(".")
        #     assert parts, "Empty name"
        #     # If top-level, parent is root, attr is full name
        #     if len(parts) == 1:
        #         return root, parts[0]
        #     # Traverse down to the parent of the target
        #     parent = root
        #     for p in parts[:-1]:
        #         parent = getattr(parent, p)
        #     return parent, parts[-1]
# 
        # # 모델 인스턴스화 직후에 패치
        # for name, module in model.named_modules():
        #     if isinstance(module, torch.nn.LayerNorm):
        #         parent, attr = find_parent_module(model, name)
        #         setattr(parent, attr, SafeLayerNorm(module.normalized_shape, module.eps, module.elementwise_affine))
        #     if isinstance(module, torch.nn.GroupNorm):
        #         parent, attr = find_parent_module(model, name)
        #         setattr(parent, attr, SafeGroupNorm(module.num_groups, module.num_channels, module.eps, module.affine))
        
        # 1) CLIP 텍스트 인코더 동결
        for p in model.cond_stage_model.parameters():
            p.requires_grad = False

        # 2) VAE first stage 동결
        for p in model.first_stage_model.parameters():
            p.requires_grad = False

        # 3) EMA 객체 동결 (사실 gradient를 갖지 않으므로 그냥 놔둬도 무방)
        for p in model.model_ema.parameters():
            p.requires_grad = False

        # 4) UNet만 학습
        for name, p in model.model.diffusion_model.named_parameters():
            p.requires_grad = True

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": False,
            }
        }

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
            #"progress_bar_custom": {
            #    "target": "main.TQDMProgressBarCustom",
            #    "params": {}
            #},
            "dataloader_tqdm": {
                "target": "main.DataLoaderTqdmCallback",
                "params": {}
            },
            "clear_chace": {
                "target": "main.ClearCacheCallback",
                "params": {}
            },
            "Nan_check": {
                "target": "main.NanGuardCallback",
                "params": {}
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {
                "target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                'params': {
                    "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    'save_top_k': -1,
                    'every_n_train_steps': 50,
                    'save_weights_only': True
                }
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # trainer = Trainer.from_argparse_args(trainer_opt, plugins=DDPPlugin(find_unused_parameters=False), **trainer_kwargs)
        # trainer = Trainer.from_argparse_args(trainer_opt, plugins=DDPStrategy(find_unused_parameters=False), **trainer_kwargs)
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        # 단일 GPU 1개만 쓰도록 devices=1, accelerator="gpu" 지정
        # trainer = Trainer(
        #     gpus=1,
        #     distributed_backend="ddp",     
        #     strategy=DDPStrategy(find_unused_parameters=False),
        #     **trainer_kwargs
        # )
        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        # if not cpu:
        #     # ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        #     ngpu = 1
        # else:
        #     ngpu = 1
        ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        #signal.signal(signal.SIGUSR1, melk)
        #signal.signal(signal.SIGUSR2, divein)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, melk)
        if hasattr(signal, "SIGUSR2"):
            signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                # model = model.cuda()
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
