import logging
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import AverageMeter, Transform
from dataset import AVADataset, preprocess
from emd_loss import EDMLoss
from model import create_model
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__file__)


def get_dataloaders(
        dataset: str, path_to_save_csv: Path, path_to_images: Path, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = Transform()

    preprocess(path_to_images)

    train_ds = AVADataset(path_to_save_csv / "train.csv", path_to_images, transform.train_transform)
    val_ds = AVADataset(path_to_save_csv / "val.csv", path_to_images, transform.val_transform)
    test_ds = AVADataset(path_to_save_csv / "test.csv", path_to_images, transform.val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_ds = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_ds


def emd_dis(x, y_true, dist_r=1):
    cdf_x = torch.cumsum(x, dim=-1)
    cdf_ytrue = torch.cumsum(y_true, dim=-1)
    if dist_r == 2:
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
    else:
        samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
    loss = torch.mean(samplewise_emd)
    return loss


def cal_metrics(output, target, bins=10):
    output = np.concatenate(output)
    target = np.concatenate(target)
    scores_mean = np.dot(output, np.arange(1, bins + 1))
    labels_mean = np.dot(target, np.arange(1, bins + 1))
    srcc, _ = spearmanr(scores_mean, labels_mean)
    plcc, _ = pearsonr(scores_mean, labels_mean)
    mse = ((scores_mean - labels_mean) ** 2).mean(axis=None)
    diff = (((scores_mean - float(bins / 2)) * (labels_mean - float(bins / 2))) >= 0)
    acc = np.sum(diff) / len(scores_mean) * 100
    output_tensor = torch.from_numpy(np.array(output))
    target_tensor = torch.from_numpy(np.array(target))
    with torch.no_grad():
        emd1 = emd_dis(output_tensor, target_tensor, dist_r=1)
        emd2 = emd_dis(output_tensor, target_tensor, dist_r=2)
        return [mse, srcc, plcc, acc, emd1, emd2]


def validate_and_test(
        dataset: str,
        path_to_save_csv: Path,
        path_to_images: Path,
        batch_size: int,
        num_workers: int,
        drop_out: float,
        path_to_model_state: Path,
) -> None:
    _, val_loader, test_loader = get_dataloaders(
        dataset=dataset, path_to_save_csv=path_to_save_csv, path_to_images=path_to_images, batch_size=batch_size,
        num_workers=num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = EDMLoss().to(device)

    best_state = torch.load(path_to_model_state)

    model = create_model(best_state["model_type"], drop_out=drop_out).to(device)
    model.load_state_dict(best_state["state_dict"])

    model.eval()
    '''
    validate_losses = AverageMeter()

    with torch.no_grad():
        for (x, y, ratio) in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model([x, ratio])
            loss = criterion(p_target=y, p_estimate=y_pred)
            validate_losses.update(loss.item(), x.size(0))
    '''

    test_losses = AverageMeter()
    scores_hist = []
    labels_hist = []
    with torch.no_grad():
        for (x, y, ratio) in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model([x, ratio])
            labels_hist.append(y.cpu().numpy())
            scores_hist.append(y_pred.cpu().numpy())
            loss = criterion(p_target=y, p_estimate=y_pred)
            test_losses.update(loss.item(), x.size(0))

    metrics = cal_metrics(scores_hist, labels_hist)
    print(f' --> Validation:')
    print(f'     - MSE {metrics[0]:.4f} | SRCC {metrics[1]:.4f} | LCC {metrics[2]:.4f} ')
    print(f'     - Acc {metrics[3]:.2f} | EMD_1 {metrics[4]:.4f}| EMD_2 {metrics[5]:.4f}')
    # logger.info(f"val loss {validate_losses.avg}; test loss {test_losses.avg}")
    logger.info(f"test loss {test_losses.avg}")


def get_optimizer(optimizer_type: str, model, init_lr: float) -> torch.optim.Optimizer:
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.5, weight_decay=9)
    else:
        raise ValueError(f"not such optimizer {optimizer_type}")
    return optimizer


class Trainer:
    def __init__(
        self,
        *,
        dataset: str,
        path_to_save_csv: Path,
        path_to_images: Path,
        num_epoch: int,
        model_type: str,
        num_workers: int,
        batch_size: int,
        init_lr: float,
        experiment_dir: Path,
        drop_out: float,
        optimizer_type: str,
        criterion: str,
    ):

        train_loader, val_loader, _ = get_dataloaders(
            dataset=dataset,
            path_to_save_csv=path_to_save_csv,
            path_to_images=path_to_images,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = create_model(model_type, drop_out=drop_out).to(self.device)
        optimizer = get_optimizer(optimizer_type=optimizer_type, model=model, init_lr=init_lr)

        self.model = model
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode="min", patience=5)
        if criterion == 'emd':
            self.criterion = EDMLoss().to(self.device)
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss().to(self.device)
        self.model_type = model_type

        experiment_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_dir = experiment_dir
        self.writer = SummaryWriter(str(experiment_dir / "logs"))
        self.num_epoch = num_epoch
        self.global_train_step = 0
        self.global_val_step = 0
        self.print_freq = 100

    def train_model(self):
        best_loss = float("inf")
        best_state = None
        for e in range(1, self.num_epoch + 1):
            train_loss = self.train()
            val_loss = self.validate()
            self.scheduler.step(metrics=val_loss)

            self.writer.add_scalar("train/loss", train_loss, global_step=e)
            self.writer.add_scalar("val/loss", val_loss, global_step=e)

            if best_state is None or val_loss < best_loss:
                logger.info(f"updated loss from {best_loss} to {val_loss}")
                best_loss = val_loss
                best_state = {
                    "state_dict": self.model.state_dict(),
                    "model_type": self.model_type,
                    "epoch": e,
                    "best_loss": best_loss,
                }
                torch.save(best_state, self.experiment_dir / "best_state.pth")

    def train(self):
        self.model.train()
        train_losses = AverageMeter()
        total_iter = len(self.train_loader.dataset) // self.train_loader.batch_size

        for idx, (x, y, ratio) in enumerate(self.train_loader):
            s = time.monotonic()

            x = x.to(self.device)
            y = y.to(self.device)
            ratio = ratio.to(self.device)
            y_pred = self.model([x, ratio])
            loss = self.criterion(p_target=y, p_estimate=y_pred)
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            train_losses.update(loss.item(), x.size(0))

            self.writer.add_scalar("train/current_loss", train_losses.val, self.global_train_step)
            self.writer.add_scalar("train/avg_loss", train_losses.avg, self.global_train_step)
            self.global_train_step += 1

            e = time.monotonic()
            if idx % self.print_freq:
                log_time = self.print_freq * (e - s)
                eta = ((total_iter - idx) * log_time) / 60.0
                print(f"iter #[{idx}/{total_iter}] " f"loss = {loss:.3f} " f"time = {log_time:.2f} " f"eta = {eta:.2f}")

        return train_losses.avg

    def validate(self):
        self.model.eval()
        validate_losses = AverageMeter()

        with torch.no_grad():
            for idx, (x, y, ratio) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                ratio = ratio.to(self.device)
                y_pred = self.model([x, ratio])
                loss = self.criterion(p_target=y, p_estimate=y_pred)
                validate_losses.update(loss.item(), x.size(0))

                self.writer.add_scalar("val/current_loss", validate_losses.val, self.global_val_step)
                self.writer.add_scalar("val/avg_loss", validate_losses.avg, self.global_val_step)
                self.global_val_step += 1

        return validate_losses.avg
