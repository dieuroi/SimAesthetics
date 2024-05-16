import logging
from pathlib import Path

import click

from clean_dataset import clean_and_split
from common import set_up_seed
from inference_model import InferenceModel
from trainer import Trainer, validate_and_test
from get_dataset import get_dataset_csv


def init_logging() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@click.group()
def cli():
    pass


@click.command("prepare-dataset", short_help="Parse, clean and split dataset")
@click.option("--dataset", help="choose a dataset(official|custom|TAD66K|EVA)", default="official", required=True,
              type=str)
@click.option("--path_to_dataset", help="origin dataset file", type=Path)
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", type=Path)
@click.option("--path_to_images", help="images directory", required=True, type=Path)
@click.option("--train_size", help="train dataset size", default=0.8, type=float)
@click.option("--num_workers", help="num workers for parallel processing", default=64, type=int)
def prepare_dataset(
        dataset: str, path_to_dataset: Path, path_to_save_csv: Path, path_to_images: Path, train_size: float,
        num_workers: int
):
    click.echo(f"Clean and split dataset to train|val|test in {num_workers} threads. It will takes several minutes")
    clean_and_split(
        dataset=dataset,
        path_to_dataset=path_to_dataset,
        path_to_save_csv=path_to_save_csv,
        path_to_images=path_to_images,
        train_size=train_size,
        num_workers=num_workers,
    )
    click.echo("Done!")


@click.command("train-model", short_help="Train model")
@click.option("--dataset", help="choose a dataset(official|custom|TAD66K|EVA)", default="official", type=str)
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", type=Path)
@click.option("--path_to_images", help="images directory", required=True, type=Path)
@click.option("--experiment_dir", help="directory name to save all logs and weight", required=True, type=Path)
@click.option("--model_type", help="nima mlsp fpg", default="mlsp", type=str)
@click.option("--batch_size", help="batch size", default=128, type=int)
@click.option("--num_workers", help="number of reading workers", default=16, type=int)
@click.option("--num_epoch", help="number of epoch", default=32, type=int)
@click.option("--init_lr", help="initial learning rate", default=0.0001, type=float)
@click.option("--drop_out", help="drop out", default=0.5, type=float)
@click.option("--checkpoint", help="checkpoint", default="PlaceHolder", type=str)
@click.option("--optimizer_type", help="optimizer type", default="adam", type=str)
@click.option("--seed", help="random seed", default=42, type=int)
@click.option("--criterion", help="criterion", default="emd", type=str)
@click.option("--distributed", help="False for single GPU, True for single Node", default="False", type=bool)
def train_model(
    dataset: str,
    path_to_save_csv: Path,
    path_to_images: Path,
    experiment_dir: Path,
    model_type: str,
    batch_size: int,
    num_workers: int,
    num_epoch: int,
    init_lr: float,
    drop_out: float,
    checkpoint: str,
    optimizer_type: str,
    seed: int,
    criterion: str,
    distributed: bool,
):
    click.echo("Train and validate model")
    path_to_save_csv = get_dataset_csv(dataset, path_to_save_csv)
    set_up_seed(seed)
    trainer = Trainer(
        dataset=dataset,
        path_to_save_csv=path_to_save_csv,
        path_to_images=path_to_images,
        experiment_dir=experiment_dir,
        model_type=model_type,
        batch_size=batch_size,
        num_workers=num_workers,
        num_epoch=num_epoch,
        init_lr=init_lr,
        drop_out=drop_out,
        checkpoint=checkpoint,
        optimizer_type=optimizer_type,
        criterion=criterion,
        distributed=distributed,
    )
    trainer.train_model()
    click.echo("Done!")


@click.command("get-image-score", short_help="Get image scores")
@click.option("--path_to_model_state", help="path to model weight .pth file", required=True, type=Path)
@click.option("--path_to_image", help="image ", required=True, type=Path)
def get_image_score(path_to_model_state, path_to_image):
    model = InferenceModel(path_to_model_state=path_to_model_state)
    result = model.predict_from_file(path_to_image)
    click.echo(result)


@click.command("validate-model", short_help="Validate model")
@click.option("--dataset", help="choose a dataset(official|custom|TAD66K|EVA)", default="official", type=str)
@click.option("--path_to_model_state", help="path to model weight .pth file", required=True, type=Path)
@click.option("--path_to_images", help="images directory", required=True, type=Path)
@click.option("--path_to_save_csv", help="where save train.csv|val.csv|test.csv", type=Path)
@click.option("--batch_size", help="batch size", default=128, type=int)
@click.option("--num_workers", help="number of reading workers", default=16, type=int)
@click.option("--drop_out", help="drop out", default=0.0, type=float)
def validate_model(dataset, path_to_model_state, path_to_save_csv, path_to_images, batch_size, num_workers, drop_out):
    path_to_save_csv = get_dataset_csv(dataset, path_to_save_csv)
    validate_and_test(
        dataset=dataset,
        path_to_model_state=path_to_model_state,
        path_to_save_csv=path_to_save_csv,
        path_to_images=path_to_images,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_out=drop_out,
    )
    click.echo("Done!")


def main():
    init_logging()
    cli.add_command(prepare_dataset)
    cli.add_command(train_model)
    cli.add_command(validate_model)
    cli.add_command(get_image_score)
    cli()


if __name__ == "__main__":
    main()
