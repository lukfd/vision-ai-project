# usage:
#  - new model: uv run models/scripts/train_model.py -t resnet
#  - existing: uv run models/scripts/train_model.py -m models/basic_cnn/20251126_094206/epoch1_0.913
import argparse
import os
import time
from datetime import datetime

import numpy as np
import pyarrow as pa
import torch
from pyarrow import parquet
from sklearn.metrics import classification_report
from torchvision import models
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from image_data import ImageData

PHOTO_DIR = "data/clean/reduced_photos"
DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

transforms = models.ResNet18_Weights.DEFAULT.transforms()


def get_model(model_type: str, model_loc: str) -> nn.Module:
    if model_type == "basic_cnn":
        from basic_cnn import BasicCNN

        model = BasicCNN().to(DEVICE)
    elif model_type == "resnet":
        if not model_loc:
            model = models.resnet18(weights="DEFAULT")
        else:
            model = models.resnet18()

        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(DEVICE)
    elif model_type == "resnet50":
        if not model_loc:
            model = models.resnet50(weights="DEFAULT")
        else:
            model = models.resnet50()

        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(DEVICE)
    else:
        print(f"{model_type} not defined, exiting")
        raise Exception(f"{model_type} not defined")

    return model


def get_loss(
    m: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, lf: nn.Module
) -> torch.Tensor:
    inputs, labels = (
        inputs.to(DEVICE),
        labels.to(DEVICE).to(torch.float32).squeeze(),
    )
    outputs = m(inputs)
    loss = lf(outputs, labels)

    return loss


def train_one_epoch(m: nn.Module, loader: DataLoader, o: torch.optim.Optimizer, lf):
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(loader):
        o.zero_grad()
        loss = get_loss(m, inputs, labels, lf)
        loss.backward()
        o.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"{i + 1} / {len(loader)} -- running loss: {running_loss / (i + 1)}")

    return m, running_loss / len(loader)


def score_model(m: nn.Module, loader: DataLoader, lf: nn.Module):
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(loader):
        loss = get_loss(m, inputs, labels, lf)
        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"{i + 1} / {len(loader)} -- running loss: {running_loss / (i + 1)}")

    return running_loss / len(loader)


def get_best_model(model_type: str, root_dir: str):
    saved_models = os.listdir(root_dir)
    m = get_model(model_type, saved_models[0])
    m.load_state_dict(torch.load(f"{root_dir}/{saved_models[0]}", weights_only=True))
    print(f"evaluating best model -- {saved_models[0]}")

    return m, saved_models[0]


def run_classification(best_model, root_dir: str, model_home: str, loader: DataLoader):
    all_labels = []
    all_predictions = []

    best_model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = best_model(inputs)
            all_labels.append(labels.argmax(dim=1).cpu().numpy())
            all_predictions.append(outputs.argmax(dim=1).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    results = classification_report(all_labels, all_predictions, zero_division=np.nan)
    print(results)

    with open(f"{root_dir}/{model_home}.txt", "w") as f:
        f.write(results)


def main(
    model_type: str,
    model_loc: str = "",
    dataset_size: int = 10_000,
    batch_size: int = 1000,
    epochs=1,
):
    # %%
    model = get_model(model_type, model_loc)

    if model_loc:
        timestamp = model_loc.split("/")[2]
        best_vloss = float(model_loc.split("_")[-1])
        epoch_number = int(model_loc.split("epoch")[1].split("_")[0])
        root_dir = model_loc.rsplit("/", 1)[0]
        model.load_state_dict(torch.load(f"{model_loc}", weights_only=True))
    else:
        epoch_number = 0
        best_vloss = 10000
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = f"models/{model_type}/{timestamp}"
        os.makedirs(root_dir)

    # %%
    photos_scores = parquet.read_table("data/clean/model_data.parquet")
    photos_scores = photos_scores.add_column(
        0,
        "star_category",
        pa.array(
            photos_scores.select(["stars"])
            .to_pandas()
            .stars.apply(lambda r: (0.0 if r <= 3 else 1.0) if r <= 4 else 2.0)
        ),
    ).take(list(range(dataset_size)))
    _, class_counts = np.unique(
        photos_scores.select(["star_category"]).to_pandas().values, return_counts=True
    )
    weights = [(class_counts.sum() - x) / (x + 1e-5) for x in iter(class_counts)]

    # %%
    yelp_photos = ImageData(
        photos_scores.select(["star_category"]).to_pandas().star_category,
        photos_scores.select(["photo_id"]).to_pandas().photo_id,
        PHOTO_DIR,
        transform=transforms,
    )

    train, val, test = torch.utils.data.random_split(yelp_photos, [0.5, 0.3, 0.2])
    training_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=4)
    validation_loader = DataLoader(val, batch_size=batch_size, num_workers=4)

    writer = SummaryWriter(f"logs/{model_type}_{format(timestamp)}")
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weights).to(torch.float32).to(DEVICE)
    )
    summary(model)

    # %%
    for _ in range(epochs):
        epoch_number += 1
        print(f"EPOCH {epoch_number}")
        st = time.time()
        model.train(True)
        model, tloss = train_one_epoch(model, training_loader, opt, loss_fn)

        model.eval()
        with torch.no_grad():
            print(f"EVALUATING MODEL FOR EPOCH {epoch_number}")
            vloss = score_model(model, validation_loader, loss_fn)

        print(f"LOSS train {round(tloss, 5)}, validation {round(vloss, 5)}")
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": tloss, "Validation": vloss},
            epoch_number + 1,
        )
        writer.flush()
        if vloss < best_vloss:
            best_vloss = vloss
            model_path = f"{root_dir}/epoch{epoch_number}_{round(vloss, 3)}"
            [os.remove(os.path.join(root_dir, f)) for f in os.listdir(root_dir)]
            torch.save(model.state_dict(), model_path)
            print("***saved***")

        print(f"EPOCH TIME: {time.time() - st}")

    model, locn = get_best_model(model_type, root_dir)
    run_classification(model, root_dir, locn, test_loader)


if __name__ == "__main__":
    description = "Check if the parquet file ids are in the directory file"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-e", type=int, help="Number of epochs to train", required=False
    )
    parser.add_argument("-t", type=str, help="type of model to train", required=False)
    parser.add_argument("-m", type=str, help="path to saved model", required=False)
    parser.add_argument("-b", type=int, help="batch size", required=False)
    parser.add_argument(
        "-d", type=int, help="number of images to train on", required=False
    )

    args = parser.parse_args()
    if args.m:
        model_type = args.m.split("/")[1]
    else:
        model_type = args.t

    print(model_type, args.e, args.m, args.b, args.d)
    main(
        model_type,
        args.m,
        args.d or 10_000,
        args.b or 200,
        0 if args.e == 0 else (args.e or 1),
    )
