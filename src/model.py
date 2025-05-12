import os

import lightning as L
import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms as T
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from timm import create_model
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


def read_data():
    load_dotenv()

    # Read changesets
    root_path = os.environ["ROOT_PATH"]
    positive = os.listdir(f"{root_path}/images/positive")
    negative = os.listdir(f"{root_path}/images/negative")

    # Create a DataFrame with paths and labels
    data = [(path, 1) for path in positive] + [(path, 0) for path in negative]
    data = pl.DataFrame(data, schema=["path", "label"], orient="row")

    return data


def split_data(data, test_size=0.2, random_state=42):
    # Split the data into train and test sets
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    # Further split the train set into train and validation sets
    train, val = train_test_split(train, test_size=test_size, random_state=random_state)

    return train, val, test


class CustomImageDataset(Dataset):
    def __init__(self, data: pl.DataFrame, img_dir: str, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get label
        label = self.data[idx]["label"].item()

        # Image directory depends on the label
        data_directory = (
            f"{self.img_dir}/positive" if label == 1 else f"{self.img_dir}/negative"
        )
        img_path = os.path.join(data_directory, self.data[idx]["path"].item())
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        # Normalize the image
        image = image.float() / 255.0

        return image, label


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling layers and activation function
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Dropout regularization
        self.dropout = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(int(64 * (INPUT_SIZE / 2**3) ** 2), 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Convolutional pass
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected pass
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Apply sigmoid activation
        x = torch.sigmoid(x)

        return x


class VandalismClassifier(L.LightningModule):
    def __init__(self, model, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.BCELoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=2
        )

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets.view(-1, 1).float())
        acc = self.accuracy(torch.sigmoid(outputs).round(), targets.view(-1, 1).float())
        self.log("train loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(
        self,
        batch,
    ):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets.view(-1, 1).float())
        self.log("val loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


if __name__ == "__main__":
    data = read_data()
    train, val, test = split_data(data)

    INPUT_SIZE = 512

    # Data loaders
    transforms = T.Compose(
        [
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
        ]
    )

    train_dataset = CustomImageDataset(
        data=train,
        img_dir=f"{os.environ['ROOT_PATH']}/images",
        transform=transforms,
    )

    val_dataset = CustomImageDataset(
        data=val,
        img_dir=f"{os.environ['ROOT_PATH']}/images",
        transform=transforms,
    )

    test_dataset = CustomImageDataset(
        data=test,
        img_dir=f"{os.environ['ROOT_PATH']}/images",
        transform=transforms,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Model
    device = torch.device(
        "mps"
        if torch.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    clf = VandalismClassifier(CNNClassifier(), lr=0.01)

    model = CNNClassifier().to(device)

    size = len(train_dataloader.dataset)
    model.train()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_dataloader):
            # Forward pass
            data = data.to(device)
            targets = targets.to(device).view(-1, 1).float()
            outputs = model(data)
            loss = loss_fn(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                )
    # Eval loop
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_dataloader):
            # Forward pass
            data = data.to(device)
            targets = targets.to(device).view(-1, 1).float()
            outputs = model(data)
            test_preds.extend(outputs.cpu().numpy())

    test_preds = torch.Tensor(test_preds).squeeze()

    test_preds
