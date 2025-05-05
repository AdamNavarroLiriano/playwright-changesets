import os

import lightning as L
import polars as pl
import torch
import torch.nn as nn
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


data = read_data()
train, val, test = split_data(data)


INPUT_SIZE = 512

image = read_image(path)
resize = T.Resize((INPUT_SIZE, INPUT_SIZE))

train[0]["path"].item()


class CustomImageDataset(Dataset):
    def __init__(
        self, data: pl.DataFrame, img_dir: str, transform=None, target_transform=None
    ):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

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
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


transforms = T.Compose(
    [
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
    ]
)

image_dataset = CustomImageDataset(
    data=train,
    img_dir=f"{os.environ['ROOT_PATH']}/images",
    transform=None,
    target_transform=None,
)

image_dataset[0]


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
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class VandalismClassifier(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.BCELoss()

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.001)


class ClassificationData(L.LightningDataModule):
    def train_dataloader(self):
        train_dataset = datasets.StanfordCars(
            root=".", download=False, transform=DEFAULT_TRANSFORM
        )
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True, num_workers=5
        )
