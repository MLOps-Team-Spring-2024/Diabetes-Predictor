import pytest
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT

dataset_path = _PATH_DATA+"/processed"


N_train = 60000
N_test = 10000

mist_transform = transforms.Compose([transforms.ToTensor()])

@pytest.fixture
def train_dataset():
    return MNIST(dataset_path, transform=mist_transform, train=True, download=True)

@pytest.fixture
def test_dataset():
    return MNIST(dataset_path, transform=mist_transform, train=False, download=True)


def test_train_dataset_length(train_dataset):
    assert len(train_dataset) == N_train

def test_test_dataset_length(test_dataset):
    assert len(test_dataset) == N_test

def test_label_representation(train_dataset, test_dataset):
    labels = set()
    for dataset in [train_dataset, test_dataset]:
        for _, label in dataset:
            labels.add(label)
    assert len(labels) == 10