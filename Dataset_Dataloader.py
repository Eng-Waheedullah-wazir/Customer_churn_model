import torch
from torch.utils.data import Dataset, DataLoader


class CustomDatasetTrain(Dataset):
    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_data_labels = train_labels

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index], self.train_data_labels[index]


class CustomDataSetTest(Dataset):
    def __init__(self, test_data, test_labels):
        self.test_data = test_data
        self.test_data_labels = test_labels

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        return self.test_data[index], self.test_data_labels[index]


class CustomDataSetVal(Dataset):
    def __init__(self, val_data, val_labels):
        self.val_data = val_data
        self.val_data_labels = val_labels

    def __len__(self):
        return len(self.val_data)

    def __getitem__(self, index):
        return self.val_data[index], self.val_data_labels[index]


def create_dataloaders(train_data, train_labels,
                       val_data, val_labels,
                       test_data, test_labels):

    train_dataset = CustomDatasetTrain(train_data, train_labels)
    val_dataset = CustomDataSetVal(val_data, val_labels)
    test_dataset = CustomDataSetTest(test_data, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=True
    )

    return train_loader, val_loader, test_loader