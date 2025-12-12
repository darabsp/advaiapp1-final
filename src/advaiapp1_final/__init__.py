# pyright: reportUnknownMemberType=false

from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils import data
from .subject import SimpleClassifier, XORDataset, eval_model, train_model, visualize_classification, visualize_samples, device

def main() -> None:
    torch.manual_seed(42)

    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    dataset = XORDataset(size=200)

    samples_fig = visualize_samples(dataset.data, dataset.label)
    samples_fig.show()

    loss_module = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_dataset = XORDataset(size=1000)
    train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    model.to(device)

    train_model(model, optimizer, train_data_loader, loss_module)

    state_dict = model.state_dict()

    torch.save(state_dict, "our_model.tar")

    state_dict = torch.load("our_model.tar")

    new_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    new_model.load_state_dict(state_dict)

    test_dataset = XORDataset(size=500)
    test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

    eval_model(model, test_data_loader)

    classified_fig = visualize_classification(model, dataset.data, dataset.label)
    classified_fig.show()

    plt.waitforbuttonpress()
