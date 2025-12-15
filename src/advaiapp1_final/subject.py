# This source code is edited from "Tutorial 1: Introduction to PyTorch" by Phillip Lippe, which is licensed under CC BY-SA.
# Please refer to the following URL for the original:
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/01-introduction-to-pytorch.html
# https://github.com/Lightning-AI/tutorials/blob/main/course_UvA-DL/01-introduction-to-pytorch/notebook.py
# This file is licensed under CC BY-SA 4.0.

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false

from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.pyplot import figure
from numpy import ndarray
import torch
from torch import Tensor
from torch.nn import Linear, Module, Tanh
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SimpleClassifier(Module):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int) -> None:
        """
        Simple neural network model for classification.

        Args:
            num_inputs (int): Number of input dimensions
            num_hidden (int): Number of neurons in hidden layer
            num_outputs (int): Number of output dimensions
        """
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = Linear(num_inputs, num_hidden)
        self.act_fn = Tanh()
        self.linear2 = Linear(num_hidden, num_outputs)

    def forward(self, x: Tensor) -> Tensor:
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class XORDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, size: int, std: float = 0.1) -> None:
        """
        XORDataset.

        Args:
            size (int): Number of data points we want to generate
            std (float): Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self) -> None:
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label

    def __len__(self) -> int:
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label


def visualize_samples(
    data: Tensor | ndarray,
    label: Tensor | ndarray,
) -> Figure:
    """
    Visualize samples with pyplot.

    Args:
        data (Tensor | ndarray): Sample data to visualize
        label (Tensor | ndarray): Ground truth labels of sample data
    """
    if isinstance(data, Tensor):
        data = data.cpu().numpy()
    if isinstance(label, Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = figure(figsize=(4, 4))
    axes = fig.add_subplot()
    axes.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    axes.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    axes.set_title("Dataset samples")
    axes.set_ylabel(r"$x_2$")
    axes.set_xlabel(r"$x_1$")
    axes.legend()

    return fig


def train_model(
    model: Module,
    optimizer: Optimizer,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    loss_module: Module,
    num_epochs: int = 100,
) -> None:
    """
    Train neural network model.

    Args:
        model (Module): Neural network model to train, this argument will be changed in-place
        optimizer (Optimizer): Optimizer to update model parameters
        data_loader (DataLoader[tuple[Tensor, Tensor]]): DataLoader providing training data and labels
        loss_module (Module): Loss function module
        num_epochs (int): Number of training epochs
    """
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)): # pyright: ignore[reportUnusedVariable]
        for data_inputs, data_labels in data_loader:
            # Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)  # Output is [Batch size, 1], but we want [Batch size]

            # Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())

            # Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            # Step 5: Update the parameters
            optimizer.step()


def eval_model(
    model: Module,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
) -> float:
    """
    Evaluate model by accuracy.

    Args:
        model (Module): Neural network model to evaluate
        data_loader (DataLoader[tuple[Tensor, Tensor]]): DataLoader providing test data and labels

    Returns:
        acc (float): Accuracy of neural network model
    """
    model.eval()  # Set model to eval mode
    true_preds, num_preds = 0.0, 0.0

    with torch.no_grad():  # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:
            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)  # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long()  # Binarize predictions to 0 and 1

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    return acc


@torch.no_grad()  # Decorator, same effect as "with torch.no_grad(): ..." over the whole function.
def visualize_classification(
    model: Module,
    data: Tensor | ndarray,
    label: Tensor | ndarray,
) -> Figure:
    """
    Visualize classification of data.

    Args:
        model (Module): Neural network model to be used in classification
        data (Tensor | ndarray): Data to be classified
        label (Tensor | ndarray): Ground truth labels of data
    """
    if isinstance(data, Tensor):
        data = data.cpu().numpy()
    if isinstance(label, Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = figure(figsize=(4, 4))
    axes = fig.add_subplot()
    axes.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    axes.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    axes.set_title("Dataset samples")
    axes.set_ylabel(r"$x_2$")
    axes.set_xlabel(r"$x_1$")
    axes.legend()

    # Let's make use of a lot of operations we have learned above
    model.to(device)
    c0 = Tensor(to_rgba("C0")).to(device)
    c1 = Tensor(to_rgba("C1")).to(device)
    x1 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    x2 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds = model(model_inputs)
    preds = torch.sigmoid(preds)
    # Specifying "None" in a dimension creates a new one
    output_image = (1 - preds) * c0[None, None] + preds * c1[None, None]
    output_image = (
        output_image.cpu().numpy()
    )  # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    axes.imshow(output_image, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5))
    axes.grid(False)

    return fig
