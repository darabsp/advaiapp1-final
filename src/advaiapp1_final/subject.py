# This source code is edited from "Tutorial 1: Introduction to PyTorch" by Phillip Lippe, which is licensed under CC BY-SA.
# Please refer to the following URL for the original:
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/01-introduction-to-pytorch.html
# https://github.com/Lightning-AI/tutorials/blob/main/course_UvA-DL/01-introduction-to-pytorch/notebook.py
# This file is licensed under CC BY-SA 4.0.

import time

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from matplotlib.colors import to_rgba
from torch import Tensor
from tqdm.auto import tqdm


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Some init for my module

    def forward(self, x):
        # Function for performing the calculation of the module.
        pass


class SimpleClassifier(nn.Module):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int) -> None:
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x: Tensor) -> Tensor:
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class XORDataset(data.Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, size: int, std: float = 0.1) -> None:
        """XORDataset.

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
    data: Tensor | np.ndarray,
    label: Tensor | np.ndarray,
) -> None:
    if isinstance(data, Tensor):
        data = data.cpu().numpy()
    if isinstance(label, Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data_loader: data.DataLoader[tuple[Tensor, Tensor]],
    loss_module: nn.Module,
    num_epochs: int = 100,
) -> None:
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
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
    model: nn.Module,
    data_loader: data.DataLoader[tuple[Tensor, Tensor]],
) -> None:
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
    print(f"Accuracy of the model: {100.0 * acc:4.2f}%")


@torch.no_grad()  # Decorator, same effect as "with torch.no_grad(): ..." over the whole function.
def visualize_classification(
    model: nn.Module,
    data: Tensor | np.ndarray,
    label: Tensor | np.ndarray,
) -> None:
    if isinstance(data, Tensor):
        data = data.cpu().numpy()
    if isinstance(label, Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

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
    plt.imshow(output_image, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)


if __name__ == '__main__':
    torch.manual_seed(42)  # Setting the seed

    x = Tensor(2, 3, 4)
    print(x)

    x = Tensor([[1, 2], [3, 4]])
    print(x)

    x = torch.rand(2, 3, 4)
    print(x)

    shape = x.shape
    print("Shape:", x.shape)

    size = x.size()
    print("Size:", size)

    dim1, dim2, dim3 = x.size()
    print("Size:", dim1, dim2, dim3)

    np_arr = np.array([[1, 2], [3, 4]])
    tensor = torch.from_numpy(np_arr)

    print("Numpy array:", np_arr)
    print("PyTorch tensor:", tensor)

    tensor = torch.arange(4)
    np_arr = tensor.numpy()

    print("PyTorch tensor:", tensor)
    print("Numpy array:", np_arr)

    x1 = torch.rand(2, 3)
    x2 = torch.rand(2, 3)
    y = x1 + x2

    print("X1", x1)
    print("X2", x2)
    print("Y", y)

    x1 = torch.rand(2, 3)
    x2 = torch.rand(2, 3)
    print("X1 (before)", x1)
    print("X2 (before)", x2)

    x2.add_(x1)
    print("X1 (after)", x1)
    print("X2 (after)", x2)

    x = torch.arange(6)
    print("X", x)

    x = x.view(2, 3)
    print("X", x)

    x = x.permute(1, 0)  # Swapping dimension 0 and 1
    print("X", x)

    x = torch.arange(6)
    x = x.view(2, 3)
    print("X", x)

    W = torch.arange(9).view(3, 3)  # We can also stack multiple operations in a single line
    print("W", W)

    h = torch.matmul(x, W)  # Verify the result by calculating it by hand too!
    print("h", h)

    x = torch.arange(12).view(3, 4)
    print("X", x)

    print(x[:, 1])  # Second column

    print(x[0])  # First row

    print(x[:2, -1])  # First two rows, last column

    print(x[1:3, :])  # Middle two rows

    x = torch.ones((3,))
    print(x.requires_grad)

    x.requires_grad_(True)
    print(x.requires_grad)

    x = torch.arange(3, dtype=torch.float32, requires_grad=True)  # Only float tensors can have gradients
    print("X", x)

    a = x + 2
    b = a**2
    c = b + 3
    y = c.mean()
    print("Y", y)

    y.backward()

    print(x.grad)

    gpu_avail = torch.cuda.is_available()
    print(f"Is the GPU available? {gpu_avail}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

    x = torch.zeros(2, 3)
    x = x.to(device)
    print("X", x)

    x = torch.randn(5000, 5000)

    # CPU version
    start_time = time.time()
    _ = torch.matmul(x, x)
    end_time = time.time()
    print(f"CPU time: {(end_time - start_time):6.5f}s")

    # GPU version
    if torch.cuda.is_available():
        x = x.to(device)
        # CUDA is asynchronous, so we need to use different timing functions
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = torch.matmul(x, x)
        end.record()
        torch.cuda.synchronize()  # Waits for everything to finish running on the GPU
        print(f"GPU time: {0.001 * start.elapsed_time(end):6.5f}s")  # Milliseconds to seconds

    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    # Printing a module shows all its submodules
    print(model)

    for name, param in model.named_parameters():
        print(f"Parameter {name}, shape {param.shape}")

    dataset = XORDataset(size=200)
    print("Size of dataset:", len(dataset))
    print("Data point 0:", dataset[0])

    visualize_samples(dataset.data, dataset.label)
    plt.show()

    data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)

    data_inputs, data_labels = next(iter(data_loader))

    # The shape of the outputs are [batch_size, d_1,...,d_N] where d_1,...,d_N are the
    # dimensions of the data point returned from the dataset class
    print("Data inputs", data_inputs.shape, "\n", data_inputs)
    print("Data labels", data_labels.shape, "\n", data_labels)

    loss_module = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_dataset = XORDataset(size=1000)
    train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Push model to device. Has to be only done once
    model.to(device)

    train_model(model, optimizer, train_data_loader, loss_module)

    state_dict = model.state_dict()
    print(state_dict)

    # torch.save(object, filename). For the filename, any extension can be used
    torch.save(state_dict, "our_model.tar")

    # Load state dict from the disk (make sure it is the same name as above)
    state_dict = torch.load("our_model.tar")

    # Create a new model and load the state
    new_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    new_model.load_state_dict(state_dict)

    # Verify that the parameters are the same
    print("Original model\n", model.state_dict())
    print("\nLoaded model\n", new_model.state_dict())

    test_dataset = XORDataset(size=500)
    # drop_last -> Don't drop the last batch although it is smaller than 128
    test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

    eval_model(model, test_data_loader)

    visualize_classification(model, dataset.data, dataset.label)
    plt.show()
