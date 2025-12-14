from pytest import fixture

from copy import deepcopy
import matplotlib
import torch
from torch import Size, Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from advaiapp1_final.subject import (
    SimpleClassifier,
    XORDataset,
    visualize_samples,
    train_model,
    eval_model,
    visualize_classification,
)


# NOTE: Tkinterが使えないんだが？というエラーの抑制
@fixture(autouse=True)
def make_matplotlib_not_use_gui():
    matplotlib.use("Agg")

@fixture
def dataset_length():
    return 200

@fixture
def simple_classifier():
    return SimpleClassifier(2, 4, 1)

@fixture
def sgd_optimizer(
    simple_classifier: SimpleClassifier,
):
    return SGD(simple_classifier.parameters())

@fixture
def bce_with_logits_loss_module():
    return BCEWithLogitsLoss()

@fixture
def xor_dataset(
    dataset_length: int,
):
    return XORDataset(dataset_length)

@fixture
def data_loader_with_xor_dataset(
    xor_dataset: XORDataset,
):
    return DataLoader(xor_dataset, batch_size=128, shuffle=True)


class TestSimpleClassifier():
    def test_forwarding(
        self,
        simple_classifier: SimpleClassifier,
    ):
        input_tensor = torch.randn(128, 2)
        output_tensor: Tensor = simple_classifier(input_tensor)

        assert output_tensor.shape == Size((128, 1))


class TestXORDataset():
    def test_dataset_length(
        self,
        xor_dataset: XORDataset,
        dataset_length: int,
    ):
        assert len(xor_dataset.data) == dataset_length
        assert len(xor_dataset.label) == dataset_length

    def test_data_size(
        self,
        xor_dataset: XORDataset,
    ):
        for data, label in xor_dataset:
            assert data.size() == Size((2,))
            assert label.size() == Size()


class TestVisualizeSamples():
    def test_figure_parameters(
        self,
        xor_dataset: XORDataset,
    ):
        visualized_fig = visualize_samples(xor_dataset.data, xor_dataset.label)

        assert len(visualized_fig.get_axes()) == 1

        axes = visualized_fig.get_axes()[0]

        assert axes.get_title() == "Dataset samples"
        assert axes.get_xlabel() == r"$x_1$"
        assert axes.get_ylabel() == r"$x_2$"
        assert axes.get_legend() is not None


class TestTrainModel():
    def test_model_parameter_changes(
        self,
        simple_classifier: SimpleClassifier,
        sgd_optimizer: SGD,
        data_loader_with_xor_dataset: DataLoader[tuple[Tensor, Tensor]],
        bce_with_logits_loss_module: BCEWithLogitsLoss,
    ):
        model_before = deepcopy(simple_classifier)
        train_model(
            model=simple_classifier,
            optimizer=sgd_optimizer,
            data_loader=data_loader_with_xor_dataset,
            loss_module=bce_with_logits_loss_module,
            num_epochs=1,
        )
        model_after = simple_classifier

        assert any(not torch.equal(param_before, param_after) for param_before, param_after in zip(model_before.parameters(), model_after.parameters()))


class TestEvalModel():
    def test_eval_model(
        self,
        simple_classifier: SimpleClassifier,
        data_loader_with_xor_dataset: DataLoader[tuple[Tensor, Tensor]],
    ):
        accuracy = eval_model(
            model=simple_classifier,
            data_loader=data_loader_with_xor_dataset,
        )

        assert accuracy >= 0.0
        assert accuracy <= 1.0


class TestVisualizeClassification():
    def test_figure_parameters(
        self,
        simple_classifier: SimpleClassifier,
        xor_dataset: XORDataset,
    ):
        visualized_fig = visualize_classification(
            model=simple_classifier,
            data=xor_dataset.data,
            label=xor_dataset.label,
        )

        assert len(visualized_fig.get_axes()) == 1

        axes = visualized_fig.get_axes()[0]

        assert axes.get_title() == "Dataset samples"
        assert axes.get_xlabel() == r"$x_1$"
        assert axes.get_ylabel() == r"$x_2$"
        assert axes.get_legend() is not None
