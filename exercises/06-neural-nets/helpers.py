from typing import Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt


def imshow(img: torch.Tensor) -> None:
    fig, ax = plt.subplots()
    ax.imshow(to_np_img(img), cmap="gray")
    ax.axis("off")
    plt.show()


def to_np_img(img: torch.Tensor) -> np.ndarray:
    return np.transpose(img.numpy(), (1, 2, 0)).squeeze()


def view_prediction(
    img: torch.Tensor,
    pred: torch.Tensor,
    classes: Iterable = range(10),
) -> None:
    """Shows prediction for MNIST style datasets (with 10 classes)

    Args:
        img: image to display (as tensor)
        pred: model prediction
        classes: class names (of size 10)
    """
    pred = pred.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 7), ncols=2)
    plt.subplots_adjust(wspace=0.4)
    ax1.imshow(to_np_img(img), cmap="gray")
    ax1.axis("off")
    ax2.barh(np.arange(10), pred)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes)
    ax2.set_xlim(0, 1.1)
    ax2.set_title("Prediction")
    plt.show()
