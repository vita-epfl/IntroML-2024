from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(
    df: pd.DataFrame,
    label: str,
    train_size: float = 0.6,
    val_size: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[int, str],
]:
    """Transforms data into numpy arrays and splits it into a train, val and test set

    Args:
        df: Data to split
        label: name of the training label
        train_size: proportion of the data used for training
        val_size: proportion of the data used for validation
        seed: random seed

    Returns:
        object: Tuple containing the training features, training label,
            validation features, validation label, test features, test label,
            names of the features and map from label to label_name
    """

    df = df.sort_values(by=label)
    df[label] = df[label].astype("category")

    df = df.sample(frac=1, random_state=seed)
    train, val, test = (
        df[: int(len(df) * train_size)],
        df[int(len(df) * train_size) : int(len(df) * (train_size + val_size))],
        df[int(len(df) * (train_size + val_size)) :],
    )

    X_train = train.drop(columns=label).to_numpy()
    X_val = val.drop(columns=label).to_numpy()
    X_test = test.drop(columns=label).to_numpy()

    y_train = train[label].cat.codes.to_numpy()
    y_val = val[label].cat.codes.to_numpy()
    y_test = test[label].cat.codes.to_numpy()

    label_map = dict(enumerate(df[label].cat.categories))
    feature_names = list(df.drop(columns=label).columns)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, label_map


def plot_labeled(
    X: np.ndarray,
    y: np.ndarray,
    label_map: Dict[int, str],
    feature_names: List[str],
    figsize: Tuple[float, float] = (6, 6),
    title: Optional[str] = None,
):
    """Plots labeled data

    Supports only 2D features and up to 10 different classes

    Args:
        X: features
        y: labels
        label_map: Dictionary mapping class number to class name
        feature_names: List of feature names
        figsize: Figure size
        title: The plot title
    """
    # Support up to 10 different classes
    markers = ["o", "v", "s", "8", "p", "x", "D", "<", "^", ">"]
    colors = (
        # Default color palette if 3 classes, otherwise use another color palette
        ([0, 0.5, 0], [0.25, 0.25, 1], [0.85, 0.85, 0])
        if len(label_map) == 3
        else sns.color_palette(n_colors=len(label_map))
    )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    for i, class_name in label_map.items():
        ax.scatter(
            x=X[y == i, 0],
            y=X[y == i, 1],
            color=colors[i],
            marker=markers[i],
            label=class_name,
            alpha=0.3,
            s=70,
        )

    plt.axis("equal")
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.legend(prop={"size": 12})
    if title is not None:
        plt.title(title)

    plt.show()


def plot_unlabeled(
    X: np.ndarray,
    feature_names: List[str],
    figsize: Tuple[float, float] = (6, 6),
    title: Optional[str] = None,
):
    """Plots unlabeled data

    Supports only 2D features

    Args:
        X: features
        feature_names: List of feature names
        figsize: Figure size
        title: The plot title
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.scatter(
        x=X[:, 0],
        y=X[:, 1],
        color="grey",
        marker="o",
        label="Unknown",
        alpha=0.3,
        s=70,
    )

    plt.axis("equal")
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.legend(prop={"size": 12})

    if title is not None:
        plt.title(title)

    plt.show()


def plot_nearest_neighbors(
    sample: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    neighbor_indices: np.ndarray,
    label_map: Dict[int, str],
    feature_names: List[str],
    figsize: Tuple[float, float] = (6, 6),
    title: Optional[str] = None,
):
    """

    Args:
        sample: Sample of shape (2, )
        X:  Dataset of shape (N, 2)
        y: Labels of shape (N, )
        neighbor_indices: Indices of k-nearest neighbors
        label_map: Dictionary mapping class number to class name
        feature_names: List of feature names
        figsize: Figure size
        title: The plot title
    """
    # Support up to 10 different classes
    markers = ["o", "v", "s", "8", "p", "x", "D", "<", "^", ">"]
    colors = (
        # Default color palette if 3 classes, otherwise use another color palette
        ([0, 0.5, 0], [0.25, 0.25, 1], [0.85, 0.85, 0])
        if len(label_map) == 3
        else sns.color_palette(n_colors=len(label_map))
    )

    # Plot all points
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    for i, class_name in label_map.items():
        ax.scatter(
            x=X[y == i, 0],
            y=X[y == i, 1],
            color=colors[i],
            marker=markers[i],
            label=class_name,
            alpha=0.3,
            s=70,
        )

    # Plot nearest neighbors
    for i, class_name in label_map.items():
        class_indices = neighbor_indices[y[neighbor_indices] == i]
        if len(class_indices) > 0:
            ax.scatter(
                x=X[class_indices][:, 0],
                y=X[class_indices][:, 1],
                color=colors[i],
                marker=markers[i],
                alpha=0.8,
                s=80,
            )

    # Plot sample
    ax.scatter(
        x=sample[0],
        y=sample[1],
        marker="*",
        color="#CCCC00",
        alpha=1,
        s=200,
        label="Unknown",
        edgecolors="#000000",
        linewidths=1,
    )

    plt.axis("equal")
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.legend(prop={"size": 12})
    if title is not None:
        plt.title(title)

    plt.show()


def plot_predictions(
    samples: np.ndarray,
    predicted_labels: np.ndarray,
    X: Optional[np.ndarray],
    y: Optional[np.ndarray],
    label_map: Dict[int, str],
    feature_names: List[str],
    figsize: Tuple[float, float] = (6, 6),
    title: Optional[str] = None,
):
    """

    Args:
        samples: Sample of shape (M, 2)
        predicted_labels: Labels of shape (M, )
        X:  Training dataset of shape (N, 2) - Set to None to avoid plotting them
        y: Training labels of shape (N, ) - Set to None to avoid plotting them
        label_map: Dictionary mapping class number to class name
        feature_names: List of feature names
        figsize: Figure size
        title: The plot title
    """
    # Support up to 10 different classes
    markers = ["o", "v", "s", "8", "p", "x", "D", "<", "^", ">"]
    colors = (
        # Default color palette if 3 classes, otherwise use another color palette
        ([0, 0.5, 0], [0.25, 0.25, 1], [0.85, 0.85, 0])
        if len(label_map) == 3
        else sns.color_palette(n_colors=len(label_map))
    )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Plot training labels
    if X is not None and y is not None:
        for i, class_name in label_map.items():
            ax.scatter(
                x=X[y == i, 0],
                y=X[y == i, 1],
                color=colors[i],
                marker=markers[i],
                alpha=0.2,
                s=50,
            )

    # Plot predictions
    for i, class_name in label_map.items():
        ax.scatter(
            x=samples[predicted_labels == i, 0],
            y=samples[predicted_labels == i, 1],
            color=colors[i],
            marker="*",
            label=class_name,
            alpha=0.9,
            s=140,
        )

    plt.axis("equal")
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.legend(prop={"size": 12})
    if title is not None:
        plt.title(title)

    plt.show()
