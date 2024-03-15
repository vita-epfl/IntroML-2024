from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def preprocess_data(
    df: pd.DataFrame,
    label: str,
    train_size: float = 0.6,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Transforms data into numpy arrays and splits it into a train and test set

    Args:
        df: Data to split
        label: name of the training label, values of column should be numerical
        train_size: proportion of the data used for training
        val_size: proportion of the data used for validation
        seed: random seed
        categorical_label: whether the label is categorical or not

    Returns:
        object: Tuple containing the training features, training label,
            test features, test label and names of the features
    """

    df = df.sort_values(by=label)

    df = df.sample(frac=1, random_state=seed)
    train, test = (df[: int(len(df) * train_size)], df[int(len(df) * train_size) :])

    X_train = train.drop(columns=label).to_numpy()
    X_test = test.drop(columns=label).to_numpy()

    y_train = train[label].to_numpy()
    y_test = test[label].to_numpy()

    feature_names = list(df.drop(columns=label).columns)

    return X_train, y_train, X_test, y_test, feature_names


def plot_data_3d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    feature_names: Tuple[str, str] = ("x1", "x2"),
    label_name: str = "y",
) -> None:
    """Plots the 2-Dimensional data

    Args:
        X_train: Training data (including constant term) of shape (N, 3)
        y_train: Training labels of shape (N, )
        X_test: Test data (including constant term) of shape (M, 3)
        y_test: Test labels of shape (M, )
        feature_names: names of features
        label_name: name of label

    """
    
     # Remove constant term
    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X_train[:, 0],
                y=X_train[:, 1],
                z=y_train,
                mode="markers",
                name="train",
                marker=dict(size=5, opacity=1.0),
            )
        ]
    )

    if (X_test is not None) and (y_test is not None):
        fig.add_trace(
            go.Scatter3d(
                x=X_test[:, 0],
                y=X_test[:, 1],
                z=y_test,
                mode="markers",
                name="test",
                marker=dict(size=5, color="green", opacity=1.0),
            )
        )

    fig.update_layout(
        autosize=True,
        width=500,
        height=500,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            zaxis_title=label_name,
        ),
    )

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    fig.show()


def plot_surface_3d(
    w: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    feature_names: Tuple[str, str] = ("x1", "x2"),
    label_name: str = "y",
) -> None:
    """Plots the 2-Dimensional data

    Args:
        w: Weights of shape (3,)
        X_train: Training data (including constant) of shape (N, 3)
        y_train: Training labels of shape (N, )
        X_test: Test data (including constant) of shape (M, 3)
        y_test: Test labels of shape (M, )
        feature_names: names of features
        label_name: name of label

    """
    
    # Remove constant term
    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]

    eps = 1e-6

    if (X_test is not None) and (y_test is not None):
        X = np.concatenate((X_train, X_test))

    else:
        X = X_train

    x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x0_diff = x0_max - x0_min
    x1_diff = x1_max - x1_min

    xrange = np.arange(x0_min - x0_diff / 10, x0_max + x0_diff / 10 + eps, x0_diff / 50)
    yrange = np.arange(x1_min - x1_diff / 10, x1_max + x1_diff / 10 + eps, x1_diff / 50)

    xx, yy = np.meshgrid(xrange, yrange)
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = (grid @ w[1:] + w[0]).reshape(xx.shape)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X_train[:, 0],
                y=X_train[:, 1],
                z=y_train,
                mode="markers",
                name="train",
                marker=dict(size=5, opacity=1.0),
            )
        ]
    )

    if (X_test is not None) and (y_test is not None):
        fig.add_trace(
            go.Scatter3d(
                x=X_test[:, 0],
                y=X_test[:, 1],
                z=y_test,
                mode="markers",
                name="test",
                marker=dict(size=5, color="green", opacity=1.0),
            )
        )

    fig.add_trace(
        go.Surface(x=xrange, y=yrange, z=preds, name="pred_surface", opacity=0.7)
    )

    fig.update_layout(
        autosize=True,
        width=500,
        height=500,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            zaxis_title=label_name,
        ),
    )

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    fig.show()


def plot_loss(loss_list):
    plt.figure(figsize=(6, 6))
    step = np.arange(1, len(loss_list) + 1)
    plt.plot(step, loss_list)
    plt.title("Evolution of the loss during the training")
    plt.xlabel("iteration")
    plt.ylabel("Loss")
    plt.show()


def plot_linear_regression_2d(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    feature_name: str = "x",
    label_name: str = "y",
) -> None:
    """Plots simple linear regression

    Args:
        X: Dataset of shape (N, D)
        y: Labels of shape (N, )
        w: Weights of shape (D, )
        feature_name: Name of feature
        label_name: Name of label

    Returns:
        None
    """

    feature = X[:, 1]

    x = np.linspace(feature.min(), feature.max(), 100)
    f = w[0] + (w[1] * x)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, f, "r", label="Prediction")
    ax.scatter(feature, y, label="Training Data")
    ax.legend(loc=2)
    ax.set_xlabel(feature_name)
    ax.set_ylabel(label_name)
