from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def normalize(X, mean, std):
    """Normalization of array
    Args:
       X (np.array): Dataset of shape (N, D)
       mean (np.array): Mean of shape (D, )
       std (float): Standard deviation of shape(D, )
    """
    return (X - mean) / std

def insert_offset(X):
    """ Adds an offset to X data
    """
    if (X[:,0]== np.ones(X.shape[0])).all():
        print( 'Your X data already has an offset vector')
    else:
        X = np.insert(X, 0, 1, axis=1)
    return X
    
def preprocess_data(
    df: pd.DataFrame,
    label: str,
    train_size: float = 0.6,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[int, str],]:
    """Transforms data into numpy arrays and splits it into a train and test set

    Args:
        df: Data to split
        label: name of the training label
        train_size: proportion of the data used for training
        seed: random seed

    Returns:
        object: Tuple containing the training features, training label,
        test features, test label, names of the features and map from label to label_name
    """

    df = df.sort_values(by=label)
    df[label] = df[label].astype("category")

    df = df.sample(frac=1, random_state=seed)
    train, test = (
        df[: int(len(df) * train_size)],
        df[int(len(df) * train_size) :],
    )

    X_train = train.drop(columns=label).to_numpy()
    X_test = test.drop(columns=label).to_numpy()

    y_train = pd.get_dummies(train[label]).to_numpy()
    y_test = pd.get_dummies(test[label]).to_numpy()

    label_map = dict(enumerate(df[label].cat.categories))
    feature_names = list(df.drop(columns=label).columns)

    return X_train, y_train, X_test, y_test, feature_names, label_map


def plot_boundaries(X, y, w, output_func, class_names, ax_titles=None, train=True):
    """ The current code works for 2D features of X, skipping the first intercept dimension. Y must also be ONE-HOT """
    markers = ["o", "v"]
    colors = ([0, 0.5, 0], [0.25, 0.25, 1])
    eps = 1e-6

    # Plot when normalized
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Plot decision boundary
    x0_min, x0_max = X[:, 1].min(), X[:, 1].max()
    x1_min, x1_max = X[:, 2].min(), X[:, 2].max()
    x0_diff = x0_max - x0_min
    x1_diff = x1_max - x1_min

    xx, yy = np.mgrid[
        x0_min - x0_diff / 10 : x0_max + x0_diff / 10 + eps : x0_diff / 50,
        x1_min - x1_diff / 10 : x1_max + x1_diff / 10 + eps : x1_diff / 50,
    ]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = output_func(np.insert(grid,0,1,axis=1), w)[:,1].reshape(xx.shape)

    contour = ax.contourf(xx, yy, probs, 25, cmap="GnBu", vmin=0, vmax=1, alpha=1)
    ax_c = fig.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks(np.arange(0, 1.01, 0.1))

    # End of plotting decision boundary

    for i, (class_name, marker, color) in enumerate(zip(class_names, markers, colors)):
        ax.scatter(
            x=X[y[:,1].squeeze() == i][:, 1],
            y=X[y[:,1].squeeze() == i][:, 2],
            color=color,
            marker=marker,
            label=class_name,
            alpha=1,
            s=100,
            edgecolors="#FFFFFF",
            linewidths=1,
        )

    ax.set_aspect(1)
    ax.set_xlim([x0_min - x0_diff / 10, x0_max + x0_diff / 10])
    ax.set_ylim([x1_min - x1_diff / 10, x1_max + x1_diff / 10])

    if train:
        plt.title("Training set ({} examples)".format(len(X)), fontsize=16)
    else:
        plt.title("Test set ({} examples)".format(len(X)), fontsize=16)
    if ax_titles is not None:
        plt.xlabel(ax_titles[0], fontsize=14)
        plt.ylabel(ax_titles[1], fontsize=14)

    plt.legend(prop={"size": 14}, loc="best")
    plt.show()


def interactive_boundaries(
    X_train,
    y_train,
    X_test,
    y_test,
    w_list,
    output_func,
    class_names,
    ax_titles=None,
    total_steps=50,
):
    """
    Plots interactive boundaries in a binary logistic regression setting using Plotly
    """

    eps = 1e-6

    total_steps = min(total_steps, len(w_list))
    colors = ["rgb(0,127,0)", "rgb(64,64,255)"]
    X = np.concatenate((X_train, X_test))

    # Create a mesh grid on which we will run our model
    x0_min, x0_max = X[:, 1].min(), X[:, 1].max()
    x1_min, x1_max = X[:, 2].min(), X[:, 2].max()
    x0_diff = x0_max - x0_min
    x1_diff = x1_max - x1_min
    x0_range = np.arange(
        x0_min - x0_diff / 10, x0_max + x0_diff / 10 + eps, x0_diff / 50
    )
    x1_range = np.arange(
        x1_min - x1_diff / 10, x1_max + x1_diff / 10 + eps, x1_diff / 50
    )

    xx, yy = np.meshgrid(x0_range, x1_range)
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Initialize Plotly figure
    fig = go.Figure()

    linspace = np.linspace(0, len(w_list) - 1, num=total_steps, dtype=int)

    # Plot decision boundaries
    for i in linspace:
        probs = output_func(np.insert(grid,0,1,axis=1), w_list[i])[:,1].reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                x=x0_range,
                y=x1_range,
                z=probs,
                showscale=True,
                colorscale="GnBu",
                opacity=0.8,
                hoverinfo="skip",
                visible=False,
            )
        )

    # Plot points
    trace_specs = [
        [X_train, y_train, 0, "Train", "circle", colors[0]],
        [X_train, y_train, 1, "Train", "triangle-down", colors[1]],
        [X_test, y_test, 0, "Test", "circle-dot", colors[0]],
        [X_test, y_test, 1, "Test", "triangle-down-dot", colors[1]],
    ]

    for X, y, label, split, marker, color in trace_specs:
        fig.add_trace(
            go.Scatter(
                x=X[y[:,1] == label, 1],
                y=X[y[:,1] == label, 2],
                name=f"{class_names[label]}, {split}",
                mode="markers",
                marker=dict(
                    size=12,
                    symbol=marker,
                    color=color,
                    line=dict(width=1, color="White"),
                ),
            )
        )

    # Add slider
    steps = []
    for i in range(len(linspace)):
        step = dict(
            method="update",
            label=f"{linspace[i]}",
            args=[
                {"visible": [False] * len(linspace) + [True] * 4}
            ],  # last 4 traces are the points, always show them
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(active=len(linspace), currentvalue={"prefix": "iteration: "}, steps=steps)
    ]

    # Customize layout
    fig.update_layout(
        sliders=sliders,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    if ax_titles is not None:
        fig.update_layout(xaxis_title=ax_titles[0], yaxis_title=ax_titles[1])

    fig["layout"].update(autosize=False, width=600, height=600)

    fig.show()
