import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Source: https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 250)
    y = np.linspace(ylim[0], ylim[1], 250)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    preds = model.predict(xy).reshape(X.shape)
    
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    pcol = plt.pcolormesh(X, Y, preds,  cmap="viridis", vmin=-1, vmax=1, alpha=0.03, shading='nearest', antialiased=True)
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, edgecolors="black", facecolors="none")
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
