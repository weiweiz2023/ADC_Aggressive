import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def plot_graph():
    # version 2 captures before the round
    files = [
        "./saved/logs/testADC_NoLoss_quant_resnet20_0_0_0_0_4_[False, False]_False_64_stochastic_MNIST_20_0.01_128_0.9_0.0001.txt",
        "./saved/logs/testADC_Loss_quant_resnet20_0_0_0_0_4_[False, False]_False_64_stochastic_MNIST_20_0.01_128_0.9_0.0001.txt",
        "./saved/logs/testADC_LossGrad_quant_resnet20_0_0_0_0_4_[False, False]_False_64_stochastic_MNIST_20_0.01_128_0.9_0.0001.txt",
        "./saved/logs/testADC_LossGradNoScale_quant_resnet20_0_0_0_0_4_[False, False]_False_64_stochastic_MNIST_20_0.01_128_0.9_0.0001"
    ]    

    N = len(files)
    fig, axs = plt.subplots(2, 1, figsize=(5 * 2, 8))
    arrays = [np.genfromtxt(file, delimiter=", ", dtype="float", skip_header=0) for file in files]

    # names=['epoch', 'loss', 'train acc', 'test acc']
    labels = ["Vanilla", "Loss", "Loss + Grad", "Loss + Grad - Scale"]
    colors = ["royalblue", "orangered", "mediumorchid", "forestgreen", "maroon", "saddlebrown", "slategray", "gold"]
    
    for i, array in enumerate(arrays):
        axs[0].plot(array[:, 0], array[:, 3], color=colors[i], label=labels[i], marker='o', markersize=8, linewidth=2)
        axs[1].plot(array[:, 0], array[:, 1], color=colors[i], label=labels[i], marker='^', markersize=8, linewidth=2)

    axs[0].set_ylim(93, 100)
    axs[0].set_ylabel('Accuracy %')
    axs[0].set_title('Comparison of Model Accuracy')
    
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('Loss Value')
    axs[1].set_title('Comparison of Model Loss')
    
    # Major ticks every 5, minor ticks every 1
    major_ticks = np.arange(1, 20 + 1, 5)
    minor_ticks = np.arange(1, 20 + 1, 1)
    for i in range(2):
        axs[i].set_xticks(major_ticks)
        axs[i].set_xticks(minor_ticks, minor=True)
        axs[i].grid(which='both')
        axs[i].grid(which='minor', alpha=0.2)
        axs[i].grid(which='major', alpha=0.5)
        axs[i].grid(True)
        axs[i].legend()
    
    plt.xlabel('Epoch')
    plt.show()
    fig.savefig('./saved/comparison_plot4.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    plot_graph()
