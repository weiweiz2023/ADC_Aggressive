import numpy as np
import matplotlib.pyplot as plt


def plot_hists():
    # version 2 captures before the round
    files = [
        "./saved/hist_csvs/testNoLoss_scaled.csv",
        "./saved/hist_csvs/testNoLoss_output.csv",
        "./saved/hist_csvs/testLoss_scaled.csv",
        "./saved/hist_csvs/testLoss_output.csv",
        "./saved/hist_csvs/testLossGrad_scaled.csv",
        "./saved/hist_csvs/testLossGrad_output.csv",
        "./saved/hist_csvs/testLossGradNoScale_input.csv",
        "./saved/hist_csvs/testLossGradNoScale_output.csv",
    ]
    names = ["Vanilla Scaled", "Vanilla ADC-Out",
             "Loss Scaled", "Loss ADC Out",
             "Loss + Grad Scaled", "Loss + Grad ADC-Out",
             "Loss + Grad - Scale Scaled", "Loss + Grad - Scale ADC-Out",
             ]

    n = len(files)
    fig, axs = plt.subplots(n, 1, figsize=(8, 5 * n))

    for i, file in enumerate(files):
        print(f"Plotting {file}")
        array = np.genfromtxt(file, delimiter=",")
        # if i == 0:
        array = np.clip(array, -10, 10)  # Clip the first array to avoid outliers
        # counts, bins = np.histogram(array, bins=100)
        axs[i].hist(array, bins=100)
        axs[i].set_title(files[i] if len(names) == 0 else names[i])

    plt.show()
    fig.savefig('./saved/hist_bad2.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    plot_hists()
