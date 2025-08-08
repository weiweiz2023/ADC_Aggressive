import numpy as np
import matplotlib.pyplot as plt


def plot_hists():
    # version 2 captures before the round
    files = [
       
        "/home/weiweiz/Documents/WW_02/ADC_aggressive/saved/hist_csvs/test_filter_input.csv",
        "/home/weiweiz/Documents/WW_02/ADC_aggressive/saved/hist_csvs/test_filter_output.csv",
        #"/home/weiweiz/Documents/WW_02/ADC_aggressive/saved/hist_csvs/test_nofilter_input.csv",
       # "/home/weiweiz/Documents/WW_02/ADC_aggressive/saved/hist_csvs/test_nofilter_output.csv",
        
    ]
    names = ["gradientfilter_ADC-input", " gradientfilter_ADC-Output",
        #     "ADC-input", " ADC-Output",
             
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
        axs[i].set_xlabel("Value")   
        axs[i].set_ylabel("Frequency")  

    plt.show()
    fig.savefig('./saved/hist_bad2.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    plot_hists()
