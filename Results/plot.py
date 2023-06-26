import matplotlib.pyplot as plt
import pandas as pd

ax = ['0','5','10','20','30']
exp1 = ['balance-scale', 'car', 'cmc', 'dermatology', 'german-credit', 'german-credit-disc', 'glass', 'glass-disc', 'haberman', 'iris', 'lymphography', 'nursery', 'wine', 'wine-disc', 'wdbc', 'wdbc-disc', 'zoo']
exp2 = ['hepatitis', 'hepatitis-disc', 'horse-colic', 'horse-colic-disc', 'house-votes', 'soybean']

def plot_classical_all_data(ax, exp1):
    cols = 3
    rows = 6

    fig, axs = plt.subplots(rows, cols)

    for i in range(len(exp1)):
        data = pd.read_excel(f'C:/Users/s164389/Desktop/Afstuderen/Thesis/Results/{exp1[i]}_results.xlsx')
        data = data.to_numpy().T
        a = i // cols
        b = i % cols
        axs[a, b].plot(ax, data[1], label='NBC')
        axs[a, b].plot(ax, data[5], label='C4.5')
        axs[a, b].plot(ax, data[7], label='SPN')
        axs[a, b].set_title(exp1[i])
        axs[a, b].set_ylim([0,100])

    for ax in axs.flat:
        ax.set(xlabel='missingness (%)', ylabel='accuracy (%)')
    
    fig.delaxes(axs[5][2])

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.subplots_adjust(left=0.30, bottom=0.11, right= 0.70, top=0.88, wspace=0.30, hspace=0.30)

    fig.legend(labels=['NBC', 'C4.5', 'SPN'], loc='upper center', ncol=3)
    plt.show()

def plot_robust_all_data(ax, exp1):
    cols = 3
    rows = 6

    fig, axs = plt.subplots(rows, cols)

    for i in range(len(exp1)):
        data = pd.read_excel(f'C:/Users/s164389/Desktop/Afstuderen/Thesis/Results/{exp1[i]}_results.xlsx')
        data = data.to_numpy().T
        a = i // cols
        b = i % cols
        axs[a, b].plot(ax, data[2], label='NCC low')
        axs[a, b].plot(ax, data[4], label='NCC robust')
        axs[a, b].plot(ax, data[8], label='CSPN low')
        axs[a, b].plot(ax, data[10], label='CSPN robust')
        axs[a, b].set_title(exp1[i])
        axs[a, b].set_ylim([0,100])

    for ax in axs.flat:
        ax.set(xlabel='missingness (%)', ylabel='accuracy (%)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
    fig.delaxes(axs[5][2])

    plt.subplots_adjust(left=0.30, bottom=0.11, right= 0.70, top=0.88, wspace=0.30, hspace=0.30)

    fig.legend(labels=['NCC low', 'NCC robust', 'CSPN low', 'CSPN robust'], loc='upper center', ncol=4)
    plt.show()


def plot_naive(ax, exp1):
    cols = 3
    rows = 6

    fig, axs = plt.subplots(rows, cols)

    for i in range(len(exp1)):
        data = pd.read_excel(f'C:/Users/s164389/Desktop/Afstuderen/Thesis/Results/{exp1[i]}_results.xlsx')
        data = data.to_numpy().T
        a = i // cols
        b = i % cols
        axs[a, b].plot(ax, data[1], label='NBC')
        axs[a, b].plot(ax, data[2], label='NCC low')
        axs[a, b].plot(ax, data[3], label='NCC high')
        axs[a, b].plot(ax, data[4], label='NCC robust')
        axs[a, b].set_title(exp1[i])
        axs[a, b].set_ylim([0,100])

    for ax in axs.flat:
        ax.set(xlabel='missingness (%)', ylabel='accuracy (%)')
    
    fig.delaxes(axs[5][2])

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.subplots_adjust(left=0.30, bottom=0.11, right= 0.70, top=0.88, wspace=0.30, hspace=0.30)

    fig.legend(labels=['NBC', 'NCC low', 'NCC high', 'NCC robust'], loc='upper center', ncol=4)
    plt.show()

def plot_tree(ax, exp1):
    cols = 3
    rows = 6

    fig, axs = plt.subplots(rows, cols)

    for i in range(len(exp1)):
        data = pd.read_excel(f'C:/Users/s164389/Desktop/Afstuderen/Thesis/Results/{exp1[i]}_results.xlsx')
        data = data.to_numpy().T
        a = i // cols
        b = i % cols
        axs[a, b].plot(ax, data[5], label='C4.5')
        axs[a, b].plot(ax, data[6], label='Credal-C4.5')
        axs[a, b].set_title(exp1[i])
        axs[a, b].set_ylim([0,100])

    for ax in axs.flat:
        ax.set(xlabel='missingness (%)', ylabel='accuracy (%)')
    
    fig.delaxes(axs[5][2])

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.subplots_adjust(left=0.30, bottom=0.11, right= 0.70, top=0.88, wspace=0.30, hspace=0.30)

    fig.legend(labels=['C4.5', 'Credal-C4.5'], loc='upper center', ncol=2)
    plt.show()

def plot_spn(ax, exp1):
    cols = 3
    rows = 6

    fig, axs = plt.subplots(rows, cols)

    for i in range(len(exp1)):
        data = pd.read_excel(f'C:/Users/s164389/Desktop/Afstuderen/Thesis/Results/{exp1[i]}_results.xlsx')
        data = data.to_numpy().T
        a = i // cols
        b = i % cols
        axs[a, b].plot(ax, data[7], label='SPN')
        axs[a, b].plot(ax, data[8], label='CSPN low')
        axs[a, b].plot(ax, data[9], label='CSPN high')
        axs[a, b].plot(ax, data[10], label='CSPN robust')
        axs[a, b].set_title(exp1[i])
        axs[a, b].set_ylim([0,100])

    for ax in axs.flat:
        ax.set(xlabel='missingness (%)', ylabel='accuracy (%)')
    
    fig.delaxes(axs[5][2])

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.subplots_adjust(left=0.30, bottom=0.11, right= 0.70, top=0.88, wspace=0.30, hspace=0.30)

    fig.legend(labels=['SPN', 'CSPN low', 'CSPN high', 'CSPN robust'], loc='upper center', ncol=4)
    plt.show()

plot_robust_all_data(ax, exp1)