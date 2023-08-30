import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import radarplot

def boxplot_overall(df_overall,resolution_string):
    palette = ['plum', 'g', 'orange']
    ax = sns.boxplot(data=df_overall[['nextqsm', 'ground', 'tgv']], fliersize=0, palette = palette)
    plt.xticks(ticks=[0, 1, 2], labels=['nextqsm', 'ground', 'tgv'])
    ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=45)
    ax.set_xlabel("Region of interest")
    ax.set_ylabel("Susceptibility (ppm)")
    ax.set_title(resolution_string)
    ax.set_ylim(-0.2, 0.2)
    plt.show()
    return 0

def boxplot_roi(melted_df_roi,resolution_string):
    fig = plt.figure()
    palette = ['plum', 'g', 'orange']
    ax = sns.boxplot(data=melted_df_roi, x='seg', y='Susceptibility', hue='Algorithms', fliersize=0, palette = palette)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=45)
    ax.set_xlabel("Region of interest")
    ax.set_ylabel("Susceptibility (ppm)")
    ax.set_title(resolution_string)
    ax.set_ylim(-1, 1)
    #plt.tight_layout()
    plt.show()
    return 0
    
def plot_radar(dfmet,resolution_string):
    colors = ['#1f77b4',
              '#aec7e8',
              '#ff7f0e',
              '#ffbb78',
              '#2ca02c',
              '#98df8a',
              '#d62728',
              '#ff9896',
              '#9467bd',
              '#c5b0d5',
              '#8c564b',
              '#c49c94',
              '#e377c2',
              '#f7b6d2',
              '#7f7f7f',
              '#c7c7c7',
              '#bcbd22',
              '#dbdb8d',
              '#17becf',
              '#9edae5']
    # Create the radar plot
    N = len(dfmet.columns)
    theta = radarplot.radar_factory(N, frame='polygon')
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    colors = ['b', 'r']
    labels = list(dfmet.keys())
    
    # Loop through the datasets and plot them on the same radar plot
    data = dfmet.values
    names = dfmet.index
    for d, name, color in zip(data, names, colors[:2]):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25, label=name)

    # Set variable labels for the radar plot
    ax.set_varlabels(labels)

    # Add a legend
    legend = ax.legend(loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    # Add a title
    ax.set_title(resolution_string)
    fig.text(0.5, 0.965, 'Metrics vs. QSM Algorithm',
             horizontalalignment='center', color='black', weight='bold', size='large')

    plt.show()
      
    return 0