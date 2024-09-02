from data_cleaning import generate_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    data = generate_data()

    grouped_data = data.groupby('group').mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.2
    r1 = np.arange(len(grouped_data))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    ax.bar(r1, grouped_data['HADS_diff'],
           width=bar_width, label='Anxiety and Depresssion', color='skyblue')
    ax.bar(r2, grouped_data['CDS_diff'], width=bar_width,
           label='Cardiac Depression', color='lightgreen')
    ax.bar(r3, grouped_data['PSQI_diff'],
           width=bar_width, label='Sleep Quality', color='salmon')

    ax.set_xlabel('Treatment Group', fontweight='bold')
    ax.set_ylabel('Mean Difference', fontweight='bold')
    ax.set_title(
        'Comparison of Feature Differences Across Treatment Groups', fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(len(grouped_data))])
    ax.set_xticklabels(['Rosemary', 'Lavender', 'Both', 'Placebo'])

    ax.legend()

    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('../figures/mean_feature_diffs.png')

    columns_of_interest = ['HADS_diff', 'CDS_diff', 'PSQI_diff']

    corr_matrix = data[columns_of_interest].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Age and Treatment Outcomes')
    plt.tight_layout()
    plt.savefig('../figures/feature_diff_heatmap.png')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    outcomes = ['HADS_diff', 'CDS_diff', 'PSQI_diff']

    for i, outcome in enumerate(outcomes):
        sns.violinplot(x='sex', y=outcome, hue='group',
                       data=data, split=True, ax=axes[i])
        axes[i].set_title(f'{outcome} by Sex and Treatment Group')
        axes[i].set_xlabel('Sex')
        axes[i].set_ylabel('Difference')

    plt.tight_layout()
    plt.savefig('../figures/sex_outcomes.png')

    variables = ['group', 'age', 'education', 'HADS_diff', 'CDS_diff', 'PSQI_diff']

    sns.pairplot(data[variables], hue='group', height=2.5)
    plt.suptitle('Scatter Plot Matrix of Age, Education, and Treatment Outcomes', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/scatterplot.png')



if __name__ == "__main__":
    main()
