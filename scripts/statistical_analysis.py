from statsmodels.stats.multicomp import pairwise_tukeyhsd
from data_cleaning import generate_data
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from scipy import stats


def one_sample_ttest(data, column):
    t_stat, p_value = stats.ttest_1samp(data[column], 0)
    return t_stat, p_value


def one_way_anova(data, dv, between):
    formula = f"{dv} ~ C({between})"
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


def eta_squared(aov):
    aov['eta2'] = aov['sum_sq'] / aov['sum_sq'].sum()
    return aov


def main() -> None:
    df = generate_data()
    outcomes = ['HADS_diff', 'CDS_diff', 'PSQI_diff']
    groups = df['group'].unique()

    for outcome in outcomes:
        print(f"\nOne-sample t-tests for {outcome}:")
        for group in groups:
            group_data = df[df['group'] == group]
            t_stat, p_value = one_sample_ttest(group_data, outcome)
            print(
                f"Group {group}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    for outcome in outcomes:
        print(f"\nOne-way ANOVA for {outcome}:")
        anova_result = one_way_anova(df, outcome, 'group')
        print(anova_result)

    for outcome in outcomes:
        print(f"\nTukey's HSD for {outcome}:")
        tukey_result = pairwise_tukeyhsd(df[outcome], df['group'])
        print(tukey_result)

    for outcome in outcomes:
        print(f"\nEffect sizes for {outcome}:")
        anova_result = one_way_anova(df, outcome, 'group')
        effect_sizes = eta_squared(anova_result)
        print(effect_sizes['eta2'])

    correlation_matrix = df[outcomes].corr()
    print("\nCorrelation matrix between outcomes:")
    print(correlation_matrix)


if __name__ == "__main__":
    main()
