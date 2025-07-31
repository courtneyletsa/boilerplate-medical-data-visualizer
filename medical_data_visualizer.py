import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')
df['height'] = df['height']/100
df['bmi'] = round(df['weight']/(df['height']**2),2)

# 2
df['overweight'] = df['bmi'].apply(lambda x: 1 if x > 25 else 0)

# 3
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
# df['cholesterol']
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    # 2. Group and reformat the data to split it by 'cardio' and show the counts of each feature
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7
    # 3. Draw the catplot using seaborn
    cat_plot = sns.catplot(
        data=df_cat,
        kind='bar',
        x='variable',
        y='total',
        hue='value',
        col='cardio'
    )

    # 4. Adjust the plot layout
    cat_plot.set_axis_labels("variable", "total")
    cat_plot.set_titles("cardio = {col_name}")
    cat_plot._legend.set_title("value")

    plt.tight_layout()
    plt.show()


    # 8
    # fig = None


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 2. Select only numeric columns for correlation
    numeric_df = df_heat.select_dtypes(include=['int64', 'float64'])

    # 12
    corr = numeric_df.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        cmap="rocket"
    )

    plt.tight_layout()
    plt.show()


    # 16
    fig.savefig('heatmap.png')
    return fig
