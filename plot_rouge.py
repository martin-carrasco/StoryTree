import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ROBERTA = 'scoring_roberta_1.csv'
NORMAL = 'scoring_roberta_base_1.csv'
EDUS = 'scoring_roberta_edus_1.csv'

df_roberta = pd.read_csv(ROBERTA)
df_normal = pd.read_csv(NORMAL)
df_edus = pd.read_csv(EDUS)


df_roberta['Label'] = df_roberta['Label'].str.replace('\n', ' ')
df_normal['Label'] = df_normal['Label'].str.replace('\n', ' ')

df_roberta = df_roberta[df_roberta['Label'].isin(df_edus['Label'])]
df_normal= df_normal[df_normal['Label'].isin(df_edus['Label'])]

df_roberta['Model'] = 'RoBERTa FT'
df_normal['Model'] = 'RoBERTa'
df_edus['Model'] = 'RoBERTa FT + EDUs'

df = pd.concat([df_normal, df_roberta, df_edus])
df.to_csv('scoring_3_1_edus.csv')


for summary_technique in list(df['Method'].unique()):
    c_df = df[df['Method'] == summary_technique]

    c_df = c_df.melt(id_vars='Model', value_vars=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], var_name='Metric', value_name='Score')

    sns.boxplot(data=c_df, x='Metric', y='Score', hue='Model')
    # Mean of L
    plt.axhline(y = 0.189, color = 'r', linestyle = '-', label='Max RL') 
    # Mean of R2
    plt.axhline(y = 0.103, color = 'b', linestyle = '-', label='Max R2') 
    #Mean of R1
    plt.axhline(y = 0.319, color = 'g', linestyle = '-', label='Max R1') 
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center') 

    plt.title(f'RoBERTa vs RoBERTa FT vs RoBERTa FT + EDUs: 22 samples on {summary_technique}')
    plt.savefig(f'img_comp_{summary_technique}.png')
    plt.clf()