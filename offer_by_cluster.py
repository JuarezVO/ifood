import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dataset/full_dataset_clustered.csv",usecols=['offer_code', 'cluster','channels','offer_success','offer_type','amount_per_minvalue'])

x = df.groupby(['cluster', 'offer_code', 'offer_type'])['offer_success'].mean() * 100
x = x.reset_index()

offer_type_colors = {
    'bogo': 'steelblue',
    'discount': 'forestgreen',
    'informational': 'firebrick'
}

fig, ax = plt.subplots(figsize=(10, 5))

x_pos = np.arange(len(x['cluster'].unique()))
offsets = np.linspace(-0.3, 0.3, len(x['offer_code'].unique()))

for i, (offer_code, group) in enumerate(x.groupby('offer_code')):
    offer_type = group['offer_type'].iloc[0]
    color = offer_type_colors[offer_type]
    ax.bar(x_pos + offsets[i], group['offer_success'], width=0.08,
           color=color, label=f"{offer_code[:6]}... ({offer_type})")

ax.set_xticks(x_pos)
ax.set_xticklabels(x['cluster'].unique())
ax.set_xlabel('Cluster')
ax.set_ylabel('Taxa de Sucesso (%)')
handles = [plt.Rectangle((0,0),1,1, color=color) for color in offer_type_colors.values()]
labels = list(offer_type_colors.keys())
ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
