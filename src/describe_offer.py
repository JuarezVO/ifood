import matplotlib.pyplot as plt
import pandas as pd

def describe_offer(df: pd.DataFrame) -> None:
    sucesso = df[df['offer_success'] == 1]
    falha = df[df['offer_success'] == 0]

    fig,axs = plt.subplots(2,2,figsize=(10,5))
    axs = axs.flatten()

    ## Barplot Gender
    offer_code = df.groupby('offer_code')['offer_success'].mean() * 100
    offer_code = pd.DataFrame({'Success Rate': offer_code, 'Fail Rate': 100 - offer_code})
    offer_code.plot(kind='bar', ax=axs[0], color=['forestgreen', 'firebrick'], rot=0)
    axs[0].set_xlabel('Offer Code')
    axs[0].set_ylabel('%')

    min_value = df.groupby('min_value')['offer_success'].mean() * 100
    min_value = pd.DataFrame({'Success Rate': min_value, 'Fail Rate': 100 - min_value})
    min_value.plot(kind='bar', ax=axs[1], color=['forestgreen', 'firebrick'], rot=0)
    axs[1].set_xlabel('Min Value')
    axs[1].set_ylabel('%')

    # channels = df.groupby('channels')['offer_success'].mean() * 100
    # channels = pd.DataFrame({'Success Rate': channels, 'Fail Rate': 100 - channels})
    # channels.plot(kind='bar', ax=axs[2], color=['forestgreen', 'firebrick'], rot=0)
    # axs[2].set_xlabel('Channels')
    # axs[2].set_ylabel('%')

    # ## Boxplot Credit Card Limit, Amount e Reward
    # for i, col_name in enumerate(['credit_card_limit']):
    #     i+=3
    #     bp = axs[i].boxplot([falha[col_name].dropna(), sucesso[col_name].dropna()],labels=['Failed Offer', 'Successful Offer'],patch_artist=True,showfliers=True)
    #     colors = ['firebrick', 'forestgreen']
    #     for patch, color in zip(bp['boxes'], colors):
    #         patch.set_facecolor(color)
    #     for median in bp['medians']:
    #         median.set_color('black')
    #     axs[i].set_title(col_name)
    #     axs[i].set_ylabel(col_name)

    plt.tight_layout()
    plt.show()
