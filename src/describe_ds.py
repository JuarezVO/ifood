import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    DESCRIBE_OFFER_COLS,
    DESCRIBE_PROFILE_COLS,
    IMAGE_DESCRIBE_OFFER_PATH,
    IMAGE_DESCRIBE_PROFILE_PATH,
    PLOT_COLOR_FAIL,
    PLOT_COLOR_SUCCESS,
    PLOT_DPI,
)

def describe_offer(df: pd.DataFrame) -> None:
    """
    Descreve as ofertas
    Args:
        df: DataFrame com os dados
    Returns:
        None
    """
    print('Descrição das ofertas\n')
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    axs = axs.flatten()

    for i, col_name in enumerate(DESCRIBE_OFFER_COLS):
        data = df.groupby(col_name)['offer_success'].mean() * 100
        data = pd.DataFrame({'Success Rate': data, 'Fail Rate': 100 - data})
        data.plot(kind='bar', ax=axs[i], color=[PLOT_COLOR_SUCCESS, PLOT_COLOR_FAIL], rot=0 if i > 0 else 90, legend=False)
        axs[i].set_xlabel(col_name.replace('_', ' ').title())
        axs[i].set_ylabel('%')

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PLOT_COLOR_SUCCESS),
        plt.Rectangle((0, 0), 1, 1, color=PLOT_COLOR_FAIL),
    ]
    fig.legend(handles, ['Success Rate', 'Fail Rate'], loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(IMAGE_DESCRIBE_OFFER_PATH, dpi=PLOT_DPI, bbox_inches='tight')


def describe_profile(df: pd.DataFrame) -> None:
    """
    Descreve o perfil dos usuários
    Args:
        df: DataFrame com os dados
    Returns:
        None
    """
    print('Descrição do perfil\n')
    sucesso = df[df['offer_success'] == 1]
    falha = df[df['offer_success'] == 0]

    fig,axs = plt.subplots(2,2,figsize=(10,5))
    axs = axs.flatten()

    ## Barplot Gender
    gender = df.groupby('gender')['offer_success'].mean() * 100
    gender = pd.DataFrame({'Success Rate': gender, 'Fail Rate': 100 - gender})
    gender.plot(kind='bar', ax=axs[0], color=[PLOT_COLOR_SUCCESS, PLOT_COLOR_FAIL], rot=0)
    axs[0].set_xlabel('Gender')
    axs[0].set_ylabel('%')

    ## Boxplot Credit Card Limit, Amount e Reward
    for i, col_name in enumerate(DESCRIBE_PROFILE_COLS):
        i+=1
        bp = axs[i].boxplot([falha[col_name].dropna(), sucesso[col_name].dropna()],labels=['Failed Offer', 'Successful Offer'],patch_artist=True,showfliers=False)
        colors = [PLOT_COLOR_FAIL, PLOT_COLOR_SUCCESS]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        for median in bp['medians']:
            median.set_color('black')
        axs[i].set_ylabel(col_name.replace('_', ' ').title())

    plt.tight_layout()
    plt.savefig(IMAGE_DESCRIBE_PROFILE_PATH, dpi=PLOT_DPI, bbox_inches='tight')


def describe_ds(df: pd.DataFrame) -> None:
    describe_offer(df)
    describe_profile(df)
