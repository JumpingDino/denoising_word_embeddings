import numpy as np
import pandas as pd
from sklearn.metrics import pairwise
from sklearn.metrics import pairwise_distances

def cossim_vec2table(vec : np.array ,df : pd.DataFrame, vec_title = None, df_idx_title = None):
    """
    [summary]

    Parameters
    ----------
    vec : np.array
        Vetor referência para realizar o produto escalar
    df : pd.DataFrame
        Dataframe com vetores dispostos horizontalmente e o título do vetor disposto no index
    vec_title : [type], optional
        Nome dado a coluna de referencia
    df_idx_title : [type], optional
        Nome dado ao grupo no index do dataframe

    Returns
    -------
    pd.DataFrame
        DataFrame contendo duas colunas, com nomes 'df_idx_title' e similarity_'vec_title',
        de forma que os valores de df_idx_title serão os índices da tabela e 
        similarity_vec_title será o valor de similaridade com o vetor de referencia
    """

    simi_values = 1 - pairwise_distances([vec],df,metric='cosine').flatten()
    index = df.index.values

    new_df = pd.DataFrame(list(zip(index,simi_values)), columns = [df_idx_title,'similarity_' + vec_title])

    return new_df


# Para um dado vetor, aplicaremos a similaridade de cosseno com todos os vetores dispostos
# de forma horizontal em df

# Parameters
# ----------
# vec : np.array
#     Vetor referência para realizar o produto escalar
# df : pd.DataFrame
#     Dataframe com vetores dispostos horizontalmente e o título do vetor disposto no index