import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import torch

df_train = pd.read_parquet(r'D:\dataset\MM_CTR\valid_new_3.parquet', engine='pyarrow')
# df_info_1 = pd.read_parquet(r'D:\dataset\MM_CTR\item_info_v2_w2v_item.parquet', engine='pyarrow')

df_w2v = pd.read_csv(r'D:\code\MM_CTR\user_w2v.gz', compression='gzip')
df_w2v = df_w2v.groupby('user_id', as_index=False).mean()
df_w2v['combined_np'] = df_w2v[['user_w2v_emb0', 'user_w2v_emb1', 'user_w2v_emb2', 'user_w2v_emb3', 'user_w2v_emb4',
                                'user_w2v_emb5', 'user_w2v_emb6', 'user_w2v_emb7', 'user_w2v_emb8', 'user_w2v_emb9',
                                'user_w2v_emb10', 'user_w2v_emb11', 'user_w2v_emb12', 'user_w2v_emb13', 'user_w2v_emb14',
                                'user_w2v_emb15']].apply(lambda row: np.array(row, dtype='float32'), axis=1)
array_list = [np.array(np.zeros(16), dtype='float32') for i in df_train['user_id']]
df_train['w2v_user'] = array_list
i = 0
for user_id in df_train['user_id']:
    i += 1
    print(i)
    if user_id in df_w2v.user_id:
        index = df_w2v[df_w2v.user_id == user_id].index
        if index.shape[0] == 0:
            continue
        index = index[0]
        data = df_w2v['combined_np'][index]
        df_train['w2v_user'][i] = np.array(data, dtype='float32')
df_train.to_parquet(r'D:\dataset\MM_CTR\valid_new_4.parquet')
