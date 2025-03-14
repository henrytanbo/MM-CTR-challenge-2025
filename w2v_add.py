import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import torch

df_info = pd.read_parquet(r'D:\dataset\MM_CTR\item_info_v2.parquet', engine='pyarrow')
# df_info_1 = pd.read_parquet(r'D:\dataset\MM_CTR\item_info_v2_w2v_item.parquet', engine='pyarrow')

df_w2v = pd.read_csv(r'D:\code\MM_CTR\item_w2v.gz', compression='gzip')
df_w2v['combined_np'] = df_w2v[['item_w2v_emb0', 'item_w2v_emb1', 'item_w2v_emb2', 'item_w2v_emb3', 'item_w2v_emb4',
                                'item_w2v_emb5', 'item_w2v_emb6', 'item_w2v_emb7', 'item_w2v_emb8', 'item_w2v_emb9',
                                'item_w2v_emb10', 'item_w2v_emb11', 'item_w2v_emb12', 'item_w2v_emb13', 'item_w2v_emb14',
                                'item_w2v_emb15']].apply(lambda row: np.array(row, dtype='float32'), axis=1)
array_list = [np.array(np.zeros(16), dtype='float32') for i in df_info['item_id']]
df_info['w2v_item'] = array_list
i = 0
for item_id in df_info['item_id']:
    if item_id in df_w2v['item_id']:
        index = df_w2v[df_w2v.item_id == item_id].index
        # w2v = df_w2v.loc[df_w2v['item_id'] == item_id, 'combined_np']
        if index.shape[0] == 0:
            i += 1
            print(i)
            continue
        index = index[0]
        data = df_w2v['combined_np'][index]
        df_info['w2v_item'][i] = np.array(data, dtype='float32')
        i += 1
        print(i)
df_info.to_parquet(r'D:\dataset\MM_CTR\item_info_v2_w2v_item.parquet')
