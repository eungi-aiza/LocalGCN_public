import torch
import numpy as np
import matplotlib.pyplot as plt
import umap


if __name__ == "__main__":
    emb = torch.load("../../embed/lgn-gowalla-3-64.pth.tar", map_location="cpu")
    user_emb = emb["embedding_user.weight"].numpy()
    item_emb = emb["embedding_item.weight"].numpy()
    
    reducer = umap.UMAP(n_components=2, verbose=True)
    user_embed = reducer.fit_transform(user_emb)
    item_embed = reducer.fit_transform(item_emb)
    np.save('./embed/umap_user.npy', user_embed)
    np.save('./embed/umap_item.npy', item_embed)
    