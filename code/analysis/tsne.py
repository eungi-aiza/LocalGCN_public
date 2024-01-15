import torch
import numpy as np
from sklearn.manifold import TSNE


if __name__ == "__main__":
    emb = torch.load("../../embed/lgn-gowalla-3-64.pth.tar", map_location="cpu")
    user_emb = emb["embedding_user.weight"].numpy()
    item_emb = emb["embedding_item.weight"].numpy()

    tsne = TSNE(n_components=2, verbose=1)
    user_tsne = tsne.fit_transform(user_emb)
    item_tsne = tsne.fit_transform(item_emb)
    np.save("embed/tsne_user.npy", user_tsne)
    np.save("embed/tsne_item.npy", item_tsne)