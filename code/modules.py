"""
Reference: https://github.com/jin530/LOCA.git 
"""
import torch
import torch.nn as nn
from torch import nn, einsum
import world
import copy

## LOCA
import math

import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
import warnings
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix, load_npz, save_npz, find
import scipy.sparse as sp
import sys
import os

warnings.filterwarnings("ignore")


class LOCA(nn.Module):
    def __init__(
        self,
        dataset,
        embed_path,
        anchor_selection="coverage",
        dist_type="arccos",
        kernel_type="epanechnikov",
        train_h=world.config["train_h"],
        test_h=world.config["test_h"],
        embedding_type=f"embedding_{world.config['subg']}.weight",
    ):
        super(LOCA, self).__init__()
        self.embed_path = embed_path
        self.num_local = world.config["groups"]
        self.anchor_selection = anchor_selection
        self.dist_type = dist_type
        self.kernel_type = kernel_type
        self.train_h = train_h
        self.test_h = test_h
        self.embedding_type = embedding_type
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items

        self.user_embedding = self.load_embedding(self.embed_path, self.embedding_type)

    def dist(self, a, norm, anchor=None):
        a_anchor = np.reshape(a[anchor], (1, -1)) 
        numer = a_anchor @ a.T 
        if np.sum(numer) == 0:
            return 999

        denom = np.maximum(norm[anchor] * norm.T, 1e-10)  
        return np.squeeze((2 / math.pi) * np.arccos(np.clip(numer / denom, -1, 1)))

    def kernel(self, a, norm, h=0.8, kernel_type="Epanechnikov", anchor=None):
        return (3 / 4) * np.maximum(1 - np.power(self.dist(a, norm, anchor) / h, 2), 0)

    def load_embedding(self, embed_path, embedding_type):
        if world.config["device"] == "cuda":
            checkpoint = torch.load(embed_path)
        else:
            checkpoint = torch.load(embed_path, map_location=torch.device("cpu"))
        embedding = checkpoint[embedding_type].cpu().numpy()
        return embedding

    def build_kernel_matrix(self):
        # for each local model
        if self.anchor_selection == "kmeans":
            user_dist_with_centers = KMeans(n_clusters=self.num_local, random_state=0).fit_transform(self.user_embedding)
            user_anchors = np.argsort(user_dist_with_centers, axis=0)[0]
        elif self.anchor_selection == "random":
            user_anchors = np.random.choice(self.num_users, size=self.num_local, replace=False)
        elif self.anchor_selection == "coverage":
            user_anchors = np.full((self.num_local,), -1, dtype=int)
            norm = np.linalg.norm(self.user_embedding, axis=1, keepdims=True)
            if os.path.exists(world.WMAT):
                world.LOGGER.info(f"WMAT making already done: {world.WMAT}")
                W_mat = load_npz(world.WMAT)
            else:
                world.LOGGER.info(f"WMAT making starts")
                W_mat = csr_matrix((self.num_users, self.num_users), dtype=np.int8)
                user_range = np.arange(self.num_users)
                # if j is covered by i, W_mat[u,i] = 1.
                step = 10 
                for u in tqdm(range(0, self.num_users, step)):  
                    u_cover = self.kernel(self.user_embedding, norm, self.test_h, self.kernel_type, u) != 0
                    W_mat[u, user_range[u_cover]] = 1
                    del u_cover
                save_npz(world.WMAT, W_mat)
            W_mat_init = copy.deepcopy(W_mat)
        else:
            raise Exception("Choose correct self.anchor_selection")
        # item_anchors = np.random.choice(self.num_items, size=self.num_local, replace=False)

        # for each local model
        train_kernel_ret = []
        self.train_coverage = []
        self.test_coverage = []
        results = ""
        for t in tqdm(range(self.num_local)):
            if self.anchor_selection == "coverage":
                assert self.num_local <= self.num_users, f"number of communities should be smaller than number of users"

                # count covered nodes per user
                idx, cnt = np.unique(W_mat.tocoo().row, return_counts=True)
                if len(idx) == 0:
                    W_mat = copy.deepcopy(W_mat_init)
                    idx, cnt = np.unique(W_mat.tocoo().row, return_counts=True)

                # select anchor
                sorted_indices = np.argsort(cnt)[::-1]
                for sorted_idx in sorted_indices:
                    if idx[sorted_idx] not in user_anchors:
                        user_anchors[t] = idx[sorted_idx]
                        break

                # eliminate nodes covered by anchor
                row_start = W_mat.indptr[user_anchors[t]]
                row_end = W_mat.indptr[user_anchors[t] + 1]
                new_covered = W_mat.indices[row_start:row_end]
                rows, cols, vals = find(W_mat)
                mask = np.isin(cols, new_covered)
                new_rows = rows[~mask]
                new_cols = cols[~mask]
                new_vals = vals[~mask]
                W_mat = csr_matrix((new_vals, (new_rows, new_cols)), shape=W_mat.shape, dtype=np.int8)

            user_anchor_t = user_anchors[t]
            # item_anchor_t = item_anchors[t]

            # train user kernel
            train_user_kernel_t = self.kernel(
                self.user_embedding, norm, self.train_h, self.kernel_type, user_anchor_t
            ) 
            train_item_kernel_t = np.ones(self.num_items) 

            train_coverage_size = (np.count_nonzero(train_user_kernel_t) * np.count_nonzero(train_item_kernel_t)) / (
                self.num_users * self.num_items
            )
            train_kernel_t = np.concatenate([train_user_kernel_t, train_item_kernel_t])

            train_kernel_ret.append(train_kernel_t)
            self.train_coverage.append(train_coverage_size)

            temp = "Anchor %3d coverage : %.5f" % (t, train_coverage_size)
            results += temp + "\n"
            del train_kernel_t
            del train_coverage_size

        train = "Coverage : %.5f (Average), %.5f (Max), %.5f (Min)" % (
            np.mean(self.train_coverage),
            max(self.train_coverage),
            min(self.train_coverage),
        )
        results += train

        train_kernel_ret = np.stack(train_kernel_ret).T

        anchor_path = os.path.join(world.ROOT_PATH, "anchor")
        world.mkdir_if_not_exist(anchor_path)
        file_name = os.path.join(
            anchor_path,
            f"anchor-{world.config['dataset']}-g{world.config['groups']}-{world.config['train_h']}-{world.config['test_h']}-{world.config['subg']}.pt",
        )
        torch.save(user_anchors, file_name)

        return train_kernel_ret, results

    def forward(self):
        train_kernel, results = self.build_kernel_matrix()
        return train_kernel, results
