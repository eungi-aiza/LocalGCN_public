"""
Reference: https://github.com/gusye1234/LightGCN-PyTorch
"""
import world
import torch
from torch import nn
from modules import LOCA
import sys
import datetime
import os
import numpy as np


class LightGCN(nn.Module):
    def __init__(self, config: dict, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()
        
        if self.__class__.__name__ == "LightGCN":
            self.Graph = self.dataset.get_sparse_graph()
            world.LOGGER.info(f"use normalized Graph(dropout:{self.config['dropout']})")

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        world.LOGGER.info('use xavier initilizer')
        self.f = nn.Sigmoid()

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        g_dropped = self.Graph
            
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_dropped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating   
        
    def get_embedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0) = self.get_embedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2) * (user_emb_0.norm(2).pow(2) + pos_emb_0.norm(2).pow(2) + neg_emb_0.norm(2).pow(2)) / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss
    

class LocalGCN(LightGCN):
    def __init__(self, config: dict, dataset):
        super().__init__(config, dataset)
        self.group = self.config["groups"]
        self.train_kernel = self.kernel_generation()
        self.g_dropped, self.g_dropped_group = self.graphs_generation()
        
    # def __init__(self, config: dict, dataset):
    #     super(LocalGCN, self).__init__()
    #     self.config = config
    #     self.dataset = dataset
    #     self.__init_weight()
    #     self.group = self.config["groups"]
    #     self.train_kernel = self.kernel_generation()
    #     self.g_dropped, self.g_dropped_group = self.graphs_generation()
            
    # def __init_weight(self):
    #     self.num_users = self.dataset.n_users
    #     self.num_items = self.dataset.m_items
    #     self.latent_dim = self.config["latent_dim_rec"]
    #     self.n_layers = self.config["n_layers"]
    #     self.keep_prob = self.config["keep_prob"]
    #     self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
    #     self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

    #     nn.init.xavier_uniform_(self.embedding_user.weight)
    #     nn.init.xavier_uniform_(self.embedding_item.weight)
    #     world.LOGGER.info("use xavier initilizer")
    #     self.f = nn.Sigmoid()      

    def kernel_generation(self):
        world.LOGGER.info(f"Kernel Making starts from pretrained embeddings: {world.PRETRAINED_EMB}")
        self.subgraph_gen = LOCA(dataset=self.dataset, embed_path=world.PRETRAINED_EMB)
        train_kernel, logs = self.subgraph_gen()
        world.LOGGER.info("\n" + logs)
        return train_kernel

    def graphs_generation(self):
        graph_path = world.GRAPH
        if os.path.exists(graph_path):
            world.LOGGER.info(f"Graph Making already done: {graph_path}")
            graphs = torch.load(graph_path, map_location="cpu")
            g_dropped = graphs["graph"].to(world.config["device"])
            g_dropped_group = graphs["graph_groups"]
        else:
            world.LOGGER.info(f"Graph Making starts: {graph_path}")
            g_dropped = self.dataset.get_sparse_graph()
            g_dropped_group = self.subgraph_generation()
            torch.save({"graph": g_dropped, "graph_groups": g_dropped_group}, graph_path)
        return g_dropped, g_dropped_group

    def subgraph_generation(self):
        s = datetime.datetime.now()
        emb_classes = torch.from_numpy(self.train_kernel).to(world.config["device"])
        group_embs = self.subgraph_basement(emb_classes)
        g_droped_group = self.dataset.get_sparse_graph_groups(group_embs)
        e = datetime.datetime.now()
        t = (e - s).seconds
        world.LOGGER.info(f"subgraph_generation: {t} seconds")
        return g_droped_group

    def subgraph_basement(self, emb_classes):
        ## overlapping
        emb_classes[emb_classes > 0] = 1.0
        return emb_classes.float()
        ## exclusive
        # top_class, top_class_idx = torch.topk(emb_classes, k=1, dim=1)
        # group_embs = (emb_classes == top_class).float()
        # u_group_embs, i_group_embs = torch.split(group_embs, [self.num_users, self.num_items], dim=0)
        # i_group_embs = torch.ones_like(i_group_embs)
        # group_embs = torch.cat([u_group_embs, i_group_embs], dim=0)
        # return group_embs
        
    def computer(self):
        users_emb = self.embedding_user.weight  # n_user, rec_dim
        items_emb = self.embedding_item.weight  # m_item, rec_dim
        ego_emb = torch.cat([users_emb, items_emb])  # E_0 (n_user + m_item, rec_dim)

        # embedding transformation
        embs = [ego_emb]  ## E_0
        side_emb = torch.sparse.mm(self.g_dropped, ego_emb)  # ui, 64
        embs.append(side_emb)  ## E_1

        ego_emb_g = [ego_emb for _ in range(0, self.group)]
        ego_emb = None  ## E_2 ~ E_k sum
        for k in range(1, self.n_layers):
            for g in range(0, self.group):
                g_gpu = self.g_dropped_group[g]
                g_gpu = g_gpu.to(dtype=torch.float32).to(world.config["device"])
                side_emb = torch.sparse.mm(g_gpu, ego_emb_g[g])  # E_i+1 per g
                ego_emb_g[g] = ego_emb_g[g] + side_emb  # E_i + E_i+1
                if ego_emb is None:
                    ego_emb = torch.sparse.mm(self.g_dropped, side_emb)
                else:
                    ego_emb += torch.sparse.mm(self.g_dropped, side_emb)

                g_gpu = g_gpu.cpu()
                del g_gpu, side_emb
                torch.cuda.empty_cache()

            embs.append(ego_emb)  # final E_i+1

        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)  # Mean of E_0 ~ E_k
        users, items = torch.split(embs, [self.num_users, self.num_items])

        return users, items

    # def get_users_rating(self, users):
    #     all_users, all_items = self.computer()
    #     users_emb = all_users[users.long()]
    #     items_emb = all_items
    #     rating = self.f(torch.matmul(users_emb, items_emb.t()))
    #     return rating

    # def get_embedding(self, users, pos_items, neg_items):
    #     all_users, all_items = self.computer()
    #     users_emb = all_users[users]
    #     pos_emb = all_items[pos_items]
    #     neg_emb = all_items[neg_items]
    #     users_emb_ego = self.embedding_user(users)
    #     pos_emb_ego = self.embedding_item(pos_items)
    #     neg_emb_ego = self.embedding_item(neg_items)
    #     return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    # def bpr_loss(self, users, pos, neg):
    #     (users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0) = self.get_embedding(
    #         users.long(), pos.long(), neg.long()
    #     )
    #     reg_loss = (
    #         (1 / 2) * (user_emb_0.norm(2).pow(2) + pos_emb_0.norm(2).pow(2) + neg_emb_0.norm(2).pow(2)) / float(len(users))
    #     )
    #     pos_scores = torch.mul(users_emb, pos_emb)
    #     pos_scores = torch.sum(pos_scores, dim=1)
    #     neg_scores = torch.mul(users_emb, neg_emb)
    #     neg_scores = torch.sum(neg_scores, dim=1)

    #     loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    #     return loss, reg_loss


class AnchorEmbRec(nn.Module):
    def __init__(self, config: dict, dataset):
        super(AnchorEmbRec, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()
        self.group = self.config["groups"]
        self.train_kernel = self.kernel_generation()
        self.anchors = self.load_anchors()
        self.Graph = self.dataset.get_sparse_graph()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["n_layers"]
        self.embedding_user, self.embedding_item = self.load_embedding(world.PRETRAINED_EMB)
        self.f = nn.Sigmoid()

    def load_anchors(self):
        anchor_path = os.path.join(world.ROOT_PATH, "anchor")
        file_name = os.path.join(
            anchor_path,
            f"anchor-{world.config['dataset']}-g{world.config['groups']}-{world.config['train_h']}-{world.config['test_h']}-{world.config['subg']}.pt",
        )
        if not os.path.exists(file_name):
            raise FileNotFoundError
        anchors = torch.load(file_name)
        return anchors

    def kernel_generation(self):
        world.LOGGER.info(f"Kernel Making starts from pretrained embeddings: {world.PRETRAINED_EMB}")
        self.subgraph_gen = LOCA(dataset=self.dataset, embed_path=world.PRETRAINED_EMB)
        train_kernel, logs = self.subgraph_gen()
        world.LOGGER.info("\n" + logs)
        return train_kernel
    
    def load_embedding(self, embed_path):
        if not os.path.exists(embed_path):
            raise FileNotFoundError
        if world.config["device"] == "cuda":
            checkpoint = torch.load(embed_path)
        else:
            checkpoint = torch.load(embed_path, map_location=torch.device("cpu"))
        users = checkpoint["embedding_user.weight"].to(world.config["device"])
        items = checkpoint["embedding_item.weight"].to(world.config["device"])
        return users, items

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user
        items_emb = self.embedding_item
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        graph = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def subgraph_basement(self, emb_classes):
        ## overlapping
        emb_classes[emb_classes > 0] = 1.0
        return emb_classes
    
    def map_users_to_clusters(self, all_users): # all_users (UxD)
        user_kernel, _ = np.split(self.train_kernel, [self.num_users])
        user_kernel_norm = user_kernel/np.sum(user_kernel, axis=1, keepdims=True) # (UxG) mapping

        ## (1) ANCHOR
        anchor_emb = all_users[self.anchors] # (GxD)

        ## (2) AVG
        # # user_kernel_one = self.subgraph_basement(user_kernel)
        # user_kernel_one = user_kernel/np.sum(user_kernel, axis=0, keepdims=True)
        # anchor_emb = []
        # for g in range(user_kernel_one.shape[1]):
        #     mapping = torch.from_numpy(user_kernel_one[:,g]).float().to(world.config["device"])
        #     mapping = mapping.unsqueeze(0) # (1,U)
        #     anchor = torch.matmul(mapping, all_users) # (1,D)
        #     anchor_emb.append(anchor) 
        # anchor_emb  = torch.cat(anchor_emb)

        mapped_users = torch.matmul(torch.from_numpy(user_kernel_norm).float().to(world.config["device"]),
                                     anchor_emb)
        return mapped_users

    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        mapped_users = self.map_users_to_clusters(all_users)
        users_emb = mapped_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
