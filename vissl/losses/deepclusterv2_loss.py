# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import pprint
import os

import torch
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    all_reduce_sum,
    gather_from_all,
    get_rank,
    get_world_size,
)
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict
from vissl.utils.misc import get_indices_sparse
from vissl.utils.io import save_file

torch.cuda.empty_cache()

@register_loss("deepclusterv2_loss")
class DeepClusterV2Loss(ClassyLoss):
    """
    Loss used for DeepClusterV2 approach as provided in SwAV paper
    https://arxiv.org/abs/2006.09882

    Config params:
        DROP_LAST (bool): automatically inferred from DATA.TRAIN.DROP_LAST
        BATCHSIZE_PER_REPLICA (int): 256  # automatically inferred from
                                            DATA.TRAIN.BATCHSIZE_PER_REPLICA
        num_crops (int): 2                # automatically inferred from DATA.TRAIN.TRANSFORMS
        temperature (float): 0.1
        num_clusters (List[int]): [3000, 3000, 3000]
        kmeans_iters (int): 10
        crops_for_mb: [0]
        embedding_dim: 128
        num_train_samples (int): -1       # @auto-filled
    """

    def __init__(self, loss_config: AttrDict):
        super().__init__()

        self.loss_config = loss_config
        size_dataset = self.loss_config.num_train_samples

        size_memory_per_process = int(math.ceil(size_dataset * 1.0 / get_world_size()))

        if self.loss_config.DROP_LAST:
            size_memory_per_process -= (
                size_memory_per_process % self.loss_config.BATCHSIZE_PER_REPLICA
            )

        self.nmb_mbs = len(self.loss_config.memory_params.crops_for_mb)
        self.nmb_heads = len(self.loss_config.num_clusters)
        self.num_clusters = self.loss_config.num_clusters
        self.embedding_dim = self.loss_config.memory_params.embedding_dim
        self.crops_for_mb = self.loss_config.memory_params.crops_for_mb
        self.nmb_unique_idx = self.loss_config.BATCHSIZE_PER_REPLICA
        self.num_crops = self.loss_config.num_crops
        self.temperature = self.loss_config.temperature
        self.nmb_kmeans_iters = self.loss_config.kmeans_iters
        self.start_idx = 0
        
        
        self.register_buffer(
            "local_memory_embeddings",
            torch.zeros(self.nmb_mbs, size_memory_per_process, self.embedding_dim),
        )
        self.register_buffer(
            "local_memory_index", torch.zeros(size_memory_per_process).long()
        )
        self.register_buffer(
            "assignments", -100 * torch.ones(self.nmb_heads, size_dataset).long()
        )
################################################################        
        self.register_buffer(
            "indexes", -100 * torch.ones(self.nmb_heads, size_dataset).long()
        )
        
        self.register_buffer(
            "distance", -100 * torch.rand(self.nmb_heads, size_dataset).half()
        )
#################################################################        
        for i, k in enumerate(self.loss_config.num_clusters):
            self.register_buffer(
                "centroids" + str(i), torch.rand(k, self.embedding_dim)
            )

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates DeepClusterV2Loss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            DeepClusterV2Loss instance.
        """
        return cls(loss_config)

    def forward(self, output: torch.Tensor, idx: int):
        output = nn.functional.normalize(output, dim=1, p=2)
        loss = 0
        for i in range(self.nmb_heads):
       
            scores = (
                torch.mm(output, getattr(self, "centroids" + str(i)).t())
                / self.temperature
            )
            loss += self.cross_entropy_loss(scores, self.assignments[i][idx])
        loss /= self.nmb_heads

        self.update_memory_bank(output, idx)
        return loss

    def init_memory(self, dataloader, model):
        logging.info(f"Rank: {get_rank()}, Start initializing memory banks")
        start_idx = 0
        with torch.no_grad():
            
            for inputs in dataloader:
                nmb_unique_idx = len(inputs["data_idx"][0]) // self.num_crops
                index = inputs["data_idx"][0][:nmb_unique_idx].cuda(non_blocking=True)
                # get embeddings
                outputs = []
                for crop_idx in self.crops_for_mb:
                    inp = inputs["data"][0][crop_idx].cuda(non_blocking=True)
                    #model = model.to(inp.device)
                    outputs.append(nn.functional.normalize(model(inp)[0], dim=1, p=2))

                # fill the memory bank
                self.local_memory_index[start_idx : start_idx + nmb_unique_idx] = index
                for mb_idx, embeddings in enumerate(outputs):
                    self.local_memory_embeddings[mb_idx][
                        start_idx : start_idx + nmb_unique_idx
                    ] = embeddings
                start_idx += nmb_unique_idx
        logging.info(
            f"Rank: {get_rank()}, Memory banks initialized: "
            "full first forward pass done"
        )

    def update_memory_bank(self, emb, idx):
        nmb_unique_idx = len(idx) // self.num_crops       
        idx = idx[:nmb_unique_idx]
        self.local_memory_index[self.start_idx : self.start_idx + nmb_unique_idx] = idx
        for i, crop_idx in enumerate(self.crops_for_mb):
            self.local_memory_embeddings[i][
                self.start_idx : self.start_idx + nmb_unique_idx
            ] = emb[crop_idx * nmb_unique_idx : (crop_idx + 1) * nmb_unique_idx]
        self.start_idx += nmb_unique_idx

    def cluster_memory(self):
        print('entering cluster_memory function in deepcluster2_loss.py')
        
        # create the directory for checkpoints
        checkpoint_dir = self.loss_config.CHECKPOINT.DIR
        logging.info(f"checkpoint_dir: {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.start_idx = 0
        j = 0
        with torch.no_grad():
            for i_K, K in enumerate(self.num_clusters):
                # run distributed k-means
                
                # init centroids with elements from memory bank of rank 0
                centroids = torch.empty(K, self.embedding_dim).cuda(non_blocking=True)         
                if get_rank() == 0:
                    random_idx = torch.randperm(len(self.local_memory_embeddings[j]))[
                        :K
                    ]
                    
                    assert len(random_idx) >= K, "please reduce the number of centroids"
                    centroids = self.local_memory_embeddings[j][random_idx]
                dist.broadcast(centroids, 0)

                for n_iter in range(self.nmb_kmeans_iters + 1):

                    # E step
                    dot_products = torch.mm(
                        self.local_memory_embeddings[j], centroids.t()
                    )
                    distance, assignments = dot_products.max(dim=1)

                    # finish
                    if n_iter == self.nmb_kmeans_iters:
                        break
                    # M step
                    where_helper = get_indices_sparse(assignments.cpu().numpy())
                    counts = torch.zeros(K).cuda(non_blocking=True).int()
                    emb_sums = torch.zeros(K, self.embedding_dim).cuda(
                        non_blocking=True
                    )
                    for k in range(len(where_helper)):
                        if len(where_helper[k][0]) > 0:
                            emb_sums[k] = torch.sum(
                                self.local_memory_embeddings[j][where_helper[k][0]],
                                dim=0,
                            )
                            counts[k] = len(where_helper[k][0])
                        
                    
                    all_reduce_sum(counts) #performing sum reduction of tensor over all processes
                    mask = counts > 0
                    all_reduce_sum(emb_sums)
                    centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

                    # normalize centroids
                    centroids = nn.functional.normalize(centroids, dim=1, p=2)

                getattr(self, "centroids" + str(i_K)).copy_(centroids)
#################################################################################
                logging.info(f'Rank: {get_rank()}, saving assignigment' )
                torch.save(self.centroids0, os.path.join(checkpoint_dir, "centroids0.pt"))
                #torch.save(self.centroids0,"/data1/runs/dcv2_ir108_100x100_k9_expats_35k_nc/checkpoints/centroids0.pt")
                #torch.save(self.centroids1,"/data1/runs/dcv_ir108_128x128_k9_germany_30kcrops_grey_10th-90th_CMA/checkpoints/centroids1.pt")
                #save_file(self.centroids0,"/home/Daniele/codes/vissl/runs/dcv2_cot_128x128_k7_germany_60kcrops_1epoch/checkpoints/centroids0.pt")
                #save_file(self.centroids1,"/home/Daniele/codes/vissl/runs/dcv2_cot_128x128_k7_germany_60kcrops_1epoch/checkpoints/centroids1.pt")
                #torch.save(self.centroids2,"/p/project/deepacf/kiste/DC/k8/germany_64_800ep/ftrain_2013_55k/centroids2.pt")
                # gather the assignments
                assignments_all = gather_from_all(assignments)
                indexes_all = gather_from_all(self.local_memory_index)  
                distance_all = gather_from_all(distance)
                self.assignments[i_K] = -100
###############################################################                
                self.indexes[i_K] = -100                
                self.indexes[i_K][indexes_all] = indexes_all
                
                self.distance[i_K] = -100                
                self.distance[i_K][indexes_all] = distance_all
################################################################                
                self.assignments[i_K][indexes_all] = assignments_all
                
                j = (j + 1) % self.nmb_mbs
#################################################################################
            
            
            torch.save(self.assignments, os.path.join(checkpoint_dir, "assignments.pt"))
            torch.save(self.indexes, os.path.join(checkpoint_dir, "indexes.pt"))
            torch.save(self.distance, os.path.join(checkpoint_dir, "distances.pt"))
            #torch.save(self.assignments,"/data1/runs/dcv2_ir108_100x100_k9_expats_35k_nc/checkpoints/assignments.pt")
            #torch.save(self.indexes,"/data1/runs/dcv2_ir108_100x100_k9_expats_35k_nc/checkpoints/indexes.pt")
            #torch.save(self.distance,"/data1/runs/dcv2_ir108_100x100_k9_expats_35k_nc/checkpoints/distances.pt")
            
            #save_file(self.assignments,"/home/Daniele/codes/vissl/runs/dcv2_cot_128x128_k7_germany_60kcrops_1epoch/checkpoints/assignments_800ep.pt")
            #save_file(self.indexes,"/home/Daniele/codes/vissl/runs/dcv2_cot_128x128_k7_germany_60kcrops_1epoch/checkpoints/indexes_800ep.pt")
            #save_file(self.distance,"/home/Daniele/codes/vissl/runs/dcv2_cot_128x128_k7_germany_60kcrops_1epoch/checkpoints/distance_800ep.pt")
            logging.info(f'Rank: {get_rank()}, assignigment saved' )
###################################################################################            
        logging.info(f"Rank: {get_rank()}, clustering of the memory bank done")

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)
        
       
