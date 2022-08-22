
import torch
import torch.nn as nn


class MoCo(nn.Module):

    # The temperature τ in Eqn.(1) is set as 0.07 [61].
    # We adopt a ResNet [33] as the encoder, whose last 
    # fully-connected layer (after global average pooling) 
    # has a fixed-dimensional output (128-D [61])
    # We set K = 65536 and m = 0.999. 
    # The temperature τ in Eqn.(1) is set as 0.07 [61].
    def __init__(self, base_encoder, dim=128, k=8096, m=0.999, T=0.07, mlp=False):
        super().__init__()
        #self.base_encoder = base_encoder
        self.T = T
        self.K = k 
        self.dim = dim
        self.m = m

        # print(self.base_encoder)
        self.encoder_q = base_encoder(dim)
        self.encoder_k = base_encoder(dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # At the core of our approach is maintaining the dictionary as 
        # a queue of data samples.
        self.register_buffer("queue", torch.randn(dim, k))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    # We propose a momentum update to address this issue.
    # Formally, denoting the parameters of fk as θk and those
    # of fq as θq, we update θk by:
    # Here m ∈ [0, 1) is a momentum coefficient. Only the parameters 
    # θq are updated by back-propagation. 
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def forward(self, im_q, im_k):

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC

        # This out- put vector is normalized by 
        # its L2-norm [61]. 
        q = nn.functional.normalize(q, dim=1)


        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()
        

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle



        return x
