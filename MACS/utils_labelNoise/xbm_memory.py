
import torch

class Memory:
    def __init__(self, args, device):
        self.K = args.memory_per_class*args.num_classes*2 # We want to store a minimum number of samples per-class. x2 due to the augmented views
        self.feats = -1.0 * torch.ones(self.K, args.low_dim).to(device) # doh
        self.targets = -1.0 * torch.ones(self.K, dtype=torch.long).to(device) # doh
        self.indices = -1.0 * torch.ones(self.K, dtype=torch.long).to(device)

        self.ptr = 0

    @property
    def is_full(self):
        #return self.targets[-1].item() != 0 #original
        return self.targets[-1].item() != -1 #doh

    def get(self):
        if self.is_full:
            return self.feats, self.targets,self.indices

        else:
            return self.feats[:self.ptr], self.targets[:self.ptr],self.indices[:self.ptr]


    def enqueue_dequeue(self, feats, targets,indices):
        q_size = len(targets) #256

        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.indices[-q_size:] = indices
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.indices[self.ptr: self.ptr + q_size] = indices
            self.ptr += q_size