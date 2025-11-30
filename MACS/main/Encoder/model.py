

import torch
import torch.nn as nn
from Encoder.spd import SPDTransform, SPDTangentSpace, SPDRectified
import numpy as np
from scipy import signal
import torch.nn.functional as F


class Gclip_Gsyn(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.dev =  torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    def patch_len(self, n, epochs):
        list_len=[]
        base = n//epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base*epochs):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')

    def forward(self, x):
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
      
        x_list = list(torch.split(x, list_patch, dim=-1))
        result_list = []
        for i, item in enumerate(x_list):
           
            item=item.view(item.shape[0],item.shape[1],item.shape[-1]).cpu().detach().numpy()
           
            result=[]
            for j in range(item.shape[0]):
                result.append((np.corrcoef(item[j])))
      
            result_list.append(result)

        x_array = np.array(result_list)
        x_array = x_array.transpose(1, 0, 2, 3) 
        x_tensor = torch.from_numpy(x_array)
        x = x_tensor.to(self.dev)
        
        return x

class Gmfd(nn.Module):
    def __init__(self):
        super().__init__()
        self.dev =  torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    def matrix_exp(self,t):
        # Condition: t is a symmetric 2D matrix
        s, u = np.linalg.eigh(t)
        return np.matmul(np.matmul(u, np.diag(np.exp(s))), u.T)
    def convert_to_symmetric_positive_definite(self,adjacency_matrix):
        # Compute the eigenvalue decomposition of the adjacency matrix
        eigenvalues, eigenvectors = np.linalg.eig(adjacency_matrix)
        # Take the absolute values of the eigenvalues
        eigenvalues_abs = np.abs(eigenvalues)
        
        # Compute the minimum eigenvalue
        min_eigenvalue = np.min(eigenvalues_abs)
        
        # Add a small positive constant to ensure positive definiteness
        epsilon = 1e-6
        modified_eigenvalues = eigenvalues_abs + epsilon * min_eigenvalue
        
        # Reconstruct the modified adjacency matrix
        modified_adjacency_matrix = eigenvectors @ np.diag(modified_eigenvalues) @ np.linalg.inv(eigenvectors)
        
        # Make the matrix symmetric by averaging with its transpose
        symmetric_matrix = (modified_adjacency_matrix + modified_adjacency_matrix.T) / 2.0
        
        return symmetric_matrix
    
    def matrix_exp(self,t):
        # Condition: t is a symmetric 2D matrix
        s, u = np.linalg.eigh(t)
        return np.matmul(np.matmul(u, np.diag(np.exp(s))), u.T)
    def forward(self, x):
        batch_size, epoch, h, w = x.shape
        spd_matrices = []

        for i in range(batch_size):
            for j in range(epoch):
                submatrix = x[i, j, :, :]  
                submatrix = submatrix.cpu().detach().numpy()  
                # is_symmetric = np.allclose(submatrix, submatrix.T)

                spd_matrix = self.convert_to_symmetric_positive_definite(submatrix)
                spd_matrices.append(spd_matrix)

        spd_array = np.concatenate(spd_matrices, axis=0)
        spd_tensors = torch.tensor(spd_array, device=self.dev)
        spd_tensors = spd_tensors.view(batch_size, epoch,h, w)
      

        return spd_tensors


class Gdatt(nn.Module):
    def __init__(self, in_embed_size, out_embed_size):
        super(Gdatt, self).__init__()
        
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.q_trans = SPDTransform(self.d_in, self.d_out)
        self.k_trans = SPDTransform(self.d_in, self.d_out)
        self.v_trans = SPDTransform(self.d_in, self.d_out)

    def tensor_log(self, t):#4dim
        u, s, v = torch.svd(t) 
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)
        
    def tensor_exp(self, t):#4dim
        # condition: t is symmetric!
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)
    def log_euclidean_distance(self, A, B):
        inner_term = self.tensor_log(A) - self.tensor_log(B)  
        inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
        _, s, _= torch.svd(inner_multi) 
        final = torch.sum(s, dim=-1)
        return final

    def LogEuclideanMean(self, weight, cov):
        bs, num_p, size ,_= cov.shape
        cov = self.tensor_log(cov).view(bs, num_p, -1)

        output_list = []

        for i in range(num_p):
            # Replicate the i-th row of cov for num_p times
            cov_replicated = torch.unsqueeze(cov[:, i, :], dim=1).expand(-1, num_p, -1)
            # Element-wise multiplication
            weighted_cov = torch.unsqueeze(weight[:, i, :], dim=1) @ cov_replicated 
            # Append to the output list
            output_list.append(weighted_cov)

        # Concatenate the results along the second dimension
        output = torch.cat(output_list, dim=1)
        output = output.view(bs, num_p, size, size)

        return self.tensor_exp(output)
        
    def forward(self, x, shape=None):
        
        if len(x.shape)==3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.float)# patch:[b, #patch, c, c]
        q_list = []; k_list = []; v_list = []  
        # calculate Q K V
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs*m, self.d_in, self.d_in)
        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1) 
    
        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1 ) 
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3], K_expand.shape[4]) 
        
        atten_energy = self.log_euclidean_distance(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1/(1+torch.log(1 + atten_energy))).permute(0, 2, 1)#now row is c.c.
        #atten_prob = nn.Softmax(dim=-2)(atten_energy).permute(0, 2, 1)#now row is c.c.


        
        # calculate outputs(v_i') of attention module
        output = self.LogEuclideanMean(atten_prob, V)

        output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out)

        shape = list(output.shape[:2])
        shape.append(-1)

        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape
class mAtt(nn.Module):
    def __init__(self,epochs,num_classes, low_dim,dim,channel):
        super().__init__()
        #FE
        dim=dim
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, channel, (channel, 1))
        self.Bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, (1, 25), padding=(0, 12))
        self.Bn2   = nn.BatchNorm2d(channel)
        self.ract1 = Gclip_Gsyn(epochs=epochs)
        # riemannian part
        self.spd=Gmfd()
        # riemannian part
        self.att2 = Gdatt(channel, dim)
        self.ract2  = SPDRectified()
        self.tangent = SPDTangentSpace(dim)
        self.flat = nn.Flatten()
     
        size1=int((dim+dim*(dim-1)/2)*epochs)
        size2=int(dim+dim*(dim-1)/2)
        self.head1 = nn.Linear(size1, size2, bias=True)
        self.head2=nn.Linear(size2, low_dim, bias=True)
        self.head1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.head2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.linear1 = nn.Linear(size1, size2, bias=True)
        self.linear2=nn.Linear(size2,num_classes, bias=True)
        self.linear1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.linear2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
    

    def forward(self, x):
        x=x.float()
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        x = self.ract1(x)
        x=self.spd(x)
        x, shape = self.att2(x)
        x = self.ract2(x)
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1) 
        x = self.flat(x)
    
        outContrast =  F.leaky_relu(self.head1(x))
        outContrast = self.head2(outContrast)
        outContrast = F.normalize(outContrast, dim=1)
        outPred = F.leaky_relu(self.linear1(x))
        outPred = self.linear2(outPred)
    
        return outPred, outContrast



