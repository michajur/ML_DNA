import torch
import torch.nn as nn
import torch.nn.functional as F


########################
### HELPER FUNCTIONS ###
########################

def compl_rev_vector_torch(vec, at_per_nt=5, seq_len=6):
    """
    Get upper-triangular distance vectors of complementary-reversed sequences.

    Parameters
    ----------
    vec : torch.Tensor
        Upper-triangular distance vectors of shape (n, m*(m-1)//2).
    at_per_nt : int
        Number of atoms per nucleotide.
    seq_len : int
        Sequence length.

    Returns
    -------
    torch.Tensor
        Complementary-reversed vectors of the same shape.
    """
    half = at_per_nt * seq_len
    m = 2 * half

    if vec.shape[-1] != m * (m - 1) // 2:
        raise ValueError("Vector size does not match at_per_nt and seq_len.")

    # Upper-triangular indices of the output matrix
    rows, cols = torch.triu_indices(m, m, offset=1, device=vec.device)

    # Corresponding indices in the original matrix
    rows = (rows + half) % m
    cols = (cols + half) % m

    # Ensure indices refer to the upper triangle
    i = torch.minimum(rows, cols)
    j = torch.maximum(rows, cols)

    # Convert matrix indices to positions in the flattened upper triangle
    indices = i * (2 * m - i - 1) // 2 + (j - i - 1)

    return vec[..., indices]



def compl_rev_x_seq(tens, mapping):
    lookup = torch.tensor(
        [mapping[i] for i in range(4)],
        device=tens.device,
        dtype=tens.dtype
    )
    return lookup[tens.flip(dims=[1])]



def compl_rev_matrix_torch(mat, at_per_nt=5, seq_len=6):
    """
    Get distance matrix of complementary reversed sequence.

    Parameters
    ----------
    mat : torch.Tensor
        Distance matrices of shape (n, m, m).
    at_per_nt : int
        Number of atoms per nucleotide.
    seq_len : int
        Sequence length.

    Returns
    -------
    torch.Tensor
        Complementary reversed distance matrices.
    """
    submat_size = at_per_nt * seq_len

    # Top left
    submat1 = mat[:, :submat_size, :submat_size]

    # Top right
    submat2 = mat[:, :submat_size, submat_size:]

    # Bottom left
    submat3 = mat[:, submat_size:, :submat_size]

    # Bottom right
    submat4 = mat[:, submat_size:, submat_size:]

    hstack1 = torch.cat([submat4, submat3], dim=2)
    hstack2 = torch.cat([submat2, submat1], dim=2)

    vstack = torch.cat([hstack1, hstack2], dim=1)

    return vstack



def conv_out_size(
    inp_size,
    kernels,
    strides,
    paddings,
    diletions,
    reductions
):
    if any(
        [
            len(kernels) != len(i) 
            for i in [strides, paddings, diletions, reductions]
        ]
    ):
        raise ValueError("kernels, strides, paddings, diletions, and\
                          reductions entries must have the same size.")


    if isinstance(inp_size, int):
        inp_size = (inp_size, inp_size)
            
    for k, s, p, d, r in zip(
        kernels, 
        strides, 
        paddings, 
        diletions, 
        reductions
    ):
        out1 = ((inp_size[0] + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1) // r[0]
        out2 = ((inp_size[1] + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1) // r[1]
        inp_size = (int(out1), int(out2))

    return inp_size


#############################
### CUSTOM LOSS FUNCTIONS ###
#############################

class EnhancedLoss(nn.Module):
    """
    To enhance A/G and C/T distinction this loss funct. adds 
    additional penalty when A and G or C and T are confused.
    
    logits:  (batch, 2, 4)
    targets: (batch, 2), integer class indices

    Encoding: A=0, G=1, C=2, T=3 -- must match base_dict 
    """
    
    def __init__(self, pair_weight=0.5):
        super().__init__()
        self.pair_weight = pair_weight


    def forward(self, logits, targets):
    
        # Standard loss -- two positions, 4 classes
        main_loss = F.cross_entropy(
            logits.reshape(-1, 4),
            targets.reshape(-1)
        )
    
        # Select the relevant pair: A/G or C/T -- get 
        # only two logits for each position: either 
        # corresponding to A/G or C/T
        pair_start = (targets // 2) * 2
        pair_indices = torch.stack(
            [pair_start, pair_start + 1], dim=-1
        )
    
        pair_logits = torch.gather(logits, -1, pair_indices)
    
        # Make new pairwise class encoding: A and C=0, G and T=1
        pair_targets = targets % 2
    
        # Calculate addtional loss within A/G and C/T pairs
        pair_loss = F.cross_entropy(
            pair_logits.reshape(-1, 2),
            pair_targets.reshape(-1)
        )
    
        return main_loss + self.pair_weight * pair_loss



class CRConsistencyLoss(nn.Module):
    def __init__(self, consistency_weight=0.1):
        super().__init__()
        self.consistency_weight = consistency_weight

    def forward(self, logits, targets):
        """
        logits:
            Joined:   (B, 2, 16)
            Separate: (B, 2, 2, 4)

        targets:
            Joined:   (B,)
            Separate: (B, 2)
        """

        logits1, logits2 = logits.unbind(dim=1)

        # Classification losses for both predictions
        loss1 = F.cross_entropy(
            logits1.reshape(-1, logits1.shape[-1]),
            targets.reshape(-1)
        )

        loss2 = F.cross_entropy(
            logits2.reshape(-1, logits2.shape[-1]),
            targets.reshape(-1)
        )

        # Log probabilities and probabilities
        log_p1 = F.log_softmax(logits1, dim=-1)
        log_p2 = F.log_softmax(logits2, dim=-1)

        p1 = log_p1.exp()
        p2 = log_p2.exp()

        # Symmetric KL divergence
        kl12 = (p1 * (log_p1 - log_p2)).sum(dim=-1).mean()
        kl21 = (p2 * (log_p2 - log_p1)).sum(dim=-1).mean()

        consistency_loss = 0.5 * (kl12 + kl21)

        return (
            0.5 * (loss1 + loss2)
            + self.consistency_weight * consistency_loss
        )


#################################
### ADDITIONAL CUSTOM MODULES ###
#################################



class LearnablePooling(nn.Module):
    def __init__(self, input_size, output_size, bias=False, activation=None, norm_weights=False):
        super().__init__()

        h_in, w_in = input_size
        h_out, w_out = output_size

        self.W1 = nn.Parameter(torch.empty(w_in, h_out))
        self.W2 = nn.Parameter(torch.empty(h_in, w_out))

        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

        if bias:
            self.bias = nn.Parameter(torch.zeros(h_out, w_out))
        else:
            self.register_parameter("bias", None)

        self.activation = activation
        self.norm_weights = norm_weights

    def forward(self, x):
        # x: (..., H_in, W_in)

        if self.norm_weights:
            W1 = torch.softmax(self.W1, dim=0)
            W2 = torch.softmax(self.W2, dim=0)

        x = x @ W1 if self.norm_weights else x @ self.W1
        x = x.transpose(-2, -1)
        x = x @ W2 if self.norm_weights else x @ self.W2

        if self.bias is not None:
            x = x + self.bias

        if self.activation is not None:
            x = self.activation(x)

        return x