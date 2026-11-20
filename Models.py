import torch
import torch.nn as nn
import torch.nn.functional as F

from Models_utils import (
    compl_rev_vector_torch, 
    compl_rev_x_seq, 
    compl_rev_matrix_torch,
    conv_out_size,
    LearnablePooling
    )



class Simple_MLP_seq(nn.Module):
    def __init__(self, inp_size=780, seq_size=16, hidden_sizes=[1024, 512, 128, 64], out_reshape=False, with_seq=False):
        super().__init__()

        out_size = 8 if out_reshape else 16

        combined_size = inp_size + seq_size if with_seq else inp_size

        self.mlp = nn.Sequential(
            nn.Linear(combined_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[3], out_size)
        )

        self.out_reshape = out_reshape
        self.with_seq = with_seq

        if self.with_seq:
            self.seq_net = nn.Sequential(
                nn.Linear(seq_size, seq_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            )

    def forward(self, x, x_seq=None):
        if self.with_seq:
            if x_seq is None:
                raise ValueError("x_seq is required when with_seq=True.")
                
            # sequence x
            x_seq = F.one_hot(x_seq, num_classes=4).float()
            x_seq = torch.flatten(x_seq, 1)
            x_seq = self.seq_net(x_seq)
    
            # Join
            x = torch.cat((x, x_seq), dim=1)

        logits = self.mlp(x)
        
        if self.out_reshape:
            # reshapes output to Bx2x4 (two rows of logits, 1 per nucleotide in step)
            logits = logits.reshape(-1, 2, 4)

        return logits



class CR_MLP_seq(nn.Module):
    def __init__(
        self, 
        base_mapping,
        step_mapping,
        inp_size=780, 
        seq_size=16,
        hidden_sizes=[1024, 512, 128, 64], 
        out_reshape=False,
        with_seq=False
    ):
        super().__init__()
        
        out_size = 8 if out_reshape else 16

        combined_size = inp_size + seq_size if with_seq else inp_size
        
        self.mlp1 = nn.Sequential(
            nn.Linear(combined_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[3], out_size)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(combined_size + out_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[3], out_size)
        )

        # helper functions 
        self.get_CR_vec = compl_rev_vector_torch
        self.get_CR_seq = compl_rev_x_seq

        # options
        self.out_reshape = out_reshape
        self.CR_mapping = base_mapping if self.out_reshape else step_mapping
        self.with_seq = with_seq

        # add-ons
        self.base_mapping = base_mapping

        if self.with_seq:
            self.seq_net = nn.Sequential(
                nn.Linear(seq_size, seq_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            )

            self.seq_net_CR = nn.Sequential(
                nn.Linear(seq_size, seq_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            )

    def forward(self, x, x_seq=None):
        if self.with_seq:
            if x_seq is None:
                raise ValueError("x_seq is required when with_seq=True.")
                
            # process sequence x
            def _proc_seq(seq):
                seq = F.one_hot(seq, num_classes=4).float()
                return torch.flatten(seq, 1)

            # pass sequence x and its CR through small seq_nets
            x_seq1 = self.seq_net(_proc_seq(x_seq))
            x_seq2 = self.seq_net_CR(_proc_seq(self.get_CR_seq(x_seq, self.base_mapping)))

            # Join x with x_seq
            x1 = torch.cat((x, x_seq1), dim=1)
        else:
            x1 = x
            
        # Calc logits for first strand
        logits1 = self.mlp1(x1)

        # Get CR x
        x2 = self.get_CR_vec(x, seq_len=4)

        # Combine CR x with logits from first strand
        x2 = torch.cat((x2, logits1), dim=1)
        
        # join CR x with CR x_seq
        if self.with_seq:
            x2 = torch.cat((x2, x_seq2), dim=1)

        # Calc logits for second strand
        logits2 = self.mlp2(x2)
        
        if self.out_reshape:
            # reshapes output to Bx2x4 (two rows of logits, 1 per nucleotide in step)
            logits2 = logits2.reshape(-1, 2, 4)
            logits2 = logits2.flip(dims=[1])
            logits2 = logits2[..., list(self.CR_mapping.values())]
        else:
            logits2 = logits2[..., list(self.CR_mapping.values())]
            
        return logits2



class CRds_MLP_seq(nn.Module):
    def __init__(
        self, 
        base_mapping,
        step_mapping, 
        inp_size=780, 
        seq_size=16,
        hidden_sizes=[1024, 512, 128, 64], 
        out_reshape=False, 
        logits_dropout=0.0, 
        detach_logits=False,
        with_seq=False
    ):
        super().__init__()
        
        out_size = 8 if out_reshape else 16

        combined_size = inp_size + seq_size if with_seq else inp_size
        
        self.mlp1 = nn.Sequential(
            nn.Linear(combined_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[3], out_size)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(combined_size + out_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[3], out_size)
        )

        self.logits_dropout = nn.Dropout(p=logits_dropout)

        # helper functions
        self.get_CR_vec = compl_rev_vector_torch
        self.get_CR_seq = compl_rev_x_seq
        
        # options 
        self.out_reshape = out_reshape
        self.CR_mapping = base_mapping if self.out_reshape else step_mapping
        self.detach_logits = detach_logits
        self.with_seq = with_seq

        # add-ons
        self.base_mapping = base_mapping

        if self.with_seq:
            self.seq_net = nn.Sequential(
                nn.Linear(seq_size, seq_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            )

            self.seq_net_CR = nn.Sequential(
                nn.Linear(seq_size, seq_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            )

    def forward(self, x, x_seq=None):
        if self.with_seq:
            if x_seq is None:
                raise ValueError("x_seq is required when with_seq=True.")
                
            # process sequence x
            def _proc_seq(seq):
                seq = F.one_hot(seq, num_classes=4).float()
                return torch.flatten(seq, 1)

            # pass sequence x and its CR through small seq_nets
            x_seq1 = self.seq_net(_proc_seq(x_seq))
            x_seq2 = self.seq_net_CR(_proc_seq(self.get_CR_seq(x_seq, self.base_mapping)))

            # Join x with x_seq
            x1 = torch.cat((x, x_seq1), dim=1)
        else:
            x1 = x

        # Calc logits for first strand
        logits1 = self.mlp1(x1)

        # Get CR x
        x2 = self.get_CR_vec(x, seq_len=4)

        # Combine CR x with logits from first strand
        if self.detach_logits:
            x2 = torch.cat((x2, self.logits_dropout(logits1.detach())), dim=1)
        else:
            x2 = torch.cat((x2, self.logits_dropout(logits1)), dim=1)

        # join CR x with CR x_seq
        if self.with_seq:
            x2 = torch.cat((x2, x_seq2), dim=1)

        # Calc logits for second strand
        logits2 = self.mlp2(x2)
        
        if self.out_reshape:
            # reshapes output to Bx2x4 (two rows of logits, 1 per nucleotide in step)
            logits1 = logits1.reshape(-1, 2, 4)            
            logits2 = logits2.reshape(-1, 2, 4)

            # reorder logits2 for complementary reversed output
            logits2 = logits2.flip(dims=[1])
            logits2 = logits2[..., list(self.CR_mapping.values())]

        else:
            logits2 = logits2[..., list(self.CR_mapping.values())]

        return torch.stack((logits1, logits2), dim=1)



class CNN_2D_seq(nn.Module):
    def __init__(
        self, 
        input_size=40,
        seq_size=16,  
        hidden_sizes=[128, 64],
        out_reshape=False,
        with_seq=False,
        conv_channels=[16, 32, 64],
        kern_sizes=[(3, 3), (3, 3), (3, 3)],
        stride=[(1, 1), (1, 1), (1, 1)], 
        padding=[(0, 0), (0, 0), (0, 0)], 
        diletion=[(1, 1), (1, 1), (1, 1)],
        pool_bias=False,
        pool_act=None,
        pool_normW=False
    ):
        super().__init__()

        out_size = 8 if out_reshape else 16

        seq_size = seq_size if with_seq else 0

        conv_out_dim = conv_out_size(
            inp_size=input_size,
            kernels=kern_sizes,
            strides=stride,
            paddings=padding,
            diletions=diletion,
            reductions=[(1, 1), (2, 2), (1, 1)]   # factor by which sizes are reduced after each convolution layers (eg. poolings)
        )

        pool_out_dim = tuple(i // 2 for i in conv_out_dim)

        self.convolution = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=kern_sizes[0], padding=0),   # 36 x 36
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=kern_sizes[1], padding=0),   # 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=kern_sizes[2], padding=0),   # 6 x 6
            nn.ReLU(),
            nn.Dropout2d(0.2),
            LearnablePooling(conv_out_dim, pool_out_dim, bias=pool_bias, activation=pool_act, norm_weights=pool_normW)
        )

        self.mlp = nn.Sequential(
            nn.Linear(pool_out_dim[0] * pool_out_dim[1] * conv_channels[-1] + seq_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], out_size),
        )
        
        # Options
        self.out_reshape = out_reshape
        self.with_seq = with_seq

        if self.with_seq:
            self.seq_net = nn.Sequential(
                nn.Linear(seq_size, seq_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            )
    
    def forward(self, x, x_seq=None):
        # Distance x -- convolution
        x = x.unsqueeze(1)
        x = self.convolution(x)
        x = torch.flatten(x, 1)

        if self.with_seq:
            if x_seq is None:
                raise ValueError("x_seq is required when with_seq=True.")
                
            # sequence x
            x_seq = F.one_hot(x_seq, num_classes=4).float()
            x_seq = torch.flatten(x_seq, 1)
            x_seq = self.seq_net(x_seq)
    
            # Join
            x = torch.cat((x, x_seq), dim=1)

        # MLP
        logits = self.mlp(x)
        
        if self.out_reshape:
            # reshapes output to Bx2x4 (two rows of logits, 1 per nucleotide in step)
            logits = logits.reshape(-1, 2, 4)

        return logits



class CR_CNN_2D_seq(nn.Module):
    def __init__(
        self, 
        base_mapping,
        step_mapping,
        input_size=40,
        seq_size=16,  
        hidden_sizes=[128, 64],
        out_reshape=False,
        with_seq=False,
        conv_channels=[16, 32, 64],
        kern_sizes=[(3, 3), (3, 3), (3, 3)],
        stride=[(1, 1), (1, 1), (1, 1)], 
        padding=[(0, 0), (0, 0), (0, 0)], 
        diletion=[(1, 1), (1, 1), (1, 1)],
        pool_bias=False,
        pool_act=None,
        pool_normW=False
    ):
        super().__init__()

        out_size = 8 if out_reshape else 16

        seq_size = seq_size if with_seq else 0

        conv_out_dim = conv_out_size(
            inp_size=input_size,
            kernels=kern_sizes,
            strides=stride,
            paddings=padding,
            diletions=diletion,
            reductions=[(1, 1), (2, 2), (1, 1)]   # factor by which sizes are reduced after each convolution layers (eg. poolings)
        )

        pool_out_dim = tuple(i // 2 for i in conv_out_dim)

        self.convolution1 = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=kern_sizes[0], padding=0),   # 36 x 36
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=kern_sizes[1], padding=0),   # 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=kern_sizes[2], padding=0),   # 6 x 6
            nn.ReLU(),
            nn.Dropout2d(0.2),
            LearnablePooling(conv_out_dim, pool_out_dim, bias=pool_bias, activation=pool_act, norm_weights=pool_normW)
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(pool_out_dim[0] * pool_out_dim[1] * conv_channels[-1] + seq_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], out_size),
        )

        self.convolution2 = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=kern_sizes[0], padding=0),   # 36 x 36
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=kern_sizes[1], padding=0),   # 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=kern_sizes[2], padding=0),   # 6 x 6
            nn.ReLU(),
            nn.Dropout2d(0.2),
            LearnablePooling(conv_out_dim, pool_out_dim, bias=False, activation=None, norm_weights=False)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(pool_out_dim[0] * pool_out_dim[1] * conv_channels[-1] + seq_size + out_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], out_size),
        )

        # helper functions
        self.get_CR_mat = compl_rev_matrix_torch
        self.get_CR_seq = compl_rev_x_seq
        
        # Options
        self.out_reshape = out_reshape
        self.CR_mapping = base_mapping if self.out_reshape else step_mapping
        self.with_seq = with_seq
        
        # add-ons
        self.base_mapping = base_mapping

        if self.with_seq:
            self.seq_net = nn.Sequential(
                nn.Linear(seq_size, seq_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            )
            
            self.seq_net_CR = nn.Sequential(
                nn.Linear(seq_size, seq_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            )
            
    
    def forward(self, x, x_seq=None):
        # Distance x -- convolution1
        x1 = x.unsqueeze(1)
        x1 = self.convolution1(x1)
        x1 = torch.flatten(x1, 1)

        if self.with_seq:
            if x_seq is None:
                raise ValueError("x_seq is required when with_seq=True.")
                
            # process sequence x
            def _proc_seq(seq):
                seq = F.one_hot(seq, num_classes=4).float()
                return torch.flatten(seq, 1)

            # pass sequence x and its CR through small seq_nets
            x_seq1 = self.seq_net(_proc_seq(x_seq))
            x_seq2 = self.seq_net_CR(_proc_seq(self.get_CR_seq(x_seq, self.base_mapping)))
    
            # Join x after convolution1 with x_seq
            x1_join = torch.cat((x1, x_seq1), dim=1)
        else:
            x1_join = x1

        # Calc logits for first strand
        logits1 = self.mlp1(x1_join)

        # Get CR x matrix
        x2 = self.get_CR_mat(x, seq_len=4)
        
        # Distance x -- convolution2
        x2 = x2.unsqueeze(1)
        x2 = self.convolution1(x2)
        x2 = torch.flatten(x2, 1)
        
        # Combine CR x after convolution2 with logits from first strand
        x2 = torch.cat((x2, logits1), dim=1)

        # join CR x after convolution2 with CR x_seq
        if self.with_seq:
            x2 = torch.cat((x2, x_seq2), dim=1)

        # Calc logits for second strand
        logits2 = self.mlp2(x2)
        
        if self.out_reshape:
            # reshapes output to Bx2x4 (two rows of logits, 1 per nucleotide in step)
            logits2 = logits2.reshape(-1, 2, 4)
            logits2 = logits2.flip(dims=[1])
            logits2 = logits2[..., list(self.CR_mapping.values())]
        else:
            logits2 = logits2[..., list(self.CR_mapping.values())]
            
        return logits2



class CRds_CNN_2D_seq(nn.Module):
    def __init__(
        self, 
        base_mapping,
        step_mapping,
        input_size=40,
        seq_size=16,  
        hidden_sizes=[128, 64],
        out_reshape=False,
        logits_dropout=0.0,
        detach_logits=False,
        with_seq=False,
        conv_channels=[16, 32, 64],
        kern_sizes=[(3, 3), (3, 3), (3, 3)],
        stride=[(1, 1), (1, 1), (1, 1)], 
        padding=[(0, 0), (0, 0), (0, 0)], 
        diletion=[(1, 1), (1, 1), (1, 1)],
        pool_bias=False,
        pool_act=None,
        pool_normW=False
    ):
        super().__init__()

        out_size = 8 if out_reshape else 16

        seq_size = seq_size if with_seq else 0

        conv_out_dim = conv_out_size(
            inp_size=input_size,
            kernels=kern_sizes,
            strides=stride,
            paddings=padding,
            diletions=diletion,
            reductions=[(1, 1), (2, 2), (1, 1)]   # factor by which sizes are reduced after each convolution layers (eg. poolings)
        )

        pool_out_dim = tuple(i // 2 for i in conv_out_dim)

        self.convolution1 = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=kern_sizes[0], padding=0),   # 36 x 36
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=kern_sizes[1], padding=0),   # 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=kern_sizes[2], padding=0),   # 6 x 6
            nn.ReLU(),
            nn.Dropout2d(0.2),
            LearnablePooling(conv_out_dim, pool_out_dim, bias=False, activation=None, norm_weights=False)
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(pool_out_dim[0] * pool_out_dim[1] * conv_channels[-1] + seq_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], out_size),
        )

        self.convolution2 = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=kern_sizes[0], padding=0),   # 36 x 36
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=kern_sizes[1], padding=0),   # 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=kern_sizes[2], padding=0),   # 6 x 6
            nn.ReLU(),
            nn.Dropout2d(0.2),
            LearnablePooling(conv_out_dim, pool_out_dim, bias=pool_bias, activation=pool_act, norm_weights=pool_normW)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(pool_out_dim[0] * pool_out_dim[1] * conv_channels[-1] + seq_size + out_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], out_size),
        )

        self.logits_dropout = nn.Dropout(p=logits_dropout)

        # helper functions
        self.get_CR_mat = compl_rev_matrix_torch
        self.get_CR_seq = compl_rev_x_seq
        
        # Options
        self.out_reshape = out_reshape
        self.CR_mapping = base_mapping if self.out_reshape else step_mapping
        self.detach_logits = detach_logits
        self.with_seq = with_seq
        
        # add-ons
        self.base_mapping = base_mapping

        if self.with_seq:
            self.seq_net = nn.Sequential(
                nn.Linear(seq_size, seq_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            )
            
            self.seq_net_CR = nn.Sequential(
                nn.Linear(seq_size, seq_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            )
            
    
    def forward(self, x, x_seq=None):
        # Distance x -- convolution1
        x1 = x.unsqueeze(1)
        x1 = self.convolution1(x1)
        x1 = torch.flatten(x1, 1)

        if self.with_seq:
            if x_seq is None:
                raise ValueError("x_seq is required when with_seq=True.")
                
            # process sequence x
            def _proc_seq(seq):
                seq = F.one_hot(seq, num_classes=4).float()
                return torch.flatten(seq, 1)

            # pass sequence x and its CR through small seq_nets
            x_seq1 = self.seq_net(_proc_seq(x_seq))
            x_seq2 = self.seq_net_CR(_proc_seq(self.get_CR_seq(x_seq, self.base_mapping)))
    
            # Join x after convolution1 with x_seq
            x1_join = torch.cat((x1, x_seq1), dim=1)
        else:
            x1_join = x1

        # Calc logits for first strand
        logits1 = self.mlp1(x1_join)

        # Get CR x matrix
        x2 = self.get_CR_mat(x, seq_len=4)
        
        # Distance x -- convolution2
        x2 = x2.unsqueeze(1)
        x2 = self.convolution1(x2)
        x2 = torch.flatten(x2, 1)
        
        # Combine CR x after convolution2 with logits from first strand
        if self.detach_logits:
            x2 = torch.cat((x2, self.logits_dropout(logits1.detach())), dim=1)
        else:
            x2 = torch.cat((x2, self.logits_dropout(logits1)), dim=1)

        # join CR x after convolution2 with CR x_seq
        if self.with_seq:
            x2 = torch.cat((x2, x_seq2), dim=1)

        # Calc logits for second strand
        logits2 = self.mlp2(x2)
        
        if self.out_reshape:
            # reshapes output to Bx2x4 (two rows of logits, 1 per nucleotide in step)
            logits1 = logits1.reshape(-1, 2, 4)
            logits2 = logits2.reshape(-1, 2, 4)

            # reorder logits2 for complementary reversed output
            logits2 = logits2.flip(dims=[1])
            logits2 = logits2[..., list(self.CR_mapping.values())]
        else:
            logits2 = logits2[..., list(self.CR_mapping.values())]
            
        return torch.stack((logits1, logits2), dim=1)