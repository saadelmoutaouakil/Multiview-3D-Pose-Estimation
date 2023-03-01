from __future__ import absolute_import
import numpy as np
import torch.nn as nn
import torch 
from models.sem_graph_conv import SemGraphConv
from models.graph_non_local import GraphNonLocal


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        #self.nonlocal = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        #out = self.nonlocal(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class SemGCN_Concat_Non_Shared(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN_Concat_Non_Shared, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []
        _concat_layers = []

        if nodes_group is None:
            for i in range(int(num_layers/2)):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _concat_layers.append(_ResGraphConv(adj, 4*hid_dim, 4*hid_dim, 4*hid_dim, p_dropout=p_dropout))
           

        first_half_arch = _gconv_layers
        last_half_arch = _concat_layers
        self.gconv_input1 = nn.Sequential(*_gconv_input)
        self.gconv_input2 = nn.Sequential(*_gconv_input)
        self.gconv_input3 = nn.Sequential(*_gconv_input)
        self.gconv_input4 = nn.Sequential(*_gconv_input)
        self.gconv_layers1 = nn.Sequential(*first_half_arch)
        self.gconv_layers2 = nn.Sequential(*first_half_arch)
        self.gconv_layers3 = nn.Sequential(*first_half_arch)
        self.gconv_layers4 = nn.Sequential(*first_half_arch)
        self.gconv_layer_post_merge = nn.Sequential(*last_half_arch)
        self.gconv_output = SemGraphConv(4*hid_dim, coords_dim[1], adj)

    def forward(self, x_1,x_2,x_3,x_4):
        out_1 = self.gconv_input1(x_1)
        out_2 = self.gconv_input2(x_2)
        out_3 = self.gconv_input3(x_3)
        out_4 = self.gconv_input4(x_4)
        
        out_1 = self.gconv_layers1(out_1)
        out_2 = self.gconv_layers2(out_2)
        out_3 = self.gconv_layers3(out_3)
        out_4 = self.gconv_layers4(out_4)
        
        out_merge = torch.cat((out_1,out_2,out_3,out_4),dim=2)
        
        out_merge = self.gconv_layer_post_merge(out_merge)
        
        out = self.gconv_output(out_merge)
        return out


class SemGCN_Concat_Shared(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN_Concat_Shared, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []
        _concat_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _concat_layers.append(_ResGraphConv(adj, 4*hid_dim, 4*hid_dim, 4*hid_dim, p_dropout=p_dropout))
        

        first_half_arch = _gconv_layers
        last_half_arch = _concat_layers
        
        self.gconv_input1 = nn.Sequential(*_gconv_input)
        self.gconv_input2 = nn.Sequential(*_gconv_input)
        self.gconv_input3 = nn.Sequential(*_gconv_input)
        self.gconv_input4 = nn.Sequential(*_gconv_input)
        
        self.gconv_layers = nn.Sequential(*first_half_arch)
        

        self.gconv_layer_post_merge = nn.Sequential(*last_half_arch)
        self.gconv_output = SemGraphConv(4*hid_dim, coords_dim[1], adj)

    def forward(self, x_1,x_2,x_3,x_4):
        out_1 = self.gconv_input1(x_1)
        out_2 = self.gconv_input2(x_2)
        out_3 = self.gconv_input3(x_3)
        out_4 = self.gconv_input4(x_4)
        
        out_1 = self.gconv_layers(out_1)
        out_2 = self.gconv_layers(out_2)
        out_3 = self.gconv_layers(out_3)
        out_4 = self.gconv_layers(out_4)
        
        out_merge = torch.cat((out_1,out_2,out_3,out_4),dim=2)
        
        out_merge = self.gconv_layer_post_merge(out_merge)
        out = self.gconv_output(out_merge)
        return out

class SemGCN_Sum_Shared(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN_Sum_Shared, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []
        _concat_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _concat_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        

        first_half_arch = _gconv_layers
        last_half_arch = _concat_layers
        
        self.gconv_input1 = nn.Sequential(*_gconv_input)
        self.gconv_input2 = nn.Sequential(*_gconv_input)
        self.gconv_input3 = nn.Sequential(*_gconv_input)
        self.gconv_input4 = nn.Sequential(*_gconv_input)
        
        self.gconv_layers = nn.Sequential(*first_half_arch)
        

        self.gconv_layer_post_merge = nn.Sequential(*last_half_arch)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x_1,x_2,x_3,x_4):
        out_1 = self.gconv_input1(x_1)
        out_2 = self.gconv_input2(x_2)
        out_3 = self.gconv_input3(x_3)
        out_4 = self.gconv_input4(x_4)
        
        out_1 = self.gconv_layers(out_1)
        out_2 = self.gconv_layers(out_2)
        out_3 = self.gconv_layers(out_3)
        out_4 = self.gconv_layers(out_4)
        
        out_merge = out_1+out_2+out_3+out_4
        
        out_merge = self.gconv_layer_post_merge(out_merge)
        out = self.gconv_output(out_merge)
        return out

class SemGCN_Sum_Non_Shared(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN_Sum_Non_Shared, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []
        _concat_layers = []

        if nodes_group is None:
            for i in range(int(num_layers/2)):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _concat_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
           

        first_half_arch = _gconv_layers
        last_half_arch = _concat_layers
        self.gconv_input1 = nn.Sequential(*_gconv_input)
        self.gconv_input2 = nn.Sequential(*_gconv_input)
        self.gconv_input3 = nn.Sequential(*_gconv_input)
        self.gconv_input4 = nn.Sequential(*_gconv_input)
        self.gconv_layers1 = nn.Sequential(*first_half_arch)
        self.gconv_layers2 = nn.Sequential(*first_half_arch)
        self.gconv_layers3 = nn.Sequential(*first_half_arch)
        self.gconv_layers4 = nn.Sequential(*first_half_arch)
        self.gconv_layer_post_merge = nn.Sequential(*last_half_arch)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x_1,x_2,x_3,x_4):
        out_1 = self.gconv_input1(x_1)
        out_2 = self.gconv_input2(x_2)
        out_3 = self.gconv_input3(x_3)
        out_4 = self.gconv_input4(x_4)
        
        out_1 = self.gconv_layers1(out_1)
        out_2 = self.gconv_layers2(out_2)
        out_3 = self.gconv_layers3(out_3)
        out_4 = self.gconv_layers4(out_4)
        
        out_merge = out_1+out_2+out_3+out_4
        
        out_merge = self.gconv_layer_post_merge(out_merge)
        
        out = self.gconv_output(out_merge)
        return out



class SemGCN_Camera_Concat_World(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN_Camera_Concat_World, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_input_Concat = [_GraphConv(adj, coords_dim[1]*4, hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []


        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                

            
        


        
        self.gconv_input1 = nn.Sequential(*_gconv_input)
        self.gconv_input2 = nn.Sequential(*_gconv_input)
        self.gconv_input3 = nn.Sequential(*_gconv_input)
        self.gconv_input4 = nn.Sequential(*_gconv_input)
        self.gconv_input_Concat = nn.Sequential(*_gconv_input_Concat)
        
        self.gconv_layers1 = nn.Sequential(*_gconv_layers)
        self.gconv_layers2 = nn.Sequential(*_gconv_layers)
        self.gconv_layers3 = nn.Sequential(*_gconv_layers)
        self.gconv_layers4 = nn.Sequential(*_gconv_layers)
        self.gconv_layers_Concat = nn.Sequential(*_gconv_layers)
        

        
        self.gconv_output_1 = SemGraphConv(hid_dim, coords_dim[1], adj)
        self.gconv_output_2 = SemGraphConv(hid_dim, coords_dim[1], adj)
        self.gconv_output_3 = SemGraphConv(hid_dim, coords_dim[1], adj)
        self.gconv_output_4 = SemGraphConv(hid_dim, coords_dim[1], adj)
        self.gconv_output_final = SemGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x_1,x_2,x_3,x_4):
        out_1 = self.gconv_input1(x_1)
        out_1 = self.gconv_layers1(out_1)
        out_1 = self.gconv_output_1(out_1)
        
        out_2 = self.gconv_input2(x_2)
        out_2 = self.gconv_layers2(out_2)
        out_2 = self.gconv_output_2(out_2)
        
        out_3 = self.gconv_input3(x_3)
        out_3 = self.gconv_layers3(out_3)
        out_3 = self.gconv_output_3(out_3)
        
        out_4 = self.gconv_input4(x_4)
        out_4 = self.gconv_layers4(out_4)
        out_4 = self.gconv_output_4(out_4)
        
        out_merge = torch.cat((out_1,out_2,out_3,out_4),dim=2)
        
        out_final = self.gconv_input_Concat(out_merge)
        out_final = self.gconv_layers_Concat(out_final)
        out_final = self.gconv_output_final(out_final)
        

        return out_final,out_1,out_2,out_3,out_4