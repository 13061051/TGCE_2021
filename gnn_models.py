import dgl
import numpy as np
import torch as th
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import math
import torch as th
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
import torch.nn.functional as F
from collections import Counter

"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity


def compute_position_encode():
	max_len=512
	d_model=8
	pe = torch.zeros(max_len, d_model)
	position = torch.arange(0, max_len).unsqueeze(1)
	div_term = torch.exp(torch.arange(0, d_model, 2) *
						 -(math.log(10000.0) / d_model))
	pe[:, 0::2] = torch.sin(position * div_term)
	pe[:, 1::2] = torch.cos(position * div_term)
	#pe = pe.unsqueeze(0)
	return pe
pe=compute_position_encode()
print('pe shape',pe.shape)



import random
def trans_to_cuda(variable):
	if th.cuda.is_available():
		device = th.device('cuda:0')
		variable = variable.to(device)
	else:
		pp=-1
	return variable

import torch
import torch.nn as nn
import torch.nn.functional as F

class ONAN(nn.Module):
	def __init__(self, input_dim, output_dim, dropout=0.0,activation=None, batch_norm=True):
		super().__init__()
		self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
		self.gru = nn.GRU(input_dim, input_dim,dropout=dropout, batch_first=True)
		self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
		self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
		self.activation = activation
		print('make ONAN Layer')
	def reducer(self, nodes):
		m = nodes.mailbox['m']
		_, hn = self.gru(m)  # hn: (1, batch_size, d)
		return {'neigh': hn.squeeze(0)}

	def forward(self, mg, feat):
		with mg.local_scope():
			if self.batch_norm is not None:
				feat = self.batch_norm(feat)
			mg.ndata['ft'] = feat
			if mg.number_of_edges() > 0:
				mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
				neigh = mg.ndata['neigh']
				rst = self.fc_self(feat) + self.fc_neigh(neigh)
			else:
				rst = self.fc_self(feat)
			if self.activation is not None:
				rst = self.activation(rst)
			return rst

class ONAN_ATT(nn.Module):
	def __init__(self, input_dim, output_dim, dropout=0.0,activation=None, batch_norm=True):
		super().__init__()
		self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
		self.gru = nn.GRU(input_dim, input_dim,dropout=dropout, batch_first=True)
		self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
		self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
		self.activation = activation
		self.dim=input_dim
		self.K = nn.Linear(self.dim, self.dim, bias=False)
		self.Q = nn.Linear(self.dim, self.dim, bias=False)
		self.V = nn.Linear(self.dim, self.dim, bias=False)
		print('make ONAN_ATT Layer')
	def self_att_compute(self, x):
		# [seq, seq_hour, mask]
		q = self.Q(x)
		k = self.K(x)
		v = self.V(x)
		att = torch.bmm(q, k.permute(0, 2, 1))
		att = torch.softmax(att, dim=2)
		out = torch.bmm(att, v)
		out = torch.sum(out, dim=1)
		return out
	def reducer(self, nodes):
		m = nodes.mailbox['m']
		#hidden_, hn = self.gru(m)  # hn: (1, batch_size, d)
		#return {'neigh': hn.squeeze(0)}
		hn=self.self_att_compute(m)
		return {'neigh': hn}

	def forward(self, mg, feat):
		with mg.local_scope():
			if self.batch_norm is not None:
				feat = self.batch_norm(feat)
			mg.ndata['ft'] = feat
			if mg.number_of_edges() > 0:
				mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
				neigh = mg.ndata['neigh']
				rst = self.fc_self(feat) + self.fc_neigh(neigh)
			else:
				rst = self.fc_self(feat)
			if self.activation is not None:
				rst = self.activation(rst)
			return rst

class ONAN_GRU_ATT(nn.Module):
	def __init__(self, input_dim, output_dim, dropout=0.0,activation=None, batch_norm=True):
		super().__init__()
		self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
		self.gru = nn.GRU(input_dim, input_dim,dropout=dropout, batch_first=True)
		self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
		self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
		self.activation = activation
		self.dim=input_dim
		self.K = nn.Linear(self.dim, self.dim, bias=False)
		self.Q = nn.Linear(self.dim, self.dim, bias=False)
		self.V = nn.Linear(self.dim, self.dim, bias=False)
		print('make ONAN_GRU_ATT Layer')
	def self_att_compute(self, x):
		# [seq, seq_hour, mask]
		q = self.Q(x)
		k = self.K(x)
		v = self.V(x)
		att = torch.bmm(q, k.permute(0, 2, 1))
		att = torch.softmax(att, dim=2)
		out = torch.bmm(att, v)
		out = torch.sum(out, dim=1)
		return out
	def reducer(self, nodes):
		m = nodes.mailbox['m']
		hidden_, hn = self.gru(m)  # hn: (1, batch_size, d)
		#return {'neigh': hn.squeeze(0)}
		hn=self.self_att_compute(hidden_)
		return {'neigh': hn}

	def forward(self, mg, feat):
		with mg.local_scope():
			if self.batch_norm is not None:
				feat = self.batch_norm(feat)
			mg.ndata['ft'] = feat
			if mg.number_of_edges() > 0:
				mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
				neigh = mg.ndata['neigh']
				rst = self.fc_self(feat) + self.fc_neigh(neigh)
			else:
				rst = self.fc_self(feat)
			if self.activation is not None:
				rst = self.activation(rst)
			return rst

class ONAN_GRU_LAST_ATT(nn.Module):
	def __init__(self, input_dim, output_dim, dropout=0.0,activation=None, batch_norm=True):
		super().__init__()
		self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
		self.gru = nn.GRU(input_dim, input_dim,dropout=dropout, batch_first=True)
		self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
		self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
		self.activation = activation
		self.dim=input_dim
		#self.K = nn.Linear(self.dim, self.dim, bias=False)
		self.Q = nn.Linear(self.dim, self.dim, bias=False)
		#self.V = nn.Linear(self.dim, self.dim, bias=False)
		print('make ONAN_GRU_LAST_ATT Layer')
	def self_att_compute(self, x,x_last):
		# [seq, seq_hour, mask]
		q = self.Q(x_last)# batch*dim #batch*M*dim
		k = x     #batch*N*dim #batch*dim*N ->batch*M*N
		v = x     #batch*N*dim ->batch*M*dim
		q=torch.reshape(q,[q.shape[1],q.shape[0],q.shape[2]])
		att = torch.bmm(q, k.permute(0, 2, 1))
		att = torch.softmax(att, dim=2)
		out = torch.bmm(att, v)
		out = torch.sum(out, dim=1)
		return out
	def reducer(self, nodes):
		m = nodes.mailbox['m']
		hidden_, hn = self.gru(m)  # hn: (1, batch_size, d)
		#return {'neigh': hn.squeeze(0)}
		hn=self.self_att_compute(hidden_,x_last=hn)
		return {'neigh': hn}

	def forward(self, mg, feat):
		with mg.local_scope():
			if self.batch_norm is not None:
				feat = self.batch_norm(feat)
			mg.ndata['ft'] = feat
			if mg.number_of_edges() > 0:
				mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
				neigh = mg.ndata['neigh']
				rst = self.fc_self(feat) + self.fc_neigh(neigh)
			else:
				rst = self.fc_self(feat)
			if self.activation is not None:
				rst = self.activation(rst)
			return rst

class SGAT(nn.Module):
	def __init__(
		self, input_dim, hidden_dim, output_dim, activation=None, batch_norm=True
	):
		super().__init__()
		self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
		self.fc_q = nn.Linear(input_dim, hidden_dim, bias=True)
		self.fc_k = nn.Linear(input_dim, hidden_dim, bias=False)
		self.fc_v = nn.Linear(input_dim, output_dim, bias=False)
		self.attn_e = nn.Linear(hidden_dim, 1, bias=False)
		self.activation = activation

	def forward(self, sg, feat):
		with sg.local_scope():
			if self.batch_norm is not None:
				feat = self.batch_norm(feat)
			q = self.fc_q(feat)
			k = self.fc_k(feat)
			v = self.fc_v(feat)
			sg.ndata.update({'q': q, 'k': k, 'v': v})
			sg.apply_edges(fn.u_add_v('q', 'k', 'e'))
			e = self.attn_e(th.sigmoid(sg.edata['e']))
			sg.edata['a'] = edge_softmax(sg, e)
			sg.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'ft'))
			rst = sg.ndata['ft']
			if self.activation is not None:
				rst = self.activation(rst)
			return rst

class ONAN_Transformer(nn.Module):
	def __init__(self, input_dim, output_dim, dropout=0.0,activation=None, batch_norm=True):
		super().__init__()
		self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
		#self.gru = nn.GRU(input_dim, input_dim,dropout=dropout, batch_first=True)
		self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
		self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
		self.activation = activation
		self.dim=input_dim
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=1,dropout=dropout)
		print('make ONAN_Transformer Layer')

	def reducer(self, nodes):
		m = nodes.mailbox['m']
		#hidden_, hn = self.gru(m)  # hn: (1, batch_size, d)
		#return {'neigh': hn.squeeze(0)}
		hn=self.encoder_layer(m)
		hn=torch.mean(hn,dim=1)
		return {'neigh': hn}

	def forward(self, mg, feat):
		with mg.local_scope():
			if self.batch_norm is not None:
				feat = self.batch_norm(feat)
			mg.ndata['ft'] = feat
			if mg.number_of_edges() > 0:
				mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
				neigh = mg.ndata['neigh']
				rst = self.fc_self(feat) + self.fc_neigh(neigh)
			else:
				rst = self.fc_self(feat)
			if self.activation is not None:
				rst = self.activation(rst)
			return rst

class POS_GAT_Layer(nn.Module):
	def __init__(self,in_dim, out_dim,dropout=0.0):
		super(POS_GAT_Layer, self).__init__()
		#self.g = g
		# equation (1)
		self.fc = nn.Linear(in_dim, out_dim, bias=False)
		# equation (2)
		self.attn_fc = nn.Linear(2 * out_dim+16, 1, bias=False)
		self.feat_drop = nn.Dropout(dropout)
		self.attn_drop = nn.Dropout(dropout)
		self.reset_parameters()
		print('make pos GAT Layer')
	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('relu')
		nn.init.xavier_normal_(self.fc.weight, gain=gain)
		nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

	def edge_attention(self, edges):
		# edge UDF for equation (2)
		z2 = torch.cat([edges.src['z'], edges.dst['z'],edges.src['pos'],edges.dst['pos']], dim=1)
		a = self.attn_fc(z2)
		return {'e': F.leaky_relu(a)}

	def message_func(self, edges):
		# message UDF for equation (3) & (4)
		return {'z': edges.src['z'], 'e': edges.data['e']}

	def reduce_func(self, nodes):
		# reduce UDF for equation (3) & (4)
		# equation (3)
		#alpha的结构是一个1*N*1的结构

		alpha = F.softmax(nodes.mailbox['e'], dim=1)
		# equation (4)
		h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
		return {'h': h}

	def forward(self,g, h):
		feat = self.feat_drop(h)
		z = self.fc(feat)
		g.ndata['z'] = z
		# equation (2)
		g.apply_edges(self.edge_attention)
		# equation (3) & (4)
		g.update_all(self.message_func, self.reduce_func)
		return g.ndata.pop('h')

class GAT_Layer(nn.Module):
	def __init__(self,in_dim, out_dim,dropout=0.0):
		super(GAT_Layer, self).__init__()
		#self.g = g
		# equation (1)
		self.fc = nn.Linear(in_dim, out_dim, bias=False)
		# equation (2)
		self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
		self.feat_drop = nn.Dropout(dropout)
		self.attn_drop = nn.Dropout(dropout)
		self.reset_parameters()
		print('make pos GAT Layer')
	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('relu')
		nn.init.xavier_normal_(self.fc.weight, gain=gain)
		nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

	def edge_attention(self, edges):
		# edge UDF for equation (2)
		z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
		a = self.attn_fc(z2)
		return {'e': F.leaky_relu(a)}

	def message_func(self, edges):
		# message UDF for equation (3) & (4)
		return {'z': edges.src['z'], 'e': edges.data['e']}

	def reduce_func(self, nodes):
		# reduce UDF for equation (3) & (4)
		# equation (3)
		#alpha的结构是一个1*N*1的结构

		alpha = F.softmax(nodes.mailbox['e'], dim=1)
		# equation (4)
		h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
		return {'h': h}

	def forward(self,g, h):
		feat = self.feat_drop(h)
		z = self.fc(feat)
		g.ndata['z'] = z
		# equation (2)
		g.apply_edges(self.edge_attention)
		# equation (3) & (4)
		g.update_all(self.message_func, self.reduce_func)
		return g.ndata.pop('h')

from dgl.nn.pytorch.conv.graphconv import GraphConv
from dgl.nn.pytorch.conv.gatconv import GATConv
class GRU_GNN_Vector(nn.Module):
	def __init__(self, in_dim, hidden_dim,drop_out=0.0,gnn_function='GRU_POSGAT',dropout_fc=0.20):
		super(GRU_GNN_Vector, self).__init__()
		self.gnn_function=gnn_function
		self.pooling_method='ATT'
		if gnn_function=='GRU_2':
			self.layers = nn.ModuleList([
				ONAN(in_dim, hidden_dim,dropout=drop_out),
				ONAN(in_dim, hidden_dim, dropout=drop_out),
			])
		elif gnn_function=='GRU_1':
			self.layers = nn.ModuleList([
				ONAN(in_dim, hidden_dim,dropout=drop_out),
			])
		elif gnn_function=='Transformer_POSGAT':
			self.layers = nn.ModuleList([
				ONAN_Transformer(in_dim, hidden_dim, dropout=drop_out),
				POS_GAT_Layer(
					in_dim,
					hidden_dim,
					dropout=drop_out,
				),
			])
		elif gnn_function=='POSGAT_GRU':
			self.layers = nn.ModuleList([
				POS_GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
				ONAN(in_dim, hidden_dim, dropout=drop_out),
			])
		elif gnn_function=='GRU_POSGAT' or gnn_function=='GRU_POSGAT_AVG':
			self.layers = nn.ModuleList([
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				POS_GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
			])
		elif gnn_function=='GRU2_POSGAT':
			print('GRU2_POSGAT')
			self.layers = nn.ModuleList([
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				POS_GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
			])
		elif gnn_function=='GRU3_POSGAT':
			print('GRU3_POSGAT')
			self.layers = nn.ModuleList([
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				POS_GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
			])
		elif gnn_function=='GRU4_POSGAT':
			print('GRU4_POSGAT')
			self.layers = nn.ModuleList([
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				POS_GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
			])
		elif gnn_function=='GRU_GAT':
			self.layers = nn.ModuleList([
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
			])
		elif gnn_function=='GRU_GCN':
			self.layers = nn.ModuleList([
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				GraphConv(in_dim, hidden_dim),
			])
		elif gnn_function=='GAT_AVG':
			self.layers = nn.ModuleList([
				GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
				GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
			])
		elif gnn_function=='GAT_Global':
			self.layers = nn.ModuleList([
				GATConv(in_dim,hidden_dim,num_heads=4),
				GATConv(in_dim, hidden_dim, num_heads=4),
			])
		elif gnn_function=='GAT_Set2Set':
			self.layers = nn.ModuleList([
				GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
				GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
			])
		elif gnn_function=='POSGAT_2':
			self.layers = nn.ModuleList([
				POS_GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
				POS_GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
			])
		elif gnn_function=='POSGAT_1':
			self.layers = nn.ModuleList([
				POS_GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
			])
		else:
			self.layers = nn.ModuleList([
				ONAN(in_dim, hidden_dim, dropout=drop_out),
				POS_GAT_Layer(in_dim, hidden_dim, dropout=drop_out),
			])
		self.w=nn.Linear(in_dim+(hidden_dim*1),hidden_dim)
		pooling_gate_nn = nn.Linear(hidden_dim+hidden_dim, 1)
		if 'Set2Set' not in self.gnn_function:
			self.gat_pool=dgl.nn.pytorch.glob.GlobalAttentionPooling(gate_nn=pooling_gate_nn)
			self.classify = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
		else:
			self.gat_pool = dgl.nn.pytorch.glob.Set2Set(hidden_dim + hidden_dim, 2, 1)
			self.classify = nn.Linear(hidden_dim*4, hidden_dim)
		#self.set2set=dgl.nn.pytorch.glob.Set2Set(hidden_dim+hidden_dim,2,1)
		#self.classify = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
		#self.classify = nn.Linear(hidden_dim+hidden_dim, hidden_dim)
		self.dropout = nn.Dropout(dropout_fc)
	def forward(self,feat, g1,g2):
		# 对于无
		h=feat
		if self.gnn_function=='GRU_2' or self.gnn_function=='GAT_AVG' or self.gnn_function=='GAT_Set2Set' or self.gnn_function=='GRU_1':
			for i, layer in enumerate(self.layers):
				if i%2==0:
					h=layer(g1,h)
				else:
					h=layer(g1,h)
				h = F.relu(h)
		elif self.gnn_function=='POSGAT_2' or self.gnn_function=='POSGAT_1':
			for i, layer in enumerate(self.layers):
				if i%2==0:
					h=layer(g2,h)
				else:
					h=layer(g2,h)
				h = F.relu(h)
		elif self.gnn_function=='GAT_Global':
			for i, layer in enumerate(self.layers):
				if i%2==0:
					h=layer(g1,h)
					h=torch.mean(h,1)
				else:
					h=layer(g1,h)
					h = torch.mean(h, 1)
				h = F.relu(h)
		elif self.gnn_function=='POSGAT_GRU':
			for i, layer in enumerate(self.layers):
				if i%2==0:
					h=layer(g2,h)
				else:
					h=layer(g1,h)
				h = F.relu(h)
		elif self.gnn_function=='GRU2_POSGAT':
			for i, layer in enumerate(self.layers):
				if i == 0 or i==1:
					h = layer(g1, h)
				else:
					h = layer(g2, h)
				h = F.relu(h)
		elif self.gnn_function=='GRU3_POSGAT':
			for i, layer in enumerate(self.layers):
				if i == 0 or i==1 or i==2:
					h = layer(g1, h)
				else:
					h = layer(g2, h)
				h = F.relu(h)
		elif self.gnn_function=='GRU4_POSGAT':
			for i, layer in enumerate(self.layers):
				if i == 0 or i==1 or i==2 or i==3:
					h = layer(g1, h)
				else:
					h = layer(g2, h)
				h = F.relu(h)
		else:
			for i, layer in enumerate(self.layers):
				if i%2==0:
					h=layer(g1,h)
				else:
					h=layer(g2,h)
				h = F.relu(h)
		h1=torch.cat([feat,h],dim=1)
		#h1=self.w(feat)
		if 'AVG' in self.gnn_function:
			g1.ndata['h1'] = h1  ## 节点特征经过两层卷积的输出
			hg = dgl.mean_nodes(g1, 'h1')  # 图的特征是所有节点特征的均值
		else:
			hg=self.gat_pool(g1,h1)
		hg = self.dropout(hg)
		return self.classify(hg)



def min_(a,b):
	if a<b:
		return a
	else:
		return b


def seq_to_shortcut_random_graph(seq,edges_random):
	items = np.unique(seq)
	iid2nid = {iid: i for i, iid in enumerate(items)}
	num_nodes = len(items)

	# 计算出每个nid第一次出现的位置
	nid_pos = np.zeros([num_nodes])
	for i in range(len(seq)):
		position_index = i % 256
		iid = seq[i]
		nid = iid2nid[iid]
		if nid_pos[nid] == 0:
			nid_pos[nid] = int(position_index)
	position_data = np.zeros([num_nodes, 8])
	for k in range(num_nodes):
		position_data[k] = pe[int(nid_pos[k])]
	position_data = th.Tensor(position_data)

	g = dgl.DGLGraph()
	g.add_nodes(num_nodes)
	g.ndata['iid'] = th.Tensor(items)

	# position data for model
	g.ndata['pos'] = position_data

	seq_nid = [iid2nid[iid] for iid in seq]
	if len(seq) > 1:
		src = seq_nid[:-1]
		dst = seq_nid[1:]
		# edges are added in the order of their occurrences.
		g.add_edges(src, dst)
	g.add_edges(seq_nid, seq_nid)

	#add random walk edges
	edges_random=np.array(edges_random)
	src_random=[]
	dst_random=[]
	for i in range(len(edges_random)):
		src_random.append(iid2nid[edges_random[i][0]])
		dst_random.append(iid2nid[edges_random[i][1]])
	g.add_edges(src_random,dst_random)

	return g


def seq_to_gat_multigraph(seq):
	items = np.unique(seq)
	iid2nid = {iid: i for i, iid in enumerate(items)}
	num_nodes = len(items)

	g = dgl.DGLGraph()
	g.add_nodes(num_nodes)
	g.ndata['iid'] = th.Tensor(items)

	seq_nid = [iid2nid[iid] for iid in seq]
	if len(seq) > 1:
		src = seq_nid[:-1]
		dst = seq_nid[1:]
		# edges are added in the order of their occurrences.
		g.add_edges(src, dst)
	g.add_edges(seq_nid,seq_nid)
	return g

def seq_to_position_multigraph(seq):
	items = np.unique(seq)
	iid2nid = {iid: i for i, iid in enumerate(items)}
	num_nodes = len(items)
	#计算出每个nid第一次出现的位置
	nid_pos=np.zeros([num_nodes])
	for i in range(len(seq)):
		position_index=i%512
		iid=seq[i]
		nid=iid2nid[iid]
		if nid_pos[nid]==0:
			nid_pos[nid]=int(position_index)
	position_data=np.zeros([num_nodes,8])
	for k in range(num_nodes):
		position_data[k]=pe[int(nid_pos[k])]
	position_data=th.Tensor(position_data)

	g = dgl.DGLGraph()
	g.add_nodes(num_nodes)
	g.ndata['iid'] = th.Tensor(items)
	g.ndata['pos'] = position_data
	#获取每个节点的位置信息
	#获取每个节点的位置信息到节点的向量里面
	seq_nid = [iid2nid[iid] for iid in seq]
	if len(seq) > 1:
		src = seq_nid[:-1]
		dst = seq_nid[1:]
		# edges are added in the order of their occurrences.
		g.add_edges(src, dst)
	g.add_edges(seq_nid,seq_nid)
	return g

class GNN_Match(nn.Module):
	def __init__(self, num_items,embeds, embedding_dim,drop_out=0.0,gnn_function='GRU',dropout_fc=0.20):
		super().__init__()
		self.num_items=num_items
		self.embedding_l1 = nn.Embedding(num_items[0], embedding_dim, max_norm=1)
		self.embedding_l2 = nn.Embedding(num_items[1], embedding_dim, max_norm=1)
		self.embedding_l3 = nn.Embedding(num_items[2], embedding_dim, max_norm=1)
		self.embedding_l4 = nn.Embedding(num_items[3], embedding_dim, max_norm=1)

		self.embedding_l1.weight.data.copy_(torch.from_numpy(embeds[0]))
		self.embedding_l1.weight.requires_grad = False

		self.embedding_l2.weight.data.copy_(torch.from_numpy(embeds[1]))
		self.embedding_l2.weight.requires_grad = False

		self.embedding_l3.weight.data.copy_(torch.from_numpy(embeds[2]))
		self.embedding_l3.weight.requires_grad = False

		self.embedding_l4.weight.data.copy_(torch.from_numpy(embeds[3]))
		self.embedding_l4.weight.requires_grad = False

		in_dim = embedding_dim
		hidden_dim=embedding_dim

		self.gcn_l1 = GRU_GNN_Vector(in_dim,hidden_dim,drop_out=drop_out,gnn_function=gnn_function,dropout_fc=dropout_fc)
		self.gcn_l2 = GRU_GNN_Vector(in_dim,hidden_dim,drop_out=drop_out,gnn_function=gnn_function,dropout_fc=dropout_fc)
		self.gcn_l3 = GRU_GNN_Vector(in_dim,hidden_dim,drop_out=drop_out,gnn_function=gnn_function,dropout_fc=dropout_fc)
		self.gcn_l4 = GRU_GNN_Vector(in_dim,hidden_dim,drop_out=drop_out,gnn_function=gnn_function,dropout_fc=dropout_fc)

		num_features=embedding_dim
		self.bacth_norm1 = nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.bacth_norm2 = nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.bacth_norm3 = nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.bacth_norm4 = nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.linear_out = nn.Sequential(nn.Linear(embedding_dim * 4, embedding_dim),
										nn.ReLU(),
										nn.Dropout(drop_out),
										nn.Linear(embedding_dim, 16),
										nn.ReLU(),
										nn.Dropout(drop_out),
										nn.Linear(16, 1)
										)
		print('the first layer base gnn aggreator ' + gnn_function)
	def forward(self,x):

		bg1_l, bg2_l, bg3_l, bg4_l, bg1_r, bg2_r, bg3_r, bg4_r=self.forward_seq(x,graph_function='common')
		bg1_l_s, bg2_l_s, bg3_l_s, bg4_l_s, bg1_r_s, bg2_r_s, bg3_r_s, bg4_r_s = self.forward_seq(x, graph_function='shortcut')

		iid = bg1_l.ndata['iid']
		iid=iid.long()
		feat = self.embedding_l1(iid)
		out1_l=self.gcn_l1(feat,bg1_l,bg1_l_s)
		iid = bg1_r.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l1(iid)
		out1_r = self.gcn_l1(feat,bg1_r,bg1_r_s)

		out1=self.bacth_norm1(out1_l)*self.bacth_norm1(out1_r)

		iid = bg2_l.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l2(iid)
		out2_l = self.gcn_l2(feat, bg2_l,bg2_l_s)
		iid = bg2_r.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l2(iid)
		out2_r = self.gcn_l2(feat, bg2_r,bg2_r_s)
		out2 = self.bacth_norm2(out2_l) * self.bacth_norm2(out2_r)

		iid = bg3_l.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l3(iid)
		out3_l = self.gcn_l3(feat, bg3_l,bg3_l_s)
		iid = bg3_r.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l3(iid)
		out3_r = self.gcn_l3(feat, bg3_r,bg3_r_s)
		out3 = self.bacth_norm3(out3_l) * self.bacth_norm3(out3_r)

		iid = bg4_l.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l4(iid)
		out4_l = self.gcn_l4(feat, bg4_l,bg4_l_s)
		iid = bg4_r.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l4(iid)
		out4_r = self.gcn_l4(feat, bg4_r,bg4_r_s)
		out4 = self.bacth_norm4(out4_l) * self.bacth_norm4(out4_r)

		#print(out1.shape,out2.shape,out3.shape,out4.shape)
		out=th.cat([out1,out2,out3,out4],1)
		#print(out.shape)
		out=self.linear_out(out)
		sim = F.sigmoid(out)
		sim = torch.squeeze(sim)
		return sim
	#def write temp
	def forward_temp(self,x):

		bg1_l, bg2_l, bg3_l, bg4_l, bg1_r, bg2_r, bg3_r, bg4_r=self.forward_seq(x,graph_function='common')
		bg1_l_s, bg2_l_s, bg3_l_s, bg4_l_s, bg1_r_s, bg2_r_s, bg3_r_s, bg4_r_s = self.forward_seq(x, graph_function='shortcut')

		iid = bg1_l.ndata['iid']
		iid=iid.long()
		feat = self.embedding_l1(iid)
		out1_l=self.gcn_l1(feat,bg1_l,bg1_l_s)
		iid = bg1_r.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l1(iid)
		out1_r = self.gcn_l1(feat,bg1_r,bg1_r_s)

		out1=self.bacth_norm1(out1_l)*self.bacth_norm1(out1_r)

		iid = bg2_l.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l2(iid)
		out2_l = self.gcn_l2(feat, bg2_l,bg2_l_s)
		iid = bg2_r.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l2(iid)
		out2_r = self.gcn_l2(feat, bg2_r,bg2_r_s)
		out2 = self.bacth_norm2(out2_l) * self.bacth_norm2(out2_r)

		iid = bg3_l.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l3(iid)
		out3_l = self.gcn_l3(feat, bg3_l,bg3_l_s)
		iid = bg3_r.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l3(iid)
		out3_r = self.gcn_l3(feat, bg3_r,bg3_r_s)
		out3 = self.bacth_norm3(out3_l) * self.bacth_norm3(out3_r)

		iid = bg4_l.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l4(iid)
		out4_l = self.gcn_l4(feat, bg4_l,bg4_l_s)
		iid = bg4_r.ndata['iid']
		iid = iid.long()
		feat = self.embedding_l4(iid)
		out4_r = self.gcn_l4(feat, bg4_r,bg4_r_s)
		out4 = self.bacth_norm4(out4_l) * self.bacth_norm4(out4_r)

		#print(out1.shape,out2.shape,out3.shape,out4.shape)
		out=th.cat([out1,out2,out3,out4],1)
		return out

	def forward_seq(self,x,graph_function='common'):

		x_l = [x[i][0] for i in range(len(x))]#x[:, 0]
		x_r = [x[i][1] for i in range(len(x))]

		x1_l = [x_l[i][0] for i in range(len(x_l))]
		x2_l = [x_l[i][1] for i in range(len(x_l))]
		x3_l = [x_l[i][2] for i in range(len(x_l))]
		x4_l = [x_l[i][3] for i in range(len(x_l))]

		x1_le = [x_l[i][4] for i in range(len(x_l))]
		x2_le = [x_l[i][5] for i in range(len(x_l))]
		x3_le = [x_l[i][6] for i in range(len(x_l))]
		x4_le = [x_l[i][7] for i in range(len(x_l))]

		x1_r = [x_r[i][0] for i in range(len(x_r))]
		x2_r = [x_r[i][1] for i in range(len(x_r))]
		x3_r = [x_r[i][2] for i in range(len(x_r))]
		x4_r = [x_r[i][3] for i in range(len(x_r))]

		x1_re = [x_r[i][4] for i in range(len(x_r))]
		x2_re = [x_r[i][5] for i in range(len(x_r))]
		x3_re = [x_r[i][6] for i in range(len(x_r))]
		x4_re = [x_r[i][7] for i in range(len(x_r))]

		#bg = dgl.batch(graphs)
		if graph_function == 'common':
			bg1_l=  [seq_to_gat_multigraph(x1_l[i]) for i in range(len(x1_l))]
			bg2_l = [seq_to_gat_multigraph(x2_l[i]) for i in range(len(x2_l))]
			bg3_l = [seq_to_gat_multigraph(x3_l[i]) for i in range(len(x3_l))]
			bg4_l = [seq_to_gat_multigraph(x4_l[i]) for i in range(len(x4_l))]
		else:
			bg1_l = [seq_to_shortcut_random_graph(x1_l[i],x1_le[i]) for i in range(len(x1_l))]
			bg2_l = [seq_to_shortcut_random_graph(x2_l[i],x2_le[i]) for i in range(len(x2_l))]
			bg3_l = [seq_to_shortcut_random_graph(x3_l[i],x3_le[i]) for i in range(len(x3_l))]
			bg4_l = [seq_to_shortcut_random_graph(x4_l[i],x4_le[i]) for i in range(len(x4_l))]
		bg1_l = dgl.batch(bg1_l)
		bg1_l=trans_to_cuda(bg1_l)
		bg2_l = dgl.batch(bg2_l)
		bg2_l = trans_to_cuda(bg2_l)
		bg3_l = dgl.batch(bg3_l)
		bg3_l = trans_to_cuda(bg3_l)
		bg4_l = dgl.batch(bg4_l)
		bg4_l = trans_to_cuda(bg4_l)

		#
		if graph_function == 'common':
			bg1_r=  [seq_to_gat_multigraph(x1_r[i]) for i in range(len(x1_r))]
			bg2_r = [seq_to_gat_multigraph(x2_r[i]) for i in range(len(x2_r))]
			bg3_r = [seq_to_gat_multigraph(x3_r[i]) for i in range(len(x3_r))]
			bg4_r = [seq_to_gat_multigraph(x4_r[i]) for i in range(len(x4_r))]
		else:
			bg1_r = [seq_to_shortcut_random_graph(x1_r[i], x1_re[i]) for i in range(len(x1_r))]
			bg2_r = [seq_to_shortcut_random_graph(x2_r[i], x2_re[i]) for i in range(len(x2_r))]
			bg3_r = [seq_to_shortcut_random_graph(x3_r[i], x3_re[i]) for i in range(len(x3_r))]
			bg4_r = [seq_to_shortcut_random_graph(x4_r[i], x4_re[i]) for i in range(len(x4_r))]
		bg1_r = dgl.batch(bg1_r)
		bg1_r=trans_to_cuda(bg1_r)
		bg2_r = dgl.batch(bg2_r)
		bg2_r = trans_to_cuda(bg2_r)
		bg3_r = dgl.batch(bg3_r)
		bg3_r = trans_to_cuda(bg3_r)
		bg4_r = dgl.batch(bg4_r)
		bg4_r = trans_to_cuda(bg4_r)

		return bg1_l,bg2_l,bg3_l,bg4_l,bg1_r,bg2_r,bg3_r,bg4_r


