import numpy as np
import networkx as nx

def randomWalk_weight(_g, _corpus_num, _deep_num, _current_word):
	#print('randomWalk_weight')
	_corpus = []
	for i in range(_corpus_num):
		sentence = [_current_word]
		current_word = _current_word
		count = 0
		while count < _deep_num:
			count += 1
			_node_list = []
			_weight_list = []
			for _nbr, _data in _g[current_word].items():
				_node_list.append(_nbr)
				_weight_list.append(_data['weight'])
			_ps = [float(_weight) / sum(_weight_list) for _weight in _weight_list]
			if len(_node_list)==0:
				break
			sel_node = roulette_ps(_node_list,_ps)
			sentence.append(sel_node)
			current_word = sel_node
		_corpus.append(sentence)
	paths = set()
	for line in _corpus:
		path = line[0]
		for i in range(1, len(line)):
			path = path + "," + line[i]
		paths.add(path)
	paths = list(paths)
	return paths

def roulette_ps(_datas, _ps):
	return np.random.choice(_datas, p=_ps)

def generate_seq_weight_paths_arg(seq,deep_num=15,path_num=2):
	#print('generate_seq_weight_paths_arg',deep_num,path_num)
	if len(seq)==1:
		return [str(seq[0])+","+str(seq[0])],[str(seq[0])]
	# 生成有向图网络
	G = nx.DiGraph()
	word_list = []
	edges_weight_dict={}
	for i in range(len(seq)):
		if i>0:
			edge=str(seq[i-1])+','+str(seq[i])
			if edge not in edges_weight_dict:
				edges_weight_dict[edge]=1
			else:
				edges_weight_dict[edge] = 1+edges_weight_dict[edge]
		word_list.append(str(seq[i]))
	for edge in edges_weight_dict:
		edge_lr=edge.split(',')
		G.add_weighted_edges_from([(str(edge_lr[0]), str(edge_lr[1]),edges_weight_dict[edge])])
	word_set = list(set(word_list))
	word_set.sort()
	paths_final=[]
	for word in word_set:
		paths = randomWalk_weight(G, path_num, deep_num, word)
		paths_final=paths_final+paths
	edges=set()
	for path in paths_final:
		path=path.split(',')
		for i in range(1,len(path)):
			edges.add(path[0]+","+path[i])
	edges=list(edges)
	return edges,paths_final


def generate_seq_paths_arg(seq,deep_num=15,path_num=2):
	#print('generate_seq_weight_paths_arg',deep_num,path_num)
	if len(seq)==1:
		return [str(seq[0])+","+str(seq[0])],[str(seq[0])]
	# 生成有向图网络
	G = nx.DiGraph()
	word_list = []
	#edges_weight_dict={}
	for i in range(len(seq)):
		if i>0:
			edge=str(seq[i-1])+','+str(seq[i])
			edge_lr = edge.split(',')
			G.add_weighted_edges_from([(str(edge_lr[0]), str(edge_lr[1]),1)])
		word_list.append(str(seq[i]))
	word_set = list(set(word_list))
	word_set.sort()
	paths_final=[]
	for word in word_set:
		paths = randomWalk_weight(G, path_num, deep_num, word)
		paths_final=paths_final+paths
	edges=set()
	for path in paths_final:
		path=path.split(',')
		for i in range(1,len(path)):
			edges.add(path[0]+","+path[i])
	edges=list(edges)
	return edges,paths_final

if __name__ == '__main__':
	seq = [1,2,3,4,5,6,1,44,12,34]
	edges,paths_final=generate_seq_weight_paths_arg(seq)
	print(paths_final)
	print(edges)