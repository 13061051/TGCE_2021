import numpy as np
import torch
import pickle


batch = 1000


def load_url_2_index_emb(lv_path):
	outp2 = 'data/tmp/facts_url.lv' + str(lv_path) + '.vector'
	fr = open(outp2, 'r')
	embed = []
	url2index = {}
	lines = fr.readlines()
	info = lines[0].replace('\n', '').split(' ')
	dim = int(info[1])
	print('lv_path', lv_path, 'embed', info)
	# 0是用来填充使用的
	embed = np.zeros([len(lines) - 1, int(info[1])])
	for i in range(1, len(lines)):
		info = lines[i].replace('\n', '').split(' ')
		for k in range(dim):
			embed[i - 1][k] = float(info[k + 1])
		url = info[0]
		url2index[url] = i - 1
	# url_index_l1, embed1
	return url2index, embed


def process_facts_2_seqs(facts, url2index, data_model='cnn'):
	v = []
	num = len(facts)
	for i in range(num):
		item= url2index[facts[i]]
		v.append(item)
	return v



def load_user_facts(lv_path):
	uid2facts = {}
	# valid for train
	# valid2 for test
	fr = open('data/tmp/facts_url_valid.lv' + str(lv_path) + '.sectional.txt', 'r')
	lines = fr.readlines()
	# 6b72ef48acd5825962ca7f0756ce3f9f	t291004 t4786 t5077 t882438 t2398 t4786 t22
	for line in lines:
		line = line.replace('\n', '')
		info = line.split('\t')
		uid = info[0]
		facts = info[1].split(' ')
		uid2facts[uid] = facts
	# valid2 for test
	fr = open('data/tmp/facts_url_valid2.lv' + str(lv_path) + '.sectional.txt', 'r')
	lines = fr.readlines()
	# 6b72ef48acd5825962ca7f0756ce3f9f	t291004 t4786 t5077 t882438 t2398 t4786 t22
	for line in lines:
		line = line.replace('\n', '')
		info = line.split('\t')
		uid = info[0]
		facts = info[1].split(' ')
		uid2facts[uid] = facts
	return uid2facts


def file2obj(path):
	print("Loading {}".format(path))
	with open(path, 'rb') as f:
		obj = pickle.load(f, encoding='bytes')
	return obj


def load_golden_set(fname):
	res = []
	with open(fname, 'r') as f:
		for line in f:
			u1, u2 = line.strip().split(',')
			res.append((u1, u2))
	return set(res)


def check(golden_res, pairs, y):
	flag = 0
	pairs_num = len(pairs)
	for i in range(pairs_num):
		pair = pairs[i]
		if (pair[0], pair[1]) in golden_res or (pair[1], pair[0]) in golden_res:
			if y[i] == 1:
				flag = 1 + flag
		else:
			if y[i] == 0:
				flag = 1 + flag
	print('check result')
	print(len(y), flag, np.sum(y))


def norm_data(x_in):
	x_in = np.array(x_in)
	int_index = [0, 1, 3, 4, 6, 7, 9, 10, 26, 34]
	for i in range(41):
		if i in int_index:
			for j in range(len(x_in)):
				x_in[j, i] = x_in[j, i] / 100
	return x_in

from generate_graph import generate_seq_weight_paths_arg
from generate_graph import generate_seq_paths_arg

import json
def generate_enhanced_weighted_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4,deep_path=10):
	# x里面存着以后要用到其他特征
	# read user facts
	uid2facts_l1 = load_user_facts(1)
	uid2facts_l2 = load_user_facts(2)
	uid2facts_l3 = load_user_facts(3)
	uid2facts_l4 = load_user_facts(4)
	# read url index and embedding
	# every user is a three lv_path seqs
	uid2seq = {}
	uids = list(uid2facts_l1.keys())
	uids.sort()
	uid2random_edges={}
	print(time_now_str(),len(uids),"is total uids size")
	for index in range(len(uids)):
		if index%100==0 and index>0:
			print(time_now_str(),'process index',index)
		uid=uids[index]
		user_seq = []
		user_random_edges=[]

		facts1 = uid2facts_l1[uid]
		seq = process_facts_2_seqs(facts1, url_index_l1, data_model='dgl')
		random_edges, paths_final = generate_seq_weight_paths_arg(seq,deep_num=deep_path)
		user_seq.append(seq)
		user_random_edges.append(random_edges)

		facts2 = uid2facts_l2[uid]
		seq = process_facts_2_seqs(facts2, url_index_l2, data_model='dgl')
		random_edges, paths_final = generate_seq_weight_paths_arg(seq,deep_num=deep_path)
		user_seq.append(seq)
		user_random_edges.append(random_edges)

		facts3 = uid2facts_l3[uid]
		seq = process_facts_2_seqs(facts3, url_index_l3, data_model='dgl')
		random_edges, paths_final = generate_seq_weight_paths_arg(seq,deep_num=deep_path)
		user_seq.append(seq)
		user_random_edges.append(random_edges)

		facts4 = uid2facts_l4[uid]
		seq = process_facts_2_seqs(facts4, url_index_l4, data_model='dgl')
		random_edges, paths_final = generate_seq_weight_paths_arg(seq,deep_num=deep_path)
		user_seq.append(seq)
		user_random_edges.append(random_edges)

		uid2seq[uid] = user_seq
		uid2random_edges[uid]=user_random_edges
	with open("data/original/uids_random_edges/uid_random_edges_n2_d"+str(deep_path)+"_weight.txt", "w") as f:
		json.dump(uid2random_edges, f,indent=4,ensure_ascii=False)
	print("加载入文件完成...")
	print(time_now_str())

def generate_enhanced_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4,deep_path=10):
	# x里面存着以后要用到其他特征
	# read user facts
	uid2facts_l1 = load_user_facts(1)
	uid2facts_l2 = load_user_facts(2)
	uid2facts_l3 = load_user_facts(3)
	uid2facts_l4 = load_user_facts(4)
	# read url index and embedding
	# every user is a three lv_path seqs
	uid2seq = {}
	uids = list(uid2facts_l1.keys())
	uids.sort()
	uid2random_edges={}
	print(time_now_str(),len(uids),"is total uids size")
	for index in range(len(uids)):
		if index%100==0 and index>0:
			print(time_now_str(),'process index',index)
		uid=uids[index]
		user_seq = []
		user_random_edges=[]

		facts1 = uid2facts_l1[uid]
		seq = process_facts_2_seqs(facts1, url_index_l1, data_model='dgl')
		random_edges, paths_final = generate_seq_paths_arg(seq,deep_num=deep_path)
		user_seq.append(seq)
		user_random_edges.append(random_edges)

		facts2 = uid2facts_l2[uid]
		seq = process_facts_2_seqs(facts2, url_index_l2, data_model='dgl')
		random_edges, paths_final = generate_seq_paths_arg(seq,deep_num=deep_path)
		user_seq.append(seq)
		user_random_edges.append(random_edges)

		facts3 = uid2facts_l3[uid]
		seq = process_facts_2_seqs(facts3, url_index_l3, data_model='dgl')
		random_edges, paths_final = generate_seq_paths_arg(seq,deep_num=deep_path)
		user_seq.append(seq)
		user_random_edges.append(random_edges)

		facts4 = uid2facts_l4[uid]
		seq = process_facts_2_seqs(facts4, url_index_l4, data_model='dgl')
		random_edges, paths_final = generate_seq_paths_arg(seq,deep_num=deep_path)
		user_seq.append(seq)
		user_random_edges.append(random_edges)

		uid2seq[uid] = user_seq
		uid2random_edges[uid]=user_random_edges
	with open("data/original/uids_random_edges/uid_random_edges_n2_d"+str(deep_path)+".txt", "w") as f:
		json.dump(uid2random_edges, f,indent=4,ensure_ascii=False)
	print("加载入文件完成...")
	print(time_now_str())




import time


def time_now_str():
	out = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
	return out


from torch.optim import lr_scheduler
import argparse
if __name__ == '__main__':
	print()
	# read_url_index
	url_index_l1, embed1 = load_url_2_index_emb(1)
	url_index_l2, embed2 = load_url_2_index_emb(2)
	url_index_l3, embed3 = load_url_2_index_emb(3)
	url_index_l4, embed4 = load_url_2_index_emb(4)
	print(time_now_str()+"start to process")
	#generate_enhanced_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4, deep_path=0)
	generate_enhanced_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4, deep_path=5)
	generate_enhanced_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4, deep_path=10)
	'''
	generate_enhanced_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4,deep_path=5)
	generate_enhanced_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4, deep_path=10)
	generate_enhanced_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4, deep_path=15)
	generate_enhanced_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4, deep_path=20)
	generate_enhanced_weighted_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4, deep_path=5)
	generate_enhanced_weighted_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4, deep_path=10)
	generate_enhanced_weighted_graph(url_index_l1, url_index_l2, url_index_l3, url_index_l4, deep_path=15)
	'''