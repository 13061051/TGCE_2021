import numpy as np
import torch
import pickle


batch = 1000


def load_url_2_index_emb(lv_path,dim=16):
	outp2 = 'data/tmp/facts_url.lv' + str(lv_path) +'_'+str(dim)+ '.vector'
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



import json

def read_data_sample(url_index_l1, url_index_l2, url_index_l3, url_index_l4,deep_path=10):
	golden_train = load_golden_set('data/original/golden_valid.csv')
	golden_test = load_golden_set('data/original/golden_valid2.csv')
	# x, y, train_pairs,valid_pairs,test_pairs, xt, xv, yv = file2obj('tmp/xy_d2v_xgb.pkl')
	x,train_y_all,train_pairs_all,test_pairs,xt,xv,yv = file2obj('data/tmp/xy_xgb.pkl')

	#
	print('load random walk generate edges',time_now_str())
	with open("data/original/uids_random_edges/uid_random_edges_n2_d"+str(deep_path)+".txt", 'r',encoding='utf-8') as load_f:
		uid2random_edges = json.load(load_f)
		#uid2random_edges[uid] = user_random_edges

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
	for uid in uids:
		user_seq = []

		facts1 = uid2facts_l1[uid]
		seq = process_facts_2_seqs(facts1, url_index_l1, data_model='dgl')
		user_seq.append(seq)

		facts2 = uid2facts_l2[uid]
		seq = process_facts_2_seqs(facts2, url_index_l2, data_model='dgl')
		user_seq.append(seq)

		facts3 = uid2facts_l3[uid]
		seq = process_facts_2_seqs(facts3, url_index_l3, data_model='dgl')
		user_seq.append(seq)

		facts4 = uid2facts_l4[uid]
		seq = process_facts_2_seqs(facts4, url_index_l4, data_model='dgl')
		user_seq.append(seq)

		#
		random_edges=uid2random_edges[uid]
		for k in range(4):
			edges_=random_edges[k]
			seq_random=[]
			for edge in edges_:
				edge=edge.split(',')
				seq_random.append([int(edge[0]),int(edge[1])])
			seq_random=np.array(seq_random)
			user_seq.append(seq_random)

		uid2seq[uid] = user_seq

	train_y = train_y_all
	train_pairs = train_pairs_all
	train_x_other = x

	test_x_other = xt
	# every pair maps to 2 user
	x_train = []

	for i in range(len(train_pairs)):
		pair = train_pairs[i]
		x_train.append([uid2seq[pair[0]], uid2seq[pair[1]]])


	x_test = []
	test_y = []
	for i in range(len(test_pairs)):
		pair = test_pairs[i]
		x_test.append([uid2seq[pair[0]], uid2seq[pair[1]]])
		if (pair[0], pair[1]) in golden_test or (pair[1], pair[0]) in golden_test:
			test_y.append(1)
		else:
			test_y.append(0)

	# validation set should be choosen by training dataset
	valid_size = int(0.05 * len(train_pairs))
	valid_y = train_y[0:valid_size]
	valid_pairs = train_pairs[0:valid_size]
	valid_x_other = train_x_other[:valid_size]
	x_valid=x_train[0:valid_size]

	#for training
	train_y = train_y[valid_size:]
	train_pairs = train_pairs[valid_size:]
	train_x_other = train_x_other[valid_size:]
	x_train = x_train[valid_size:]

	train_y = np.array(train_y)
	valid_y = np.array(valid_y)

	test_y = np.array(test_y)
	# dataset
	print('read data shape')
	print('train', len(x_train), np.sum(train_y))
	print('valid', len(x_valid), np.sum(valid_y))
	print('test', len(x_test), np.sum(test_y))

	# check my data
	check(golden_train, train_pairs[0:1000], train_y[0:1000])
	check(golden_train, valid_pairs[0:1000], valid_y[0:1000])
	check(golden_test, test_pairs[0:1000], test_y[0:1000])

	return x_train, train_y, x_valid, valid_y, x_test, test_y, train_x_other, valid_x_other, test_x_other


def trans_to_cuda(variable):
	if torch.cuda.is_available():
		device = torch.device('cuda:0')
		variable = variable.to(device)
	else:
		pp = -1
	return variable


import gnn_match_models
import torch.nn as nn
import torch.optim as optim


def evaluate_validation(valid_x, valid_y, model):
	#
	batch_num = int(len(valid_x) / batch)
	if batch_num * batch < len(valid_x):
		batch_num = 1 + batch_num
	preds = []
	for i in range(batch_num):
		input_x = valid_x[i * batch:i * batch + batch]
		#input_x = torch.LongTensor(input_x).to(device)
		#input_x_other = valid_x_other[i * batch:i * batch + batch]
		#input_x_other = trans_to_cuda(torch.Tensor(input_x_other))
		out = model(input_x)
		out = out.detach().cpu().numpy()
		for k in range(len(out)):
			preds.append(out[k])
	# fetch ths
	# it picks recall num
	ths = [t * 0.01 for t in range(100)]
	# print(preds[0:10])
	# print(valid_y[0:10])
	acc = 0
	best_th = 0.5
	best_F1 = 0
	best_p = 0
	best_R = 0
	for th in ths:
		#
		TP = 0
		FN = 0
		FP = 0
		TN = 0
		count0 = 0
		count_l1 = 0
		for i in range(len(preds)):
			if preds[i] > th:
				count_l1 = count_l1 + 1
			# TP: Ture Positive 把正的判断为正的数目 True Positive,判断正确，且判为了正，即正的预测为正的。
			if preds[i] > th and valid_y[i] == 1:
				TP = TP + 1
			# TN: True Negative 把负的判为负的数目 True Negative,判断正确，且判为了负，即把负的判为了负的
			elif preds[i] <= th and valid_y[i] == 0:
				TN = TN + 1
			# FP: False Positive 把负的错判为正的数目 False Positive, 判断错误，且判为了正，即把负的判为了正的
			elif preds[i] > th and valid_y[i] == 0:
				FP = FP + 1
			# FN: False Negative 把正的错判为负的数目 False Negative,判断错误，且判为了负，即把正的判为了负的
			elif preds[i] <= th and valid_y[i] == 1:
				FN = FN + 1
		# 准确率是指有在所有的判断中有多少判断正确的，即把正的判断为正的，还有把负的判断为负的
		Acc = (TP + TN) / (TP + TN + FN + FP)
		# 精确率是相对于预测结果而言的，它表示的是预测为正的样本中有多少是对的
		if TP + FP > 0:
			P = TP / (TP + FP)
		else:
			P = TP / (TP + FP + 1)
		# 召回率是相对于样本而言的，即样本中有多少正样本被预测正确了，这样的有TP个，所有的正样本有两个去向，一个是被判为正的，另一个是错判为负的，因此总共有TP+FN个，所以，召回率 R= TP / (TP+FN)
		if TP+FN==0:
			R=0
		else:
			R = TP / (TP + FN)
		if P + R == 0:
			F1 = 0
		else:
			F1 = (2 * P * R) / (P + R)
		if F1 > best_F1:
			best_p = P
			best_th = th
			best_F1 = F1
			best_R = R
	print(time_now_str() + " Valid Pre: " + str(best_p) + " Rec: " + str(best_R) + " F1: " + str(best_F1)+" best Th: "+str(best_th))
	return best_F1, best_th


def evaluate_test_acc(test_x, test_y, model):
	#
	batch_num = int(len(test_x) / batch)
	if batch_num * batch < len(test_x):
		batch_num = 1 + batch_num
	preds = []
	for i in range(batch_num):
		input_x = test_x[i * batch:i * batch + batch]
		#input_x = torch.LongTensor(input_x).to(device)
		#input_x_other = test_x_other[i * batch:i * batch + batch]
		#input_x_other = trans_to_cuda(torch.Tensor(input_x_other))
		out = model(input_x)
		out = out.detach().cpu().numpy()
		for k in range(len(out)):
			preds.append(out[k])
	# fetch ths
	# print(preds[0:10])
	# print(test_y[0:10])
	acc = 0
	# compute precission recall,F1
	best_th = 0.5
	best_F1 = 0
	best_p = 0
	best_R = 0
	ths = [0.01 * i for i in range(0,100)]
	golden_test = load_golden_set('data/original/golden_valid2.csv')
	gnum=len(golden_test)
	for th in ths:
		TP = 0
		FN = 0
		FP = 0
		TN = 0
		count0 = 0
		count_l1 = 0
		for i in range(len(preds)):
			if preds[i] > th:
				count_l1 = count_l1 + 1
			# TP: Ture Positive 把正的判断为正的数目 True Positive,判断正确，且判为了正，即正的预测为正的。
			if preds[i] > th and test_y[i] == 1:
				TP = TP + 1
			# TN: True Negative 把负的判为负的数目 True Negative,判断正确，且判为了负，即把负的判为了负的
			elif preds[i] <= th and test_y[i] == 0:
				TN = TN + 1
			# FP: False Positive 把负的错判为正的数目 False Positive, 判断错误，且判为了正，即把负的判为了正的
			elif preds[i] > th and test_y[i] == 0:
				FP = FP + 1
			# FN: False Negative 把正的错判为负的数目 False Negative,判断错误，且判为了负，即把正的判为了负的
			elif preds[i] <= th and test_y[i] == 1:
				FN = FN + 1
		# 准确率是指有在所有的判断中有多少判断正确的，即把正的判断为正的，还有把负的判断为负的
		# Acc = (TP + TN) / (TP + TN + FN + FP)
		# 精确率是相对于预测结果而言的，它表示的是预测为正的样本中有多少是对的
		if TP + FP == 0:
			P = 0
		else:
			P = TP / (TP + FP)
		# 召回率是相对于样本而言的，即样本中有多少正样本被预测正确了，这样的有TP个，所有的正样本有两个去向，一个是被判为正的，另一个是错判为负的，因此总共有TP+FN个，所以，召回率 R= TP / (TP+FN)

		# R = TP / (TP + FN)
		R = TP / gnum
		if P + R == 0:
			F1 = 0
		else:
			F1 = (2 * P * R) / (P + R)
		if F1 > best_F1:
			best_p = P
			best_th = th
			best_F1 = F1
			best_R = R
		#if th == 0.5:
		print('th,P,R,F1', th, P, R, F1)
	# print(time_now_str() + " test result with threshold: " + str(th), 'precission: ', P, 'recall: ', R, 'F1: ', F1)
	print(time_now_str() + " Test Pre: " + str(best_p) + " Rec: " + str(best_R) + " F1: " + str(best_F1) + " bth: " + str(best_th))
	print('\n')


def write_train_test_scores(train_x_all, test_x, model,epoch=5):
	batch_num = int(len(train_x_all) / batch)
	if batch_num * batch < len(train_x_all):
		batch_num = 1 + batch_num
	preds_train = []
	for i in range(batch_num):
		input_x = train_x_all[i * batch:i * batch + batch]
		out = model(input_x)
		out = out.detach().cpu().numpy()
		for k in range(len(out)):
			preds_train.append(out[k])
	preds_train = np.array(preds_train)
	np.save('data/results/e'+str(epoch)+'gat_train.npy', preds_train)
	#
	batch_num = int(len(test_x) / batch)
	if batch_num * batch < len(test_x):
		batch_num = 1 + batch_num
	preds = []
	for i in range(batch_num):
		input_x = test_x[i * batch:i * batch + batch]
		#input_x = torch.LongTensor(input_x).to(device)
		#input_x_other = test_x_other[i * batch:i * batch + batch]
		#input_x_other = trans_to_cuda(torch.Tensor(input_x_other))
		out = model(input_x)
		out = out.detach().cpu().numpy()
		for k in range(len(out)):
			preds.append(out[k])
	preds = np.array(preds)
	np.save('data/results/e'+str(epoch)+'gat_test.npy', preds)
	print('wrte seq_gat scores')


import time


def time_now_str():
	out = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
	return out


from torch.optim import lr_scheduler
import argparse
if __name__ == '__main__':

	print(time_now_str() + " is time run !")
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
	parser.add_argument(
		'--batch-size', type=int, default=800, help='the batch size for training'
	)
	parser.add_argument(
		'--epochs', type=int, default=21, help='the number of training epochs'
	)
	parser.add_argument(
		'--weight-decay',
		type=float,
		default=1e-5,
		help='the parameter for L2 regularization',
	)
	parser.add_argument(
		'--dim',
		type=int,
		default=16,
		help='the parameter for dim',
	)
	parser.add_argument(
		'--deep_path',
		type=int,
		default=10,
		help='the deep path length',
	)
	parser.add_argument(
		'--dropout',
		type=float,
		default=0.20,
		help='the drop_out rate',
	)
	parser.add_argument(
		'--gnn_function', type=str, default='GRU_POSGAT', help='the gnn_function'
	)
	parser.add_argument(
		'--model_path', type=str, default='GRU_POSGAT_d16_p10', help='the model_path'
	)
	args = parser.parse_args()
	print(args)
	batch = args.batch_size

	print()
	# read_url_index
	url_index_l1, embed1 = load_url_2_index_emb(1,dim=args.dim)
	url_index_l2, embed2 = load_url_2_index_emb(2,dim=args.dim)
	url_index_l3, embed3 = load_url_2_index_emb(3,dim=args.dim)
	url_index_l4, embed4 = load_url_2_index_emb(4,dim=args.dim)
	#
	if torch.cuda.is_available():
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')
	param = {}
	param['vocab_size1'] = len(embed1)
	param['embed1'] = embed1
	param['vocab_size2'] = len(embed2)
	param['embed2'] = embed2
	param['vocab_size3'] = len(embed3)
	param['embed3'] = embed3
	param['vocab_size4'] = len(embed4)
	param['embed4'] = embed4
	num_items=[len(embed1),len(embed2),len(embed3),len(embed4)]
	embeds=[embed1,embed2,embed3,embed4]
	embedding_dim=args.dim
	user_macth_model = gnn_match_models.GNN_Match(num_items=num_items, embeds=embeds, embedding_dim=embedding_dim, drop_out=args.dropout,gnn_function=args.gnn_function)
	##
	criterion = nn.BCELoss()
	user_macth_model = user_macth_model.to(device)
	optimizer = torch.optim.Adam(user_macth_model.parameters(), lr=0.001,weight_decay=args.weight_decay)

	print(time_now_str() + 'start read dataset')
	x_train, train_y, x_valid, valid_y, x_test, test_y, train_x_other, valid_x_other, test_x_other = read_data_sample(
		url_index_l1, url_index_l2, url_index_l3,url_index_l4,
		deep_path=args.deep_path)
	print(time_now_str(), 'data sample ')
	print(x_train[0][0][0][0:10], x_train[0][1][0][0:10], train_y[0])
	batch_train_num = int(len(x_train) / batch)
	if batch_train_num * batch < len(x_train):
		batch_train_num = 1 + batch_train_num
	print(batch_train_num, " is batch_train_num")
	threshold = 0.5
	best_F1=0
	best_epoch=0
	for epoch in range(args.epochs):
		user_macth_model.train()
		loss_all = 0
		print('\n---------------------------------')
		print(time_now_str() + " epoch " + str(epoch) + " train")
		for i in range(batch_train_num):
			optimizer.zero_grad()
			input_x = x_train[i * batch:i * batch + batch]
			input_y = train_y[i * batch:i * batch + batch]
			input_y = torch.Tensor(input_y).to(device)
			out = user_macth_model(input_x)
			loss = criterion(out, input_y)
			loss.backward()
			optimizer.step()
			loss_all = loss.item() + loss_all
			if i % 100 == 0:
				print(time_now_str()+': '+str(epoch) + ' epoch ' + str(i) + ' index,loss: ', loss.item())
				if i == 100 and epoch % 5 == 0:
					print(input_y.detach().cpu().numpy()[0:10])
					print(out.detach().cpu().numpy()[0:10])
		print(time_now_str()+": "+str(epoch) + ' epoch loss ', loss_all)

		print(time_now_str() + ' start evaluate')
		user_macth_model.eval()
		F1, th = evaluate_validation(valid_x=x_valid, valid_y=valid_y,
		                             model=user_macth_model)
		print('')
		if F1>best_F1:
			best_F1=F1
			torch.save(user_macth_model.state_dict(), 'save_gnn/'+str(args.model_path)+'.pkl')
			best_epoch=epoch
		if epoch-best_epoch>3:
			print('early stop')
			break
	print(time_now_str() + ' start test')
	user_macth_model.load_state_dict(torch.load('save_gnn/'+str(args.model_path)+'.pkl'))
	user_macth_model = user_macth_model.to(device)
	user_macth_model.eval()
	evaluate_test_acc(test_x=x_test, test_y=test_y, model=user_macth_model)
