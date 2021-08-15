import json
import gzip
import random

def dictFromFileUnicode(path):
	print("Loading {}".format(path))
	try:
		with gzip.open(path, 'r') as f:
			return json.loads(f.read(),encoding='utf-8')
	except:
		# in case not gzip file
		with open(path, 'r') as f:
			return json.loads(f.read(),encoding='utf-8')


def filter_order_list(pairs, topk, reverse=False):
	'''
	filter topk nearest neighbors
	'''
	print('filter top_{}'.format(topk))
	order = {}
	order_ks = set()
	for p in pairs:
		u = p[0]
		if u in order_ks:
			order[u].append(p)
		else:
			order_ks.add(u)
			order[u] = [p]
	
	res = []
	for u in order_ks:
		order[u] = sorted(order[u], key=lambda x: x[2], reverse=reverse)
		order[u] = order[u][:topk]
		res += order[u]
	return sorted(res, key=lambda x: x[2], reverse=reverse)


def load_golden_set(fname):
	res = []
	with open(fname, 'r') as f:
		for line in f:
			u1, u2 = line.strip().split(',')
			res.append((u1, u2))
	return set(res)
def remove_duplicate(pairs):
	res = []
	pset = set()
	for p in pairs:
		u1,u2 = p[:2]
		if (u1,u2) not in pset and (u2,u1) not in pset:
			pset.add((u1,u2))
			res.append(p)
	return res


def load_sample_pos_neg_pairs(pair_type, k_knn):
	all_pairs = dictFromFileUnicode('data/candidates/candidate_pairs.{}.sfml.lv3.json.gz'.format(pair_type))
	all_pairs = filter_order_list(all_pairs, k_knn)
	pairs = remove_duplicate(all_pairs)
	
	print('Candidate pairs ({}) recall:'.format(pair_type))
	
	golden_res = load_golden_set('data/original/golden_{}.csv'.format(pair_type))
	#pos_pairs = []
	neg_pairs = []
	for pair in pairs:
		u1, u2 = pair[:2]
		if (u1, u2) in golden_res or (u2, u1) in golden_res:
			continue#pos_pairs.append(pair)
		else:
			neg_pairs.append(pair)
	#
	pos_pairs=list(set(golden_res))
	random.shuffle(pos_pairs)
	random.shuffle(neg_pairs)
	limit_pos_samples=len(pos_pairs)
	pos_pairs = [(_[0], _[1]) for _ in pos_pairs[:limit_pos_samples]]
	neg_pairs = [(_[0], _[1]) for _ in neg_pairs[:limit_pos_samples*3]]
	#
	fo=open('data/samples/train_pairs.txt','w',encoding='utf-8')
	train_pairs=[]
	for i in range(limit_pos_samples-10000):
		train_pairs.append([pos_pairs[i][0],pos_pairs[i][1],1])
	for i in range(limit_pos_samples*3-30000):
		train_pairs.append([neg_pairs[i][0],neg_pairs[i][1],0])
	random.shuffle(train_pairs)
	for i in range(len(train_pairs)):
		fo.write(str(train_pairs[i][0])+" "+str(train_pairs[i][1])+" "+str(train_pairs[i][2])+'\n')
	fo.close()
	#valid pairs
	#
	fo = open('data/samples/valid_pairs.txt', 'w', encoding='utf-8')
	valid_pairs = []
	for i in range(limit_pos_samples - 10000,limit_pos_samples):
		valid_pairs.append([pos_pairs[i][0], pos_pairs[i][1], 1])
	for i in range(limit_pos_samples*3 - 30000,limit_pos_samples*3):
		valid_pairs.append([neg_pairs[i][0], neg_pairs[i][1], 0])
	random.shuffle(valid_pairs)
	for i in range(len(valid_pairs)):
		fo.write(str(valid_pairs[i][0]) + " " + str(valid_pairs[i][1]) + " " + str(valid_pairs[i][2]) + '\n')
	fo.close()
	return pos_pairs, neg_pairs
def generate_test_samples():
	pair_type='valid2'
	k_knn=32
	all_pairs = dictFromFileUnicode('data/candidates/candidate_pairs.{}.sfml.lv3.json.gz'.format(pair_type))
	all_pairs = filter_order_list(all_pairs, k_knn)
	pairs = remove_duplicate(all_pairs)
	pairs=[(pair[0],pairs[1]) for pair in pairs]
	print('Candidate pairs ({}) recall:'.format(pair_type))
	golden_res = load_golden_set('data/original/golden_{}.csv'.format(pair_type))
	count=0
	for pair in golden_res:
		u1=pair[0]
		u2=pair[1]
		if (u1, u2) in pairs or (u2, u1) in pairs:
			count=count+1
	print(count,len(golden_res))
import pickle
def file2obj(path):
	print("Loading {}".format(path))
	with open(path, 'rb') as f:
		obj = pickle.load(f,encoding='bytes')
	return obj
import numpy as np
def norm_data(x):
	x=np.array(x)
	int_index = [0, 1, 3, 4, 6, 7, 9, 10, 26, 34]
	for i in range(41):
		if i in int_index:
			for j in range(len(x)):
				x[j,i]=x[j,i]/100
	return x
def watch_train_data():
	x, train_y_all, train_pairs_all, valid_pairs2, test_pairs, xt, xv, yv = file2obj(
		'data/tmp/xy_d2v_xgb.pkl')
	'''
	int_index=[0,1,3,4,6,7,9,10,26,34]
	float_index=[]
	for i in range(41):
		if i in int_index:
			float_index.append(i)
	x_int=[]
	x_float=[]
	for i in range(41):
		if i in int_index:
			x_int.append(x[0][i])
		else:
			x_float.append(x[0][i])
	print(x[0])
	print(x_int)
	print(x_float)
	'''
	x=norm_data(x)
	print(x[0])
if __name__ == '__main__':
	#pos_pairs, neg_pairs=load_sample_pos_neg_pairs(pair_type='valid',k_knn=32)
	#[order1,order2, score]
	#5*3
	'''
	# SF-ML based features
		f+=order_lst.get_order_lst_features(u1,u2)
		
		# Time-related features
		h1 = get_24hr_distribution(u1,u2f)
		h2 = get_24hr_distribution(u2,u2f)
		f.append(pearson_correlation(h1,h2))
		f.append(cosine_distance(h1,h2))
		f.append(pearson_correlation(np.log(h1+1.0),np.log(h2+1.0)))
		f.append(cosine_distance(np.log(h1+1.0),np.log(h2+1.0)))

		h1 = get_24hr_weekly_distribution(u1,u2f)
		h2 = get_24hr_weekly_distribution(u2,u2f)
		f.append(pearson_correlation(h1,h2))
		f.append(cosine_distance(h1,h2))
		f.append(pearson_correlation(np.log(h1+1.0),np.log(h2+1.0)))
		f.append(cosine_distance(np.log(h1+1.0),np.log(h2+1.0)))

		# Probabilistic features
		f+=self.get_prob_features(u1,u2,u2f,p_domains,p_matching)

		# Semantic Embedding features
		f+=self.get_d2v_features(u1,u2,u2v_model)
		[
		101, 101, 3,
		33, 101, 0.8875344172537274,
		3, 1, 1.0400223536535687,
		2, 4, 0.46842502642222994,
		0.720885859109287, 0.21889956359781115, 0.9620063284875287, 0.026517773111055964,
		 0.25019250355972256, 0.7103542235214528, 0.3429797172247191, 0.5912004962580333,
		  #res += [lcpd, np.log(lcpd+1.0), lcu, float(lcpd)/lcu if lcu>0 else -1]
		  0, 0.0, 16, 0.0,
		  #		if len(probs)>0:
			p = np.sum(np.log(probs+1.0))
			r += [p,
				p/len(probs),
				len(probs),
				np.min(probs),
				np.max(probs),
				np.average(probs),
				np.prod(probs)]
		else:
			r += [0.0,
				0.0,
				0,
				0.0,
				0.0,
				0.0,
				0.0]
		if len(probs_pos)>0:
			r.append(np.prod(probs_pos))
		else:
			r.append(-1)
		#
		  0.2411515585704579, 0.015071972410653618, 16, 0.0, 0.03364670446309708, 0.015240794279455808, 0.0,2.663868362273003e-21,
		  0.28281211187812905, 0.020200865134152073, 14, 0.0, 0.1111111111111111, 0.02100871472682721,0.0, 1.1180910716435693e-25,
		  0.890289]
	'''
	#all_int_features
	#all_float_features
	watch_train_data()