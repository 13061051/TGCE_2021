import numpy as np

from scipy import stats

def get_p_value(arrA, arrB):

	a = np.array(arrA)

	b = np.array(arrB)

	t, p = stats.ttest_ind(a,b)

	return p

if __name__ == "__main__":

	p=get_p_value([1, 2, 3, 5], [6, 7, 8, 9])
	print(p)