import sys
import numpy as np
import math as m
def sens(TP, FN):
	return round(100 * TP/(TP+FN), 1)
def spec(TN, FP):
	return round(100 * TN/(TN+FP), 1)
def acc(TP, TN, FP, FN):
	return round(100 * (TP+TN)/(TP+TN+FP+FN), 1)
def MCC(TP, TN, FP, FN):
	num = (TP * TN - FP * FN)
	denom = m.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
	return round(num/denom, 3)
pref = ['pred_dna', 'pred_rna', 'pred_drna', 'pred_non_drna']
pred_dna = sys.argv[1:5]
pred_rna = sys.argv[5:9]
pred_drna = sys.argv[9:13]
pred_non_drna = sys.argv[13:17]
matr = np.array([pred_dna, pred_rna, pred_drna, pred_non_drna], dtype = float)
d = dict()
for i in range(4):
	d[i] = list()
	TP, TN, FP, FN = 0,0,0,0
	for j in range(4):
		for k in range(4):
			if j == i:
				if k == i:
					TP += matr[j][k]
				else:
					FP += matr[j][k]
			elif k == i:
				if j == i:
					pass
				else:
					FN += matr[j][k]
			else:
				TN += matr[j][k]
	d[i].append([pref[i], TP, TN, FP, FN, sens(TP, FN), spec(TN, FP), acc(TP, TN, FP, FN), MCC(TP, TN, FP, FN)])
MCCs, TPs = list(), list()
for k, v in d.items():
	v = v[0]
	print(f'{v[0]}\n{v[1]}\n{v[2]}\n{v[3]}\n{v[4]}\n{v[5]}%\n{v[6]}%\n{v[7]}%\n{v[8]}')
	MCCs.append(v[8])
	TPs.append(v[1])
	denom = sum(v[1:5])
print(round(sum(MCCs)/len(MCCs), 3))
print(f'{round(100 * sum(TPs)/denom, 1)}%')
