import sys
import pandas as pd
with open('../sequences_training.txt', 'r') as seq_file:
	classes = [c[-1] for c in [line.split(',') for line in seq_file.read().splitlines()]]

class1 = 'DNA'
class2 = 'RNA'
class3 = 'DRNA'
class4 = 'nonDRNA'

c1, c2, c3, c4 = 0,0,0,0

for c in classes:
	if c == class1:
		c1 += 1
	elif c == class2:
		c2 += 1
	elif c == class3:
		c3 += 1
	else:
		c4 += 1
# df_1 = pd.read_csv(sys.argv[1])
# # df_1.drop(df_1.index[8794:], inplace=True)
# print(len(df_1))
# df_2 = pd.read_csv(sys.argv[2])
# # df_2.drop(df_2.index[8794:], inplace=True)
# print(len(df_2))
# df_3 = pd.read_csv(sys.argv[3])
# # df_3.drop(df_3.index[8794:], inplace=True)
# print(len(df_3))
# df_4 = pd.read_csv(sys.argv[4])
# # df_4.drop(df_4.index[8795:], inplace=True)
# print(len(df_4))
# df_5 = pd.read_csv(sys.argv[5])
# # df_5.drop(df_5.index[8795:], inplace=True)
# print(len(df_5))
# df = pd.concat([df_1, df_2, df_3, df_4, df_5], axis = 1)
# del df['ID']
# # df['Class'] = classes[:-1]
# df.to_csv('test_out.csv', index = False)