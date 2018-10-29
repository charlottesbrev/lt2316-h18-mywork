# Testing file for decision tree induction.
# This file is intended to help you figure out what is expected
# for you to pass the assignment. I don't recommend you change it, but
# rather copy it and make your own tests or write tests afresh.
# We can/will test your code using other data.

import os, sys
import pandas as pd
import numpy as np
from myid3 import DecisionTree # What you will write in the myid3.py file.

def load_data(csv_file):
	data = pd.read_csv(csv_file, header=None)
	data = data.sample(frac=1).reset_index(drop=True) #shuffla shuffla
	if os.path.isfile(csv_file + ".header"):
		header_data = pd.read_csv(csv_file + ".header")
		if len(data.columns) == len(header_data.columns):
			return data, [(header_data.columns[i]) for i in range(len(header_data.columns))]
	return data, [("label " + str(x+1)) for x in range(len(data.columns.count))]

def convert_values(func, v):
	for i in range(len(v)):
		v[i] = func(v[i])
#	return [(func(v[i])) for i in range(len(v))]

# assumes class is last column
def split_class(data, headers):
	ncols = len(data.columns)
	return data[data.columns[:ncols-1]], data[ncols-1], headers[:ncols-1]

def find_index(name, headers):
	for i in range(len(headers)):
		if name == headers[i]:
			return i
	return -1

def do_work(csv_file, cancer_value, save_file):
	data, headers = load_data(csv_file)
	X, y, X_headers = split_class(data, headers)
	y_copy = [(x) for x in y]
	print("converting cancer values")
	convert_values(lambda x: x > cancer_value, y)
	dt = DecisionTree()
	print("creating decision tree")
	dt.train(X, y, X_headers)
	print("saving decision tree")
	with open(save_file, "w") as modelfile:
		dt.save(modelfile)
	modelfile.close()
	print("done")
	return dt, data, headers

def conv(func, name, data, headers):
	i = find_index(name, headers)
	if i < 0:
		return
	convert_values(func, data[i])

def conv2(func, y):
	return [func(x) for x in y]

def to_lmh(x):
	low_limit = 7000
	high_limit = 40000
	if x < low_limit:
		return "Low"
	elif x < high_limit:
		return "Medium"
	else:
		return "High"

def to_lmh2(x):
	low_limit = 40
	high_limit = 100
	if x < low_limit:
		return "Low"
	elif x < high_limit:
		return "Medium"
	else:
		return "High"

def find_rows(func, v):
	rows = []
	for i in range(len(v)):
		if func(v[i]):
			rows.append(i)
	return rows

def find_invalid_rows(data):
	all_rows = []
	for i in range(len(data.columns)):
		idx = find_rows(lambda x:x<0, data[i])
		if len(idx) > 0:
			all_rows = all_rows + idx
	return [x for x in set(all_rows)]


data, headers = load_data("test_data.csv")
data2 = data.drop(find_invalid_rows(data)).reset_index().drop(columns=['index'])

print("converting chemicals")
conv(to_lmh, 'Toxic_Chem', data2, headers)
print("converting lung cancer")
conv(lambda x:x>60, 'Lung_Cancer', data2, headers)
print("converting population density")
conv(to_lmh2, 'Population_Density', data2, headers)


print(data2)
print(len(data2))

X, y, X_headers = split_class(data2, headers)

trainlen = int(len(X)*0.8)
train_X = X[:trainlen]
train_y = y[:trainlen]
test_X = X[trainlen:]
test_y = y[trainlen:]

print("training 80% of the data")
dt = DecisionTree()
dt.train(train_X, train_y, X_headers)
print("Primary split attribute: %s" % (dt.root_node.split_name))
