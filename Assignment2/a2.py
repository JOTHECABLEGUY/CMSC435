## Jordan Dube
## CMSC 435 Assignment 2
## Mean and Hot Deck imputation on datasets

import pandas as pd
from os import getcwd
import numpy as np
from time import time
from sys import maxsize
from copy import deepcopy
from math import floor

# rounding function to round to specific number of decimals
#
# @params 		n 			number to be rounded, can be float or int
# 		  		decimals 	number of decimals to round to, can be int or float, default is 0
#
# @returns 					result of rounding
def round_to_n_decimals(n, decimals=0):
    multiplier = 10 ** decimals
    return floor(n*multiplier + 0.5) / multiplier

# naive median imputation algorithm
#
# @params		dataset 	dataset of values, some missing to be imputed
# 
# @returns 					dataset with missing values replaced by median of column
def median_impute(dataset):

	# initialize loop control variables
	num_rows = len(dataset)
	num_cols = len(dataset[0])
	# make a list for each column
	acc_arr = [list()]*num_cols
	# initialize array to store median 
	median_arr = [0]*num_cols

	# fill the column lists with each element that is not missing O(n)
	for row_i in range(num_rows):
		for col_i in range(num_cols):
			element = dataset[row_i][col_i]
			if element != -1.0:
				acc_arr[col_i].append(element)

	# populate the median array with column-specific medians using numpy median O(n)
	for i in range(num_cols):
		median_arr[i] = np.median(acc_arr[i])

	# replace missing values in dataset with corresponding median 
	for row_i in range(num_rows):
		for col_i in range(num_cols):
			element = dataset[row_i][col_i]
			if element == -1.0:
				dataset[row_i][col_i] = median_arr[col_i]

	return dataset


# function to impute missing values in a dataset using the mean of each column
#
# @params  		dataframe 	dataset with missing values to be imputed, will be numpy 2d array
#
# @returns 					imputed dataset, meaning missing values have been replaced with column-specific mean
def mean_impute(dataframe):

	# length of a row
	row_length = len(dataframe[0])
	# array to store column-specific sums
	acc_arr = [0]*row_length
	# array to store column-specific counts
	n_arr = [0]*row_length

	# loop through dataset
	for row_i in range(len(dataframe)):
		for col_i in range(row_length):
			element = dataframe[row_i][col_i]
			if element != -1.0:
				# add value of element to corresponding value in accumulation array based on column
				acc_arr[col_i] += element
				# increment corresponding value in counting array based on column
				n_arr[col_i] += 1

	# array to hold column-specific mean values
	mean_arr = [0]*row_length

	# populate mean array with each column's sum divided by its number of non-missing values, rounded to 5 places
	for i in range(row_length):
		mean_arr[i] = round_to_n_decimals(acc_arr[i]/n_arr[i], 5)

	# loop through dataset and replacing missing values with the corresponding column's mean
	for row_i in range(len(dataframe)):
		for col_i in range(row_length):
			element = dataframe[row_i][col_i]
			if element == -1.0:
				dataframe[row_i][col_i] = mean_arr[col_i]

	return dataframe

# utility function to calculate the Manhattan distance between 2 rows 
#
# @params 		row1 	numpy array of floats that designates one of the rows to be used
#		  		row2	numpy array of floats that designates one of the rows to be used
#
# @returns 				Manhattan distance between the 2 rows
def returnDistBetween(row1, row2):

	# initialize distance
	dist = 0.0

	# loop through the 2 rows, element by element. If one of the elements is missing, add 1, 
	#	if not, add the positive difference between the elements
	for i, j in zip(row1, row2):
		if j == -1.0 or i == -1.0:
			dist += 1.0
		else:
			dist += abs(i-j)

	return dist

# utility function to return the index of the row in the dataset that is closest 
#	to the given row index with respect to every other row
#
# @params 		row_i 				index of starting row (stays constant for each iteration of loop)
#		  		dataframe 			dataset of values (some missing) to use in distance calculations
# 		  		excluded  			list of indices of rows that should not be used in calculations
#										this is composed of the starting row (distance from a row to itself is 0)
#										and rows that do not contain the desired element (only used in further iterations)
#										if a row is closest, but the corresponding element is missing, this function will be 
#										called again, but with the previously returned index appended to the excluded list to
#										get the second closest list and so on.
#		  		row_row_dist_dict	dictionary of stored Manhattan distances between rows so that distances only need to be found once
# 
# @returns 							index of row within dataset that is closest to the starting row (at index row_i)
def get_neighbor_index(row_i, dataframe, excluded, row_row_dist_dict):

	# initialize minimum distance and neighbor index so the first check will update these
	min_dist = maxsize
	neighbor_index = 0

	# loop through rows of dataset
	for row_j in range(len(dataframe)):
		# checks if the row should be processed
		if row_j not in excluded:
			# checks if the Manhattan distance between the starting row (row_i) and current row has been calculated
			#	if yes, pulls distance from dictionary, if no, calculate the Manhattan distance between the 2 rows and
			#	adds the distance into the dictionary
			if f'{row_i}{row_j}' in row_row_dist_dict.keys():
				distance = row_row_dist_dict[f'{row_i}{row_j}']
			else:
				distance = returnDistBetween(dataframe[row_i], dataframe[row_j])
				row_row_dist_dict[f'{row_i}{row_j}'] = distance
				row_row_dist_dict[f'{row_j}{row_i}'] = distance

			# checks if distance above is lower than the previous minimum. 
			#	If yes, update the minimum distance and index of neighbor	
			if distance < min_dist:
				min_dist = distance
				neighbor_index = row_j

	return neighbor_index

# function to impute missing values in a dataset using each element's closest neighbor (with respect to Manhattan distance)
#
# @params 		dataframe 	dataset with missing values to be imputed, will be numpy 2d array
#
# @returns 					imputed dataset, meaning missing values have been replaced with complete value from nearest row
def hotdeck_impute(dataframe):

	# dictionary to store information about which elements should be imputed and the values with which they will be replaced
	imputed_elements = dict()
	# dictionary of stored Manhattan distances between rows so that distances only need to be found once
	row_row_dist_dict = dict()

	# loop through rows of the dataset
	for row_i in range(len(dataframe)):
		if -1.0 in dataframe[row_i]:
			# if there is a missing value in the row, make a list in the dictionary
			imputed_elements[row_i] = list()
			# add elements of the form [column index, -1] for each missing value in the row (the -1 gets replaced later)
			for col_i in range(len(dataframe[0])):
				if dataframe[row_i][col_i] == -1.0:
					imputed_elements[row_i].append([col_i, -1.0])

			# loop through each element and get the new value to replace the -1 in each element
			for element in imputed_elements[row_i]:
				# initialize an excluded list to block useless rows
				excluded = list()
				# distance from row to itself is 0, also it doesnt have any values that 
				#	can be used to replace missing values within itself, so exclude it
				excluded.append(row_i)
				# get the row index of the row "closest" to the current row based on Manhattan distance
				element_neighbor_index = get_neighbor_index(row_i, dataframe, excluded, row_row_dist_dict)

				# while loop to account for repeated occurrences of a row not having the replacement value for 
				#	the current element
				while dataframe[element_neighbor_index][element[0]] == -1.0:
					# each time a new neighbor fails to provide a complete feature, add that row index to excluded 
					#	so it doesnt get rechecked, and then get the next closest neighbor that is not excluded
					excluded.append(element_neighbor_index)
					element_neighbor_index = get_neighbor_index(row_i, dataframe, excluded, row_row_dist_dict)

				# once a finalized neighbor has been found, replace the -1 in the element with the new value taken from
				# 	the closest row not missing a value in the current element's position. current form is now
				#	[column index, new value]
				element[1] = dataframe[element_neighbor_index][element[0]]

	# iterate through all the imputed elements and actually update them in the dataset. This was done to avoid
	#	already imputed elements impacting distance calculations
	for row, elements in imputed_elements.items():
		for element in elements:
			dataframe[row][element[0]] = element[1]

	return dataframe

# utility function to name files and write to disk
#
# @params 		csv_name	name denoting which csv was used
#		  		dataset 	imputed dataset to be written to a CSV file
#		  		mode		number specifying whether the imputation was done through mean (0) or hot-deck (1)
#		  		cols 		list of column names to be used in the CSV
#		  		curr_dir	current directory (will be location of the saved CSV)
#
# @returns 					None
def nameFile(csv_name, dataset, mode, cols, curr_dir):

	# convert the dataset into pandas DataFrame using the list of columns and imputed dataset
	dataframe = pd.DataFrame(dataset, columns = cols)
	# translate mode into corresponding string
	m = "hd" if mode == 1 else "mean"
	# format the new file's name
	export_filename = f'{curr_dir}/V00946473_{csv_name}_imputed_{m}.csv'

	# export the dataframe to CSV with the formatted name
	dataframe.to_csv(export_filename, index = False)

# method find the mean absolute error of an imputed dataset
#
# @params 		orig_dataset 		original dataset with missing values, used for finding indices of missing values
#		  		imputed_dataset 	dataset containing imputed values instead of missing values
#		  		full_set 			dataset without missing values to check imputed values against
#
# @returns 							float representing the average error per imputed value in the imputed dataset
def find_error(orig_dataset, imputed_dataset, full_set):

	# initialize error list to hold positive differences between elements corresponding to missing elements from orig_dataset
	indiv_errors = list()

	# loop through each element in the original dataset, if a missing value is found, find the difference and add to error list
	for i in range(len(orig_dataset)):
		for j in range(len(orig_dataset[0])):
			element = orig_dataset[i][j]
			if element == -1.0:
				err = abs(imputed_dataset[i][j]-full_set[i][j])
				indiv_errors.append(err)

	# once differences are found, add together and divide by number of missing elements to get average error per imputation
	mean_err = round_to_n_decimals(sum(indiv_errors)/len(indiv_errors), 4)
	return mean_err

# main driver block
if __name__ == '__main__':

	# store current directory to be used for disk read/write
	curr_dir = getcwd()

	# read the complete set, store the column names for disk write, and convert all values to float
	full_set = pd.read_csv(curr_dir + "/dataset_complete.csv")
	cols = full_set.columns
	full_set_arr = full_set.to_numpy(dtype = float)

	# read the 1% and 10% missing datasets, replace all missing values with -1, and convert everything to float
	orig_slight_empty_arr = (pd.read_csv(curr_dir + "/dataset_missing01.csv")).replace(to_replace = "?", value = "-1.0").to_numpy(dtype = float)
	orig_more_empty_arr = (pd.read_csv(curr_dir + "/dataset_missing10.csv")).replace(to_replace = "?", value = "-1.0").to_numpy(dtype = float)

	# run mean imputation on 1% missing dataset, time it, convert time to int milliseconds, find error of imputation,
	# 	write to file
	Runtime_01_mean_start = time()
	mean_impute01 = mean_impute(deepcopy(orig_slight_empty_arr))
	Runtime_01_mean = int((time()-Runtime_01_mean_start)*1000)
	MAE_01_mean = find_error(orig_slight_empty_arr, mean_impute01, full_set_arr)
	nameFile("missing01", mean_impute01, 0, cols, curr_dir)

	# run mean imputation on 10% missing dataset, time it, convert time to int milliseconds, find error of imputation,
	# 	write to file
	Runtime_10_mean_start = time()
	mean_impute10 = mean_impute(deepcopy(orig_more_empty_arr))
	Runtime_10_mean = int((time()-Runtime_10_mean_start)*1000)
	MAE_10_mean = find_error(orig_more_empty_arr, mean_impute10, full_set_arr)
	nameFile("missing10", mean_impute10, 0, cols, curr_dir)

	# run hot deck imputation on 1% missing dataset, time it, convert time to int milliseconds, find error of imputation,
	# 	write to file
	Runtime_01_hd_start = time()
	hd_impute01 = hotdeck_impute(deepcopy(orig_slight_empty_arr))
	Runtime_01_hd = int((time()-Runtime_01_hd_start)*1000)
	MAE_01_hd = find_error(orig_slight_empty_arr, hd_impute01, full_set_arr)
	nameFile("missing01", hd_impute01, 1, cols, curr_dir)

	# run hot deck imputation on 1% missing dataset, time it, convert time to int milliseconds, find error of imputation,
	# 	write to file
	Runtime_10_hd_start = time()
	hd_impute10 = hotdeck_impute(deepcopy(orig_more_empty_arr))
	Runtime_10_hd = int((time()-Runtime_10_hd_start)*1000)
	MAE_10_hd = find_error(orig_more_empty_arr, hd_impute10, full_set_arr)
	nameFile("missing10", hd_impute10, 1, cols, curr_dir)

	# print timing and error metrics as specified in assignment specification
	print(f'MAE_01_mean = {MAE_01_mean}\nRuntime_01_mean = {Runtime_01_mean}') 
	print(f'MAE_01_hd = {MAE_01_hd}\nRuntime_01_hd = {Runtime_01_hd}') 
	print(f'MAE_10_mean = {MAE_10_mean}\nRuntime_10_mean = {Runtime_10_mean}')
	print(f'MAE_10_hd = {MAE_10_hd}\nRuntime_10_hd = {Runtime_10_hd}\n')
