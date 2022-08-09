import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import params
from util import networkValues

modelname = params.modelname
m   = __import__(modelname)

# Threshold value used to filter out DNN architectures
max_connections = int(sys.argv[1])

dw_conv_str = ["0001","0010", "0011", "0100", "0101", "0110", "0111", "1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111"]
fc_str = ["000", "001", "010", "011", "100", "101", "110", "111"]
width_mult = [0.2, 0.4, 0.7, 0.8, 0.9, 1.0]
h_len = 68
w_len = [68]
d_len = [1]

total_checked = len(dw_conv_str) * len(fc_str) * len(width_mult) * 1 * len(w_len) * len(d_len)

# Create directory for new dataset if it doesn't already exist
if not os.path.exists(params.dataset):
	os.makedirs(params.dataset)

# Write all successful model combination to a csv file
pass_file = open("{}/pass_models_{}_{}.csv".format(params.dataset, max_connections, width_mult[0]), 'w', newline='')
writer = csv.writer(pass_file)
writer.writerow(["dw_conv_str", "fc_str", "width_mult", "h_len", "w_len", "d_len", "Weights", "Connections"])

num_passes = 0
for c in dw_conv_str:
	for f in fc_str:
		for mult in width_mult:
			for w in w_len:
				for d in d_len:
					model = m.createModel(c, f, mult, h_len, w, d)
					[weights, connections] = networkValues(model)
					if connections <= max_connections:
						num_passes += 1
						print("{}-{}, {}, {}x{}x{} --> {}, {}".format(c, f, mult, h_len, w, d, weights, connections))
						writer.writerow([c, f, mult, h_len, w, d, weights, connections])

print("Total number of models checked: {}".format(total_checked))
print("Number of models that pass: {}".format(num_passes))
pass_file.close()
