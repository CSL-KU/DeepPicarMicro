import sys
conv_str = sys.argv[1]
fc_str = sys.argv[2]
width_mult = float(sys.argv[3])
h_len = int(sys.argv[4])
w_len = int(sys.argv[5])
d_len = int(sys.argv[6])

import large_nas as m
model = m.createModel(conv_str, fc_str, width_mult, h_len, w_len, d_len)

model.summary()
