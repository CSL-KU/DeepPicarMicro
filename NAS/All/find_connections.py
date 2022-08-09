import sys
conv_str = sys.argv[1]
fc_str = sys.argv[2]

import large_nas as m
model = m.createModel(conv_str, fc_str)
from connections import calculate_connections

calculate_connections(model)
