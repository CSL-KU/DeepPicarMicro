modelname = "pilotnet_nas"
inputres = (66,200)
inputchannels = 3
dataset="deeppicar"
totalframes=30000
batch_size=128
epochs=15
val_loss=0
val_high=0

if dataset == "deeppicar":
    val_loss = 0.0350
    val_high = 0.0450

if dataset == "udacity":
    val_loss = 0.0440
    val_high = 0.0500