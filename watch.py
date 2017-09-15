from train import Training, parse_params
import pickle as pkl, sys, cv2
from multiprocessing import Value
import numpy as np

p = pkl.load(open(sys.argv[1]+"/model.pkl", "rb"))
args = pkl.load(open(sys.argv[1]+"/params.pkl", "rb"))
temp_p = parse_params()
for a in args.__dict__:
  setattr(temp_p,a, args.__dict__[a])
args = temp_p
print args
print
args.testing = True
setattr(args, "init_num_moves", 2)
args.fps = 60
t = Training(np.random.RandomState(), 0, p, Value("i", 0, lock=False), args)
