import argparse
import os
from algorithms import Finetuning, Reptile

parser=argparse.ArgumentParser()
parser.add_argument('--folder', default="./data/", help="Data folder to store the dataset") # do not change this line!
args = parser.parse_args()
if not os.path.isdir(args.folder):
    os.mkdir(args.folder)

iterations = 30000 
samples = 100        # The amount of samples the algorithm is trained on at each iteration
batch_size = 10      # The amount of subsamples that samples are divided into for minibatch training
support_size = 20    # The amount of suport samples used during transfer
query_size = 100     # The amount of query samples used during transfer

Finetuning(iterations, samples, batch_size, support_size, query_size, step=0.01).run(eval=100)
# Reptile(iterations, samples, batch_size, support_size, query_size, outerstep=0.1, innerstep=0.01, innerepochs=3).run(eval=100)