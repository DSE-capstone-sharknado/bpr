from sampling import UniformUserUniformItem, UniformUserUniformItemWithoutReplacement
import amzn_dataset_utils

ui_matrix = amzn_dataset_utils.load_to_user_item_matrix("reviews.csv")

from scipy.sparse import *
data = csr_matrix(ui_matrix)
print data.nnz

sampler = UniformUserUniformItem(True)

  
from bpr import BPRArgs, BPR
args = BPRArgs(  bias_regularization=1.0,
                 user_regularization=1.0,
                 positive_item_regularization=1,
                 negative_item_regularization=.2,learning_rate = 1)

K = 10
model = BPR(K, args)

num_iters = 10
model.train(data,sampler,num_iters)


# .05,10=8000
#  .1,10= overflow
# .09,10= 7802.62088005 5th iteration math range error (l: 86)