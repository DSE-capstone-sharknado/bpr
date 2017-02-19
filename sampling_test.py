from sampling import UniformUserUniformItem, UniformUserUniformItemWithoutReplacement
import amzn_dataset_utils
import model as mdl

ui_matrix = amzn_dataset_utils.load_to_user_item_matrix("../reviews.csv")

from scipy.sparse import *
data = csr_matrix(ui_matrix)
print data.nnz

sampler = UniformUserUniformItem(True)

  
from bpr import BPRArgs, BPR
args = BPRArgs(  bias_regularization=0.5,
                 user_regularization=0.5,
                 positive_item_regularization=0.5,
                 negative_item_regularization=0.5,learning_rate = 0.1)

K = 10
model = BPR(K, args)

num_iters = 10
model.train(data,sampler,num_iters)

mdl.save_model(model.item_bias, model.user_factors, model.item_factors)
# .05,10=8000
#  .1,10= overflow
# .09,10= 7802.62088005 5th iteration math range error (l: 86)