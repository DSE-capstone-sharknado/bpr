from sampling import UniformUserUniformItem, UniformUserUniformItemWithoutReplacement
import amzn_dataset_utils
import model as mdl
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import *

ui_matrix = amzn_dataset_utils.load_to_user_item_matrix("../reviews.csv")
print csr_matrix(ui_matrix).nnz
train_set, test_set = amzn_dataset_utils.test_train_split(ui_matrix)

data = csr_matrix(train_set)

sampler = UniformUserUniformItem(True)
lam=.9
lr=.1
  
from bpr import BPRArgs, BPR

lrs=[]
tests_loss=[]
for lam in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
  print lam
  args = BPRArgs(  bias_regularization=lam,
                   user_regularization=lam,
                   positive_item_regularization=lam,
                   negative_item_regularization=lam,
                   learning_rate = lr)
  K = 20
  model = BPR(K, args)

  num_iters = 10
  losses=model.train(data,sampler,num_iters)
  test_loss = model.auc_w_sampler(test_set)
  print "test acu: %f"%test_loss

  lrs.append(losses[-1])
  tests_loss.append(test_loss)

  
  plt.plot(losses)
  plt.plot(tests_loss)
plt.show()
#
# fn="bpr-breg%.2f-ureg%.2f-lr%.2f-k%d-epochs%d"%(args.bias_regularization, args.user_regularization, args.learning_rate, K, num_iters)
# mdl.save_model(fn,model.item_bias, model.user_factors, model.item_factors)
#
#
# recsys=model
# scores = []
# for u in range(900):
#   for i in range(900):
#     ranking = recsys.predict(u,i)
#     scores.append(ranking)
#
# print np.max(scores)
# print np.mean(scores)
# print np.min(scores)
#
# plt.plot(losses)
# plt.show()
