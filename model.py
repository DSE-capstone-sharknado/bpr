import numpy as np
def save_model(item_bias, user_factors, item_factors):
  np.savetxt('item_bias.txt', item_bias)
  np.savetxt('user_factors.txt', user_factors)
  np.savetxt('item_factors.txt', item_factors)
  print "saved model params"