import numpy as np
def save_model(file_name, item_bias, user_factors, item_factors):
  np.savez_compressed(file_name, item_bias=item_bias, user_factors=user_factors, item_factors=item_factors )
  print "saved model params"