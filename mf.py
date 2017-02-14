import csv
import numpy as np

class MF(object):
  """docstring for MF"""
  def __init__(self, K=10, lmbda=10, biasReg=1.0):
    super(MF, self).__init__()
    self.reviews=[]
    self.n_users=0
    self.n_items=0
    self.n_reviews=0
    self.pos_per_user = {}
    self.pos_per_item = {}
    self.test_per_user = {}
    self.val_per_user = {}
    self.items_per_user = {}
    self.items={} # asin -> i
    self.users={} # amzn_id -> u
    self.num_pos_events=0
    self.K=K
    self.lmbda=lmbda
    self.biasReg=biasReg
    self.gamma_user = 0.01 * np.random.randn(self.n_users, K)
    self.gamma_item = 0.01 * np.random.randn(self.n_items, K)
    self.beta_item =  0.01 * np.random.randn(self.n_items)
    
  def _preprocess(self):
    for review in self.reviews:
      user_id=review[0]
      asin=review[1]
      time=review[2]
      
      if asin not in self.items:
        self.items[asin]=self.n_items
        self.n_items+=1
    
      if user_id not in self.users:
        self.users[user_id]=self.n_users
        self.n_users+=1
      
      u = self.users[user_id]
      i = self.items[asin]
      
      if u not in self.pos_per_user:
        self.pos_per_user[u]={}
      if u not in self.test_per_user:
        self.test_per_user[u]=i 
      else:
        self.pos_per_user[u][i]=time
    
      if i not in self.pos_per_item:
        self.pos_per_item[i]={}
      self.pos_per_item[i][u]=time
    
      if u not in self.items_per_user:
        self.items_per_user[u]=[]
      self.items_per_user[u].append(i)
      
      self.n_reviews+=1
      
    #initialize latent factor vectors gamma_u & gamma_i
    #matrix dim: user x K (latent factor dimension [20])
    self.gamma_item = np.random.randn(self.n_items, self.K)
    self.gamma_user = np.random.randn(self.n_users, self.K)
    self.beta_item =  np.random.randn(self.n_items)
      
    print "finished preprocessing..."
      
      
  #HERE is where the gradent is taken - tuple u,i,j are first 3 args
  def _updateFactors(self, user_id, pos_item_id, neg_item_id, learn_rate):
    # print user_id, pos_item_id, neg_item_id, learn_rate
    
    #equation 7
    
    # xuij = beta_i - beta_j + dot(g_U, g_I) - dot(g_U, g_J)
    x_uij = self.beta_item[pos_item_id] - self.beta_item[neg_item_id]
    x_uij += np.dot(self.gamma_user[user_id][:self.K], self.gamma_item[pos_item_id][:self.K]) - np.dot(self.gamma_user[user_id][:self.K], self.gamma_item[neg_item_id][:self.K]);
    
    deri = 1.0 / (1.0 + np.exp(x_uij));
    #eq 8
    self.beta_item[pos_item_id] += learn_rate * (deri - self.biasReg * self.beta_item[pos_item_id])
    self.beta_item[neg_item_id] += learn_rate * (-deri - self.biasReg * self.beta_item[neg_item_id])
    
  	#adjust latent factors
    for f in range(self.K):
  		w_uf = self.gamma_user[user_id][f]
  		h_if = self.gamma_item[pos_item_id][f]
  		h_jf = self.gamma_item[neg_item_id][f]

  		self.gamma_user[user_id][f]     += learn_rate * ( deri * (h_if - h_jf) - self.lmbda * w_uf)
  		self.gamma_item[pos_item_id][f] += learn_rate * ( deri * w_uf - self.lmbda * h_if)
  		self.gamma_item[neg_item_id][f] += learn_rate * (-deri * w_uf - self.lmbda / 10.0 * h_jf)
    
  def _sample_user(self):
    while (True):
      user_id = np.random.randint(1, high=self.n_users)
      if len(self.pos_per_user[user_id]) == 0 or len(self.pos_per_user[user_id]) == self.n_items:
        continue
      
      return user_id 

  
  def _step(self, learn_rate):
    # uniformally sample users in order to approximatelly optimize AUC for all users
    user_id=0
    pos_item_id=0
    neg_item_id=0
    
  	#now it begins! iterate all implicit ratings
    for i in range(self.n_reviews):
      #get random user u
      user_id = self._sample_user()

      #get all the items for u
      user_items = self.items_per_user[user_id]
      
      #get random pos sample for u
      rand_num = np.random.randint(1, high=len(user_items))
      pos_item_id = user_items[rand_num]
    
      #get random neg sample for u
      while True:
        neg_item_id = np.random.randint(1, high=self.n_items)
        if neg_item_id not in user_items:
          break
    
      self._updateFactors(user_id, pos_item_id, neg_item_id, learn_rate)
    
  def fit(self, reviews, iterations=10, learn_rate=.05):
    self.reviews=reviews
    self._preprocess()
    
    for i in range(iterations):
      self._step(learn_rate)
      print i
      
      #calc loss
      #valid, test, std = self.auc();
      #print("[Valid AUC = %f], Test AUC = %f, Test Std = %f\n"%valid, test, std)
      _loss = self.loss()
      print _loss
    
  
  def predict(self, u, i):
    user = u
    item = i

    return self.beta_item[item] + np.dot(self.gamma_user[user][:self.K], self.gamma_item[item][:self.K]);

  def loss(self):
    
    _loss_samples=[]
    for u in xrange(self.n_users):
      a=(u, self.test_per_user[u], np.random.randint(1, high=self.n_items))
      _loss_samples.append(a)

    
    ranking_loss = 0;
    for u,i,j in _loss_samples:
        x = self.predict(u,i) - self.predict(u,j)
        ranking_loss += 1.0/(1.0+np.exp(x))

    # complexity = 0;
    # for u,i,j in self.loss_samples:
    #     complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
    #     complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
    #     complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
    #     complexity += self.bias_regularization * self.item_bias[i]**2
    #     complexity += self.bias_regularization * self.item_bias[j]**2
  
    return ranking_loss
          
  def auc(self):
    # AUC_u_val = []
    AUC_u_test = []

    for u in range(self.n_users):
      item_test = self.test_per_user[u]
      # item_val  = val_per_user[u].first
      x_u_test = self.predict(u, item_test);
      # x_u_val  = self.prediction(u, item_val);

      # count_val = 0;
      count_test = 0;
      max = 0;

      for j in xrange(self.n_items):
        # if (j in self.pos_per_user[u] or item_test == j):# or item_val == j):
        #   continue;

        max+=1
        x_uj = self.predict(u, j)
        if (x_u_test > x_uj):
          count_test+=1

        # if (x_u_val > x_uj):
        #   count_val+=1

    # AUC_u_val[u] = 1.0 * count_val / max
    AUC_u_test[u] = 1.0 * count_test / max
    print "user %d AUC=%f"%u,AUC_u_test[u]

    #sum up AUC
    # AUC_val = 0;
    AUC_test = 0;
    for u in range(n_users):
      # AUC_val += AUC_u_val[u]
      AUC_test += AUC_u_test[u]

    # AUC_val /= self.n_users
    AUC_test /= self.n_users

    # calculate standard deviation
    variance = 0;
    for u in range(self.n_users):
      variance += square(AUC_u_test[u] - AUC_test)

    std = sqrt(variance/self.n_users);

    return AUC_val, AUC_test, std
    
def load_amzn_reviews(path):
  reviews=[]
  with open(path, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      user_id=row[0]
      asin=row[1]
      time=row[2]
      
      review = [user_id, asin, time]
      reviews.append(review)
  return np.array(reviews)
  
if __name__ == '__main__':
  #load dataset
  reviews = load_amzn_reviews('reviews.csv')
  
  # from sklearn.model_selection import train_test_split
  # X_train, X_test, Y_train, Y_test = train_test_split(reviews[:,:2], reviews[:,2], test_size=0.4, random_state=4)
  
  
  recsys = MF()
  recsys.fit(reviews, iterations=10, learn_rate=1)
  print recsys.n_users
  
  # print recsys.gamma_user
  # print recsys.gamma_item
  
  print "done"
  

  