"""
Bayesian Personalized Ranking

Matrix Factorization model and a variety of classes
implementing different sampling strategies.
"""

import numpy as np
from math import exp
import random
from sampling import UniformPairWithoutReplacement, UniformUserUniformItem


class BPRArgs(object):

    def __init__(self,learning_rate=0.05,
                 bias_regularization=1.0,
                 user_regularization=0.0025,
                 positive_item_regularization=0.0025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors

class BPR(object):

    def __init__(self,D,args):
        """initialise BPR matrix factorization model
        D: number of factors
        """
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors

    def train(self, data, sampler, num_iters):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        """
        self.init(data)

        print 'initial loss = {0}'.format(self.loss())
        for it in xrange(num_iters):
            print 'starting iteration {0}'.format(it)
            for u,i,j in sampler.generate_samples(self.data):
                self.update_factors(u,i,j)
            print 'iteration {0}: loss = {1}'.format(it,self.loss())

    def init(self,data):
        self.data = data
        self.num_users,self.num_items = self.data.shape

        self.item_bias = np.zeros(self.num_items)
        self.user_factors = np.random.random_sample((self.num_users,self.D))
        self.item_factors = np.random.random_sample((self.num_items,self.D))

        self.create_loss_samples()

    def create_loss_samples(self):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100*self.num_users**0.5)

        print 'sampling {0} <user,item i,item j> triples...'.format(num_loss_samples)
        sampler = UniformUserUniformItem(True)
        self.loss_samples = [t for t in sampler.generate_samples(data,num_loss_samples)]

    def update_factors(self,u,i,j,update_u=True,update_i=True):
        """apply SGD update"""
        update_j = self.update_negative_item_factors

        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u,:],self.item_factors[i,:]-self.item_factors[j,:])

        z = 1.0/(1.0+exp(x))

        # update bias terms
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i,:]-self.item_factors[j,:])*z - self.user_regularization*self.user_factors[u,:]
            self.user_factors[u,:] += self.learning_rate*d
        if update_i:
            d = self.user_factors[u,:]*z - self.positive_item_regularization*self.item_factors[i,:]
            self.item_factors[i,:] += self.learning_rate*d
        if update_j:
            d = -self.user_factors[u,:]*z - self.negative_item_regularization*self.item_factors[j,:]
            self.item_factors[j,:] += self.learning_rate*d

    def loss(self):
        ranking_loss = 0;
        for u,i,j in self.loss_samples:
            x = self.predict(u,i) - self.predict(u,j)
            ranking_loss += 1.0/(1.0+exp(x))

        complexity = 0;
        for u,i,j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2

        return ranking_loss + 0.5*complexity

    def predict(self,u,i):
        return self.item_bias[i] + np.dot(self.user_factors[u],self.item_factors[i])



if __name__ == '__main__':

    # learn a matrix factorization with BPR like this:

    import sys
    from scipy.io import mmread
    from scipy import sparse

    data = mmread(sys.argv[1]).tocsr()

    args = BPRArgs()
    args.learning_rate = 0.3

    num_factors = 10
    model = BPR(num_factors,args)

    sample_negative_items_empirically = True
    sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
    num_iters = 10
    model.train(data,sampler,num_iters)
