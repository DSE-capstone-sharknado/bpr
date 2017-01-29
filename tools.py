# we need to get a utility-matirx/user-item matrix from the input json file

import re
import os
import sys
import gzip


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

def parseLst(lst):
  res = ""
  for e in lst:
    res += e.strip('\'') + ' '
  return res.strip(' ')

# if len(sys.argv) != 3:
#   print "Parameters: "
#   print "1. INPUT: Amazon review json file (.json.gz)"
#   print "2. OUTPUT: Amazon simple format file (.gz)"
#   sys.exit()
  
users={}
users_r={}
items={}
items_r={}
n_users=0
n_items=0

vote_map={}

count = 0
for dict in parse("../data/reviews_Clothing_Shoes_and_Jewelry_5.json.gz"):
  count += 1
  if count % 10000 == 0:
    print count
    break

  reviewer_id = dict['reviewerID']
  asin = dict['asin']
  rating = str(dict['overall'])
  time = str(dict['unixReviewTime'])
  
  if asin not in items: #new item
    items_r[n_items] = asin
    items[asin] = n_items
    n_items+=1
  
  
  if reviewer_id not in users: #new user
    users_r[n_users] = reviewer_id
    users[reviewer_id] = n_users
    n_users+=1
    
  vote_map[users[reviewer_id], items[asin]] = time
    
print "***"
print n_users #39,387
print n_items #23,033
print count #278,677 votes

votes=[]
for k,v in vote_map.iteritems():
  vote = {'user': k[0], 'item': k[1], 'vote_time': v, 'label': 1}
  votes.append(vote)
  
print len(votes)


#how can I convert this collection of x,y's into a matrix
import numpy as np
from scipy.sparse import csr_matrix
utility_matrix = csr_matrix((n_users, n_items))
for k,v in vote_map.iteritems():  
  utility_matrix[k[0],k[1]] = 1
  
print utility_matrix.shape

from scipy.io import mmwrite

mmwrite("amzn_util.mtx", utility_matrix)

m = csr_matrix([[0,0,0],[1,0,0],[0,1,0]])
mmwrite("test.mtx", m)