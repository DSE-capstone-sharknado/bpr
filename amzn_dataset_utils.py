def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)
    
import json
import gzip
import numpy as np
import csv

def load_to_user_item_matrix(path):
  users={}
  n_users=0
  items={}
  n_items=0
  
  #first pass
  #count users and items
  
  reviews=[]
  with open(path, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      user_id=row[0]
      asin=row[1]
      time=row[2]
      
      if user_id not in users:
        users[user_id] = n_users
        n_users+=1
      
      if asin not in items:
        items[asin] = n_items
        n_items+=1
      
      review = [users[user_id], items[asin], time]
      reviews.append(review)
  
  #second pass populate matrix
  uimatrix = np.zeros((n_users, n_items), dtype=bool)
  print uimatrix.shape
  for review in reviews:
    uimatrix[review[0],review[1]]=True
      
  return np.array(uimatrix)
  
  
if __name__ == '__main__':
  #load dataset
  reviews = load_to_user_item_matrix('../reviews.csv')
  print reviews[:199,:]
  