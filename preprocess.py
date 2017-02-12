#this script will convert a reviews file from the website to a simple CSV: user,item,unixtime

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)
    
import json
import gzip
import numpy as np



def source_to_csv(path):
  reviews = []
  for d in parse(path):
    review = [d["reviewerID"], d["asin"], d["unixReviewTime"]]
    reviews.append(review)
  _reviews = np.array(reviews)
  # np.savetxt("reviews.csv", _reviews, delimiter=",", fmt="")
  import pandas as pd 
  df = pd.DataFrame(_reviews)
  df.to_csv("reviews.csv", cols=["0", "1", "2"], index=False, header=False)
  return reviews


path='data/reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
source_to_csv(path)