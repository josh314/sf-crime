import pandas as pd
import numpy as np
import gzip
import sf_crime_config as conf
import os.path
from sklearn.neighbors import KNeighborsClassifier

import optparse
parser = optparse.OptionParser()

parser.add_option("-o", action="store", type="string", dest="o")
parser.add_option("-k", action="store", type="int", dest="k")
parser.set_defaults(o="out.csv.gz",k="5")
opts, args = parser.parse_args()

#Set number of neighbors, k
k = opts.k
submission_file = opts.o

#File locations
train_file = conf.train_raw
test_file = conf.test_raw


#load training file to data frame
train = pd.read_csv(train_file,header=0)
locations = train[['X','Y']].values
categories = train['Category'].values

clf = KNeighborsClassifier(n_neighbors=k,algorithm='kd_tree')
clf.fit(locations, categories)
