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
parser.add_option("-s", action="store", type="int", dest="s")
parser.set_defaults(o="out.csv.gz",k=5, s=1000)
opts, args = parser.parse_args()

#Set number of neighbors, k
k = opts.k
submission_file = opts.o
num_sample = opts.s

#File locations
train_file = conf.train_raw
test_file = conf.test_raw

#load training file to data frame
train = pd.read_csv(train_file,header=0)

sample = np.random.choice(len(train), num_sample, replace=False)
sample_data = train.iloc[sample]

locations = sample_data[['X','Y']].values
target = sample_data['Category'].values

knn = KNeighborsClassifier(n_neighbors=k,algorithm='kd_tree')
knn.fit(locations, target)

#load test data
test = pd.read_csv(test_file,header=0)
test_locations = test[['X','Y']].values

preds = knn.predict_proba(test_locations)

out = pd.DataFrame(preds,columns=sorted(train['Category'].unique()))
out.insert(loc=0,column='Id',value=test['Id'])

with gzip.open(submission_file,'wt') as archive:
    out.to_csv(archive,index=False,float_format="%.8f")
