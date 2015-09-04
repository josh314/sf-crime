import pandas as pd
import numpy as np
import gzip
import sf_crime_config as conf
from sklearn.cluster import MiniBatchKMeans
import os.path

import optparse
parser = optparse.OptionParser()

parser.add_option("-o", action="store", type="string", dest="o")
parser.add_option("-n", action="store", type="int", dest="n")
parser.add_option("-s", action="store", type="int", dest="s")
parser.set_defaults(o="out.csv.gz",n="8",s=100)
opts, args = parser.parse_args()

#Set number of clusters
n_clusters = opts.n
submission_file = opts.o
batch_size = opts.s

#File locations
train_file = conf.train_raw
test_file = conf.test_raw


#load training file to data frame
train = pd.read_csv(train_file,header=0)
locations = train[['X','Y']].values
crime_categories = sorted(train['Category'].unique())
n_categories = len(crime_categories)

#Init & train clusterer
cl = MiniBatchKMeans(n_clusters=n_clusters,max_iter=100,\
                     batch_size=batch_size,verbose=True)
train_clusters = cl.fit_predict(locations)

cluster_crimes = pd.DataFrame(train_clusters,columns=['Cluster'])
cluster_crimes['Category'] = train['Category']

by_category = cluster_crimes.groupby('Category')

#Array for cluster aggregated crime stats
cluster_crime_agg = np.zeros((n_clusters,n_categories))
idx = 0
for _, group in by_category:
    clusters = group['Cluster']
    for cluster in clusters:
        cluster_crime_agg[cluster,idx] += 1
    idx+=1

#Convert cluster crime counts to probabilities by normalizing
for cluster in np.arange(n_clusters):
    total_crimes = cluster_crime_agg[cluster,:].sum()
    cluster_crime_agg[cluster,:] /= total_crimes

#load test data
test = pd.read_csv(test_file,header=0)
test_locations = test[['X','Y']].values

#Obtain cluster each test item is predicted to belong
cluster_preds = cl.predict(test_locations)

#Empty array for building output 
out_array = np.empty((len(test),n_categories))
for test_idx,cluster in enumerate(cluster_preds):
    out_array[test_idx,:] = cluster_crime_agg[cluster,:]

#TODO: make @columns
columns = crime_categories
out = pd.DataFrame(out_array,columns=columns)
out.insert(loc=0,column='Id',value=test['Id'])

with gzip.open(submission_file,'wt') as archive:
    out.to_csv(archive,index=False,float_format="%.8f")

