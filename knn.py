import pandas as pd
import numpy as np
import gzip
import sf_crime_config as conf
import os.path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
import optparse

#Parse command line info
parser = optparse.OptionParser()

parser.add_option("-o", action="store", type="string", dest="o")
parser.add_option("-s", action="store", type="int", dest="s")
parser.set_defaults(o="out.csv.gz",s=1000)
opts, args = parser.parse_args()

#Set number of neighbors, k
submission_file = opts.o
num_sample = opts.s

#File locations
train_file = conf.train_raw
test_file = conf.test_raw

#load training file to data frame and take a sample for training kNN
print("Loading training data...")
train = pd.read_csv(train_file,header=0)
sample = np.random.choice(len(train), len(train)/10, replace=False)
sample_data = train.iloc[sample]
locations = sample_data[['X','Y']].values
target = sample_data['Category'].values

print("Training...")
knn = KNeighborsClassifier(algorithm='kd_tree')
k = { 'n_neighbors': np.linspace(100,500,5) }
cv = StratifiedShuffleSplit(target)
knn_cv = GridSearchCV(estimator=knn, param_grid=k, scoring='log_loss', cv=cv, n_jobs=-1)
knn_cv.fit(locations,target)

#load test data and make predictions
print('Loading test data...')
test = pd.read_csv(test_file,header=0)
test_locations = test[['X','Y']].values
print('Computing predictions on test data...')
preds = knn_cv.predict_proba(test_locations)

#Start building dataframe for output
print('Organizing output...')
sample_categories = sorted(sample_data['Category'].unique())
out = pd.DataFrame(preds,columns=sample_categories)

#Pad prediction data with empty columns for any crime categories which 
#weren't in the sample
categories = sorted(train['Category'].unique())
if len(categories)!=len(sample_categories):
   category_set = set(categories)
   sample_category_set = set(sample_categories)
   to_add = category_set.difference(sample_category_set)
   for new_cat in to_add:
      out.insert(loc=0, column=new_cat, value=0)

out = out.reindex_axis(sorted(out.columns), axis=1)
out.insert(loc=0,column='Id',value=test['Id'])

print('Saving results to file...')
with gzip.open(submission_file,'wt') as archive:
    out.to_csv(archive,index=False,float_format="%.8f")
