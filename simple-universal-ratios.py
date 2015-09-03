import pandas as pd
import numpy as np
import gzip

#Hardcoded file locations
train_file = 'data/raw/train.csv'
test_file = 'data/raw/test.csv'
submission_file = 'simple-universal-ratios-submission.csv.gz'

#load training file to data frame
train = pd.read_csv(train_file,header=0)

#Aggregate total number of each type of crime in training set
crime_numbers = train.groupby('Category').size()
#Create a row of overall probabilities out of the crime numbers
#This vector is thus normalized to sum up to 1.
crime_ratios = crime_numbers / len(train)
#Convert to list
probs = crime_ratios.values.tolist()

#load test file to data frame
test = pd.read_csv(test_file,header=0)

#Create a matrix of probabilities for each row of test data
#Each row gets the same values -- the overall probs.
probs_array = np.array([probs]*len(test))

#Create empty data frame for submission file
columns = crime_ratios.index.tolist()
df = pd.DataFrame(probs_array, columns=columns)

df.insert(loc=0,column='Id',value=test['Id'])

with gzip.open(submission_file,'wt') as archive:
    df.to_csv(archive,index=False)
