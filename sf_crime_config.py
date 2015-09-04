import os.path

project_dir = '/Users/josh/dev/kaggle/sf-crime'

conf_dir = __file__

data_dir = os.path.join(project_dir, 'data')
data_raw_dir = os.path.join(data_dir, 'raw')

submission_dir = os.path.join(project_dir, 'sub')

train_raw = os.path.join(data_raw_dir, 'train.csv')
test_raw = os.path.join(data_raw_dir, 'test.csv')
