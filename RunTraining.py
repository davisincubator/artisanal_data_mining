import utils
from ModelDefinitions import *
import pandas as pd


params = {'load_file':'model_saves/model-cnn_256x256_5000',
          'save_file': 'cnn',
          'save_interval': 1000,
          'batch_size': 32,
          'lr': .0001,  # Learning rate
          'rms_decay': 0.9,  # RMS Prop decay
          'rms_eps': 1e-8,  # RMS Prop epsilon
          'width':256,
          'height':256,
          'numParam':17,
          'num_epoch':5000}


## initialize net
net = CNN(params)#(from ModelDefinitions.py)

# get files and labels
train_files = utils.file('../planet/train-tif-v2/*')
train_labels = pd.read_csv('../planet/train_v2.csv')
train_labels.index = train_labels['image_name']
# this ugly bit gets unique categories
categories = sorted(list(set(' '.join(train_labels['tags'].tolist()).split())))
# dict converts labels to indices of labels matrix
cat_dict = dict(zip(categories,range(len(categories))))

###### run training ######

# moving average cost
avgCost = 100*[np.inf]

for itrain in range(params['num_epoch']):
  # read random images
  batchNames,batchImages = utils.get_batch(train_files,params['batch_size'])
  # get labels of batch images
  labelsList = train_labels.loc[batchNames]['tags'].tolist()
  labels = utils.get_labels(cat_dict,labelsList)
  # train net
  cnt, cost = net.train(batchImages,labels)
  #update moving average cost
  avgCost.append(cost)
  avgCost.pop(0)
  print('count: {}, cost: {}, avg_cost: {}'.format(cnt, cost, np.mean(avgCost)))
  # save network
  if (params['save_file']):
    if cnt % params['save_interval'] == 0:
      net.save_ckpt('model_saves/model-' + params['save_file'] + "_" + str(params['width'])+'x'+str(params['height']) + '_' + str(cnt))
      print('Model saved')
