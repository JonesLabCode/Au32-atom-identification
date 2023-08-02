import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import read_npy

def cluster_tracking(stack):

  file_folder = 'marked frames/'+stack+' marked frames/'
  for frame_num in range(20):
    atoms = np.load(file_folder+'pt_location_frame{}.npy'.format(frame_num))
    num_cluster = 20
    """## Kmeans clustering"""
    imeans = KMeans(n_clusters=num_cluster, random_state=0).fit(atoms)
    klabel = imeans.labels_

    plt.rcParams.update({'font.size': 30});
    plt.figure(figsize=(21,21));
    color = ['r','g','b','#469990','#000075','#000000','#e6194B',
             '#f58231','#ffe119','#bfef45','#3cb44b','#42d4f4','#4363d8','#911eb4',
             'magenta','#a9a9a9','#fabed4','#ffd8b1','#fffac8','#aaffc3','#dcbeff',
             'pink','orange','skyblue','black','purple']
    posx = atoms[:,0]
    posy = 1024-atoms[:,1]
    clusterlabels = klabel
    for i in range(num_cluster):
      clusteri = clusterlabels==i
      if sum(clusteri)> 15:
        plt.scatter(posx[clusteri],posy[clusteri],c=color[i],label = 'cluster {}={} '.format(i, sum(clusteri)));
      #else:
        #print('cluster',i, sum(clusteri),'atoms','cluster too small')

    plt.xlabel('x');
    plt.ylabel('y');
    plt.xlim(0,1024);
    plt.ylim(0,1024);
    plt.legend();
    plt.savefig('sorted clusters/{} sorted clusters/frame{}.png'.format(stack,frame_num));
    plt.close();

