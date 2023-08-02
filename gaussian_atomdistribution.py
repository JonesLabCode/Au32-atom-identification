import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn
import warnings
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score
from utils import readImages_tif




def atombimodal(stack,bimodal=True):
    # by setting the check_background to true, one can examine how many background pixels in each atom image
    file = 'single atoms/'+stack+" atom count/"+stack+".csv"
    atom_pd = pd.read_csv(file)
    if bimodal:
        integral = atom_pd['integral'].to_numpy().reshape(-1,1)
        peak = atom_pd['pixel intensity']
        area = atom_pd['area']
        imeans = KMeans(n_clusters=2, random_state=0).fit(integral)
        klabel = imeans.labels_
        class1 = klabel==0
        class2 = klabel==1
        plt.scatter(integral[class1],np.zeros_like(integral[class1]))
        plt.scatter(integral[class2],np.zeros_like(integral[class2]))
        plt.ylim((-0.1,0.1))
        plt.title('classify by integrals')
        integralsilhoutte = silhouette_score(integral,klabel)
        print('silhouette_score is {:.2f}'.format(integralsilhoutte))
        print('class1 peak intensity{:.2f}'.format(np.average(peak[class1])))
        print('class2 peak intensity{:.2f}'.format(np.average(peak[class2])))


        class1_integral = np.average(integral[class1])
        class2_integral = np.average(integral[class2])
        std1 = np.std(integral[class1])
        std2 = np.std(integral[class2])

        print("integral of class1 {:.2f}".format(class1_integral),"+-{:.2f}".format(std1))
        print("integral of class2 {:.2f}".format(class2_integral),"+-{:.2f}".format(std2))
        c1_area = np.average(area[class1])
        c1area_std = np.std(area[class1])
        c2area_std = np.std(area[class2])
        c2_area = np.average(area[class2])
        ksize1 = np.sqrt(c1_area/np.pi)*2
        ksize2 = np.sqrt(c2_area/np.pi)*2

        print('area class1 {:.2f}'.format(ksize1),'std ',np.sqrt(c1area_std/np.pi)*2)
        print('area class2 {:.2f}'.format(ksize2),'std ',np.sqrt(c2area_std/np.pi)*2)
        if (class1_integral > class2_integral):
            peak =np.average(peak[class1])
            ksize = ksize1
        else:
            peak=np.average(peak[class2])
            ksize = ksize2
    return int(peak),int(ksize)


stack='1-3'
atombimodal(stack)