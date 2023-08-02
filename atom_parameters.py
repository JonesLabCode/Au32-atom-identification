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

# in the TEM images there are background pixels that shouldn't be counted as the integral
# one should set a proper background pixel intensity
# by setting the check_background=True
# The number of 'background pixels' will be printed out
# if the thresh is a proper number, the background pixel in each single atom image shouldn't exceed 50% of total atom image size.
def extract_imgs(stack,save_atominfo=False,check_background=False):
    dir = 'single atoms/' + stack + ' atom count'
    image_names = readImages_tif(dir)
    if stack=='1-1':
        thresh = 32
        dir = 'single atoms/' + stack + ' atom count'
        image_names = readImages_tif(dir)
    if stack=='1-3':
        #thresh = 35
        dir_frame10 = 'single atoms/' + stack + ' atom count/frame 10'
        #dir_frame3 = 'single atoms/' + stack + ' atom count/frame 3'
        thresh = 45 #for frame 10
        image_names = readImages_tif(dir_frame10)
    if stack =='1-4':
        thresh = 50
    if stack=='1-2':
        thresh=70
    if stack=='1-5':
        thresh = 35 #frame=0
        thresh = 65
        dir = 'single atoms/' + stack + ' atom count' + '/frame 10'
        image_names = readImages_tif(dir)
    if stack=='2-2':
        thresh = 60
    if stack=='2-3':
        thresh = 44




    colum_names = ['integral','peak','integral average','height','width','area']
    df = pd.DataFrame(columns=colum_names)
    background_pix_number=[]
    for i, imgname in enumerate(image_names):
        img = cv2.imread(imgname,0)
        h,w = img.shape
        peak = np.max(img)
        below = img<thresh
        img[below] = 0
        integral = np.sum(img)
        integral_avg = np.sum(img)/(h*w-np.sum(below))
        background_pix_number.append(np.sum(below))
        df=df.append({'integral':integral,
                   'peak':peak,
                   'integral average':integral_avg,
                   'height':h,
                   'width':w,
                   'area':h*w},ignore_index=True)
        if check_background:
            print('atom',i,'background pixels',np.sum(below),'total pixels',h*w,'peak intensity',peak)
    print('average number of pixels recognized as background pixels',np.average(background_pix_number))
    if save_atominfo == True:
        df.to_csv('single atoms/{}.csv'.format(stack))


    return df


def atombimodal(stack,bimodal=True):
    # by setting the check_background to true, one can examine how many background pixels in each atom image
    atom_pd = extract_imgs(stack, save_atominfo=False, check_background=False)
    if bimodal:
        integral = atom_pd['integral'].to_numpy().reshape(-1,1)
        peak = atom_pd['peak']
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

        print('area class1 {:.2f}'.format(c1_area),'std ',c1area_std)
        print('area class2 {:.2f}'.format(c2_area),'std ',c2area_std)
        if class1_integral > class2_integral:
            peak =np.average(peak[class1])
            ksize = np.sqrt(c1_area-c1area_std)
        else:
            peak=np.average(peak[class2])
            ksize = np.sqrt(c2_area-c2area_std)
        return int(peak),int(ksize)


stack='1-4'
atom_pd = extract_imgs(stack, save_atominfo=True, check_background=True)
atombimodal(stack)