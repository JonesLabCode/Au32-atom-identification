# Au32-atom-identification
The following code is developed for finding single atoms in STEM images

## Code Structure and Requirements
This code consists of 
main.py
atom_parameters.py
frames_processsing.py
track_cluster.py 
creat_video.py
util.py

Dependencies
- pandas
- scipy
- opencv-python
- numpy
- matplotlib

## Work flow 

1. Crop out single atoms\
Crop out the single atoms in from an image stack with ImageJ, then put all the single atom files into the 'single atoms/x-x atom count/' folder. Examples of the single atoms cropped from one image stack is uploaded

2. Run `main.py`\
`main.py` calls following files sequentially, the function of each file is listed below
   
- `atom_parameters.py` is used to find the proper peak intensity and the kernel size\
The  `atom_parameters.py` assumes the single atom images found in the previous step are from bimodal distribution. With Au and Br as the single atoms, we will find two atom species, the atom with higher peak intensity and larger size is the Au atom. Therefore we can get the average peak intensity for Au atoms, and the average atom radius.

- `frames_processsing.py` is used to generate images with atoms marked.

- `track_clusters.py` are used for automatically counting the number of atoms in each clusters. This is done by Kmean clustering method.
This is a file that supports the identification of Au32 clusters in STEM images. However, the number of atoms counted by the KMeans algorithm is not accurate. After finding the cluster that has around 25-35 atoms, re-examination is done to exclude the single atoms in the background or atoms counted more than once.
