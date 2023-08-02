from frames_processing import localization
from atom_parameters import atombimodal
from track_clusters import cluster_tracking



stack='1-4'

if stack=='1-4':
    threshold, size = atombimodal(stack, bimodal=True)
    print(threshold, size)
    localization(stack, threshold, size)
    cluster_tracking(stack)