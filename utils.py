import os
from glob import glob
def readImages_tif(image_dir):
    extensions = ['tif']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))

    return image_files

def read_npy(dir):
    extensions = ['npy']

    search_paths = [os.path.join(dir, '*.' + ext) for ext in extensions]
    npy_files = sorted(sum(map(glob, search_paths), []))

    return npy_files
def readImages_png(image_dir):
    extensions = ['png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))

    return image_files