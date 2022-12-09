import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import regex
import glob
import time
import itertools
from collections import namedtuple
import collections
import copy
from scipy.spatial import distance
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from math import sqrt
from sys import stderr
from numpy import linalg
import networkx as nx
import re
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
#key functions
# from utils.Hypersphere import fit_hypersphere
# from utils.read_msms import read_msms
# from utils.linear_algebra_functions import *
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.PDBParser import PDBParser
from sklearn.preprocessing import StandardScaler
from scipy.spatial import Delaunay, ConvexHull
import pickle
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.vq import kmeans, vq
import scipy.cluster.hierarchy as sch
from collections import OrderedDict
import subprocess
import unittest
import glob
import matplotlib.pyplot as plt
import dask.array as da
import dask_distance
import dask.dataframe as daf
import numpy as np
import hvplot.pandas  
import hvplot.dask  

SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams.update({'figure.autolayout': True})
def write_pdb_X_file(filename,s1,path,sub_path,OUTPUT):

    with open(filename, 'r') as f:
        k = f.read()
    # pattern is created by the compile function in re
    pattern = re.compile(r"(.{20,})(?:\bA\b)", flags=re.M | re.I)
    pattern2 = re.compile(r"(.{20,})(?:\bS\w+\b)(.+)", flags=re.M)
    
    l1 = pattern.findall(k) #k which is the filename, we are finding whether the characters that are defined by the pattern are found
    l2 = pattern2.findall(k)
    iterables1 = {}
    iterables_orig = {}
    iterables_normal_area = {}

    for x, y in l2:
        search = regex.search(
            r"(\w{,3})\s*(\w+)(?:\*?)\s*(\w+)(?:\*|'?)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)", x)
        iterables1[tuple(map(float, [search.group(4), search.group(5), search.group(6)]))] = [
            x, list(map(float, y.split()[:]))]
        
    

    for x in l1:
        search = regex.search(
            r"(\w{,3})\s*(\w+)(?:\*?)\s*(\w+)(?:\*|'?)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)", x)
        iterables_orig[tuple(
            map(float, [search.group(4), search.group(5), search.group(6)]))] = [x]
    pattern_new = regex.compile(
        r"(\w{3})\s*(\w+)\s*(\w+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)")
    data = np.array([x for x in iterables1.keys()])
    data = np.array(data, 'float64')
    Z = linkage(data, 'complete')  # ward --> complete
    max_d = 5  # patch
    clusters = fcluster(Z, max_d, criterion='distance')
    curvature = collections.defaultdict(list)
    centroid = np.median(data, axis=0)
    with open(os.path.expanduser(path.as_posix()+'/%s/%s_%s.pdb' % (OUTPUT, s1, 'X')), 'w') as f:
        dist = []
        j = 0
        for i in range(1, max(clusters)+1):
            curv1_p = []
            curv2_p = []
            curv_m = fit_hypersphere(data[clusters == i])
            ci = curv_m[1]
            count = []
            d_centroid = np.linalg.norm(centroid-ci)

            for x in data[clusters == i]:

                d = np.linalg.norm(ci-x)
                d_c = np.linalg.norm(centroid-x)
                if d_c > d_centroid:
                    # if d>curv_m[0]:
                    count.append(1)
                    
                    curv1_p.append(x)
                else:
                    count.append(-1)
                    curv2_p.append(x)

            A = (len(curv1_p)/len(data[clusters == i]))
            B = (len(curv2_p)/len(data[clusters == i]))
            for x in curv1_p:
                curvature[tuple(x)] = A*100/curv_m[0]**1
            for x in curv2_p:
                curvature[tuple(x)] = B*-100/curv_m[0]**1  # put - sign

    j = 0
    with open(os.path.expanduser(path.as_posix()+'/%s/%s_%s.pdb' % (OUTPUT, s1, 'X')), 'w') as f:
        for _, x in enumerate(curvature.keys()):

            loc1 = iterables1[tuple(x)]
            loc = loc1[0].split()
            print("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format("ATOM", j, "A", " ", loc[0], "X",
                                                                                                                                  int(loc[1].rstrip(regex.search(r'(\d+)(.*)', loc[1]).group(2))), '', x[0], x[1], x[2], loc1[1][0],  curvature[tuple(x)], '', loc[2]), file=f)

            j += 1

    dist = [curvature[x] for x in curvature]
    dist = np.array(dist)
    dots = len(dist)
    plt.figure()
    plt.xlabel(
        "Curvature($\kappa$)\n$\longleftarrow$ concave | convex $\longrightarrow$")
    plt.ylabel("number of surface points")
    plt.title('%s %s:Number of surface points: %d\nScaling factor: 100*$\kappa$' %
              (s1.upper(), "", len(dist)))
    plt.hist(dist, bins=15, color='gray', alpha=0.8)
    plt.savefig(os.path.expanduser(path.as_posix()+'/%s/%s_%s_%s hist.jpeg' %
                                   (OUTPUT, dots, s1, 'X')), format='jpeg', dpi=300)
#     plt.show()
