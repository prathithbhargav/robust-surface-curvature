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


"""
Protein - Ligand

"""
def generate_the_complimentarity_plot(path,sub_path,OUTPUT):
    
    pdb_id = collections.defaultdict(list)
    dms_id = collections.defaultdict(list)
    dms_path = os.path.expanduser(sub_path.as_posix()+"/*.dms")

    for files in glob.glob(dms_path):
        filename = files
        structure_id = regex.search(
            r"(?:.+[/\\])(.+)(?:\.dms)", filename, flags=regex.I).group(1)
        s1 = structure_id
        dms_id[structure_id].append(filename)

    dms_normal = {}
    for name_dms in dms_id:
        with open(dms_id[name_dms][0], 'r') as f:
            k = f.read()
        pattern = re.compile(r"(.{20,})(?:\bA\b)", flags=re.M | re.I)
        pattern2 = re.compile(r"(.{20,})(?:\bS\w+\b)(.+)", flags=re.M)
        l1 = pattern.findall(k)
        l2 = pattern2.findall(k)
        iterables1 = {}
        iterables_orig = {}
        iterables_normal_area = {}
        for x, y in l2:
            search = regex.search(
                r"(\w{,3})\s*(\w+)(?:\*?)\s*(\w+)(?:\*|'?)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)", x)
            iterables1[tuple(map(float, [search.group(4), search.group(
                5), search.group(6)]))] = list(map(float, y.split()[1:]))
        dms_normal[name_dms] = copy.deepcopy(iterables1)

    for files in glob.glob((path/OUTPUT).as_posix()+"/*.pdb"):
        filename = files
        structure_id = regex.search(
            r"(?:.+/)(.{4})(?:.*_X\.pdb)", filename).group(1)
        pdb_id[structure_id].append(filename)


    for name_pdb in pdb_id:
        for i in itertools.combinations(pdb_id[name_pdb], 2):
            # print(i)
            with open(i[0], 'r') as f:
                k = f.readlines()
            iterables = {}
            arr1 = []
            arr1_norm = []
            if(regex.search(r"_lig_X", i[0])):
                suffix = "_lig"
            else:
                suffix = ""
            for x in k:
                iterables.setdefault(x[60:66].replace(" ", ""), []).append(list
                                                                           (map(float, [x[30:38].replace(" ", ""),
                                                                                        x[38:46].replace(
                                                                                            " ", ""),
                                                                                        x[46:54].replace(" ", "")])))
                arr1.append(list(map(float, [x[30:38].replace(" ", ""),
                                             x[38:46].replace(" ", ""),
                                             x[46:54].replace(" ", ""), x[60:66].replace(" ", "")])))
                arr1_norm.append(dms_normal[name_pdb+suffix][tuple(map(float, [x[30:38].replace(" ", ""),
                                                                               x[38:46].replace(
                                                                                   " ", ""),
                                                                               x[46:54].replace(" ", "")]))])
            with open(i[1], 'r') as f:
                k1 = f.readlines()

            iterables1 = {}
            arr2 = []
            arr2_norm = []
            if(regex.search(r"_lig_X", i[1])):
                suffix = "_lig"
            else:
                suffix = ""
            for x in k1:
                iterables1.setdefault(x[60:66].replace(" ", ""), []).append(list
                                                                            (map(float, [x[30:38].replace(" ", ""),
                                                                                         x[38:46].replace(
                                                                                             " ", ""),
                                                                                         x[46:54].replace(" ", "")])))
                arr2.append(list(map(float, [x[30:38].replace(" ", ""),
                                             x[38:46].replace(" ", ""),
                                             x[46:54].replace(" ", ""), x[60:66].replace(" ", "")])))
                arr2_norm.append(dms_normal[name_pdb+suffix][tuple(map(float, [x[30:38].replace(" ", ""),
                                                                               x[38:46].replace(
                                                                                   " ", ""),
                                                                               x[46:54].replace(" ", "")]))])
            arr1 = np.array(arr1)
            arr2 = np.array(arr2)
            arr1_norm = np.array(arr1_norm)
            arr2_norm = np.array(arr2_norm)
            arr1 = da.from_array(arr1)
            arr1_norm = da.from_array(arr1_norm)
            arr2 = da.from_array(arr2)
            arr2_norm = da.from_array(arr2_norm)
            # print('works')
            normal_product = da.dot(arr2_norm, arr1_norm.T)
            # print('worksNP')
            arr_dist = dask_distance.euclidean(
                arr2[:, 0:3], arr1[:, 0:3])
            # print('cdist')
            new_dist = da.exp(-1*(arr_dist-da.mean(arr_dist, axis=0))
                              ** 2/(2*da.var(arr_dist, axis=0)))

            new_curv = dask_distance.cityblock(arr2[:,3:4], -1*arr1[:, 3:4])
            new_curv = da.asarray(new_curv,chunks=(500,500))
            new_dist = da.asarray(new_dist,chunks=(500,500))
            new_dist.compute()
            dat_new = (da.multiply(new_curv, new_dist)).flatten()
#             length_of_array = dat_new.size
#             value_division = length_of_array//5
#             hist_final = np.zeros((10))
#             p = pd.DataFrame(dat_new)
#             for n in range(5):
#                 hist,bins = da.histogram(dat_new[n*value_division:n+1*value_division].compute(),bins=20, range=[0,100])
                
            
#             dat_new = dat_new.compute()
#             print(dat_new)
#             length = len(dat_new)
#             hist, bin_edges = da.histogram(dat_new, bins=10, range=dat_new.)
#             h = hist.compute()
#             hist = hist.compute()
#             plt.bar(bin_edges[:int(-1)], hist)
#             h, bins = da.histogram(dat_new, bins=20, range=[0,100],density=True)
#             h_final = h.compute(ch)
#             plt.hist(h_final,bins)

            plt.hist(dat_new, bins=20, density=True, color='gray', alpha=0.8)
            plt.title("%s_%s" % (name_pdb, "_lig"))

            plt.savefig(path.as_posix()+"/%s_%s_plot.jpeg" %
                        (name_pdb, "_lig"), format='jpeg', 
                        dpi=300)
            plt.show()
