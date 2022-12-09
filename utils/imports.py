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