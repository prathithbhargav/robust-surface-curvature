from utils.imports import *
# from utils.write_the_X_file import *
from utils.Hypersphere import fit_hypersphere
from utils.read_msms import read_msms
from utils.linear_algebra_functions import *
# from utils.generate_the_complimentarity_plot import *
from utils.PDB_reader import *

list_of_arguements = sys.argv

input_folder_name = list_of_arguements[1]
output_folder_name = list_of_arguements[2]

path = Path(os.path.abspath('./data/'))
sub_path = Path(os.path.abspath('./data/'+input_folder_name))
OUTPUT = output_folder_name
try:
    os.mkdir(os.path.expanduser(path/OUTPUT))
    # this is making the directory of that name
except FileExistsError:
    print("it exists")
    
df_of_dms_files = pd.DataFrame(columns=['file_id','file_path'])
for files in glob.glob(sub_path.as_posix()+"/*.dms"): #gets files having .dms extension in the said directory
    filename = files
    if filename in list(df_of_dms_files['file_path']):
        break
    else:
        structure_id = regex.search(
            r"(?:.+[/\\])(.+)(?:\.dms)", filename).group(1) 
        s1 = structure_id #s1 becomes structure ID
        new_df = pd.DataFrame({'file_id':[s1],'file_path':[filename]})
        df_of_dms_files = pd.concat([df_of_dms_files,new_df],ignore_index=True)
#     df_of_dms_files=pd.concat(df_of_dms_files,pd.DataFrame({'file_id':s1,'file_path':filename}),ignore_index=True)
# df_of_dms_files

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

for i in range(2):
    filename = df_of_dms_files.iloc[i][1]
    s1 = df_of_dms_files.iloc[i][0]
    write_pdb_X_file(filename=filename,s1=s1,path = path,sub_path=sub_path,OUTPUT=OUTPUT)
    time.sleep(2)
print('Done with generation of the X files')
generate_the_complimentarity_plot(path,sub_path,OUTPUT)
print('Done with generation of the complimentary plot')