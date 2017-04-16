import math
import numpy as np

# calculate 3d direction of convolution points
# TODO: remake to process any number of points in a convolution
def calculate_direction(coords1, coords2, coords3):
#    X = np.matrix([[1, coords1[0], coords1[1]],
#                   [1, coords2[0], coords2[1]],
#                   [1, coords3[0], coords3[1]]])
#    y = np.matrix([[coords1[2]], [coords2[2]], [coords3[2]]])
#    appr = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
    diff = coords3-coords1
    ln = sum(diff*diff)**0.5
    return diff/ln


# v: electrode positions, numpy array nv x 3
# c: list of electrodes in convolutions (L(L(position in v)))
# N: number of convolutions in each group to find
# dth: maximum angle between perpendicular to direction of the starting
#   convolution and center of new convolution
# dfi: maximum angle between two convolutions to be paired
def make_conv_trees(v, c, N, dth, dfi):
    result = []
    s_dth = math.sin(dth)
    s_dfi = math.sin(dfi)
    # 1. For each initial convolution
    # generate direction vector
    num_c = np.shape(c)[0] # total number of raw convolutions
    dirs = np.zeros([num_c,3]) # direction of each convolution
    for i in range(0,num_c):
        dirs[i,:] = calculate_direction(v[c[i,0]], v[c[i,1]], v[c[i,2]])
    # 2. For each convolution get all other convolutions in dth and dfi angular range
    for i in range(0,num_c):
        poss_pairs_list = [] # possible (by dth and dfi) pairs for checking
        for j in range(i+1, num_c):
            if c[j,1]!=c[i,1]: # if middle points are equal the pair can not exist
                dir_ij = v[c[j,1]] - v[c[i,1]]
                norm_dir_ij = dir_ij / (sum(dir_ij*dir_ij)**0.5) # direction from ith to jth convolution
                cr = np.cross(dirs[i,:], dirs[j,:]) # cross product to evaluate fi angle 
                l_cr = sum(cr*cr)**0.5
                if l_cr<s_dfi and math.fabs(np.dot(dirs[i,:], norm_dir_ij))<s_dth:
                    poss_pairs_list.append(j)
        # 3. Create chain of convolutions based on distance from the initial point
        # 3.1. Calculate base vector to compute distances
        if poss_pairs_list!=[]:
            main_dir = v[c[poss_pairs_list[0],1]] - v[c[i,1]]
            main_dir = main_dir / sum(main_dir*main_dir)**0.5
            # all points with distances
            dists = [(curr_v, np.dot(main_dir, v[c[curr_v,1]] - v[c[i,1]])) for curr_v in poss_pairs_list]
            # add start point
            dists.append((i,0))
            # sort points
            dists.sort(key=lambda v: v[1])
            # generate convolutions
            for start_p in range(0, len(dists)-N+1):
                poss_res = [curr_v for (curr_v,curr_dist) in dists[start_p:start_p+N]]
                if poss_res[0]>poss_res[-1]:
                    poss_res.reverse() # to ensure uniqueness
                if poss_res not in result:
                    result.append(poss_res)
    return result

# tests
# create test array of convolutions (simplest form)
def gen_test_array(N):
    xc = np.repeat(np.arange(1,N+1),N)
    yc = np.reshape(np.repeat(np.asmatrix(np.arange(1,N+1)), N, 0), (1, N*N))
    return np.column_stack([xc, np.asarray(yc)[0], np.repeat([0], N*N)])

test_N = 4
test_v = gen_test_array(test_N)
test_convs = np.array([[0,1,2], [4,5,6], [9,10,11], [12,13,14], [3,7,11], [7,14,15]])
res_seq = make_conv_trees(test_v, test_convs, 3, 0.1, 0.1)
assert(res_seq==[[0,1,3]])
