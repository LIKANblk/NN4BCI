import math
import numpy as np
import Queue as qu

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

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

# TODO: generate Delaunay triangulation for vertices in the matrix

# reorder edges to make cycle or chain (such that ed(i,2)==ed(i+1,1))
def order_edges(edges):
    # check is there a cycle or not (if there are any vertices 
    # in column 1 but not 2 then it is not a cycle)
    starts = [e[0] for e in edges]
    ends = [e[1] for e in edges]
    # find non-paired vertice if any
    is_cycle = True
    non_pair_start = -1
    for v_st in starts:
        if v_st not in ends:
            is_cycle = False
            non_pair_start = v_st
            break
    if is_cycle:
        non_pair_start = min(starts)
    # now non_pair_start will be starting point for processing
    ordered_edges = []
    curr_v = non_pair_start
    has_next = True
    while has_next: # process all edges
        try:
            curr_ind = starts.index(curr_v)
            has_next = True
            ordered_edges.append((curr_v, ends[curr_ind]))
            curr_v = ends[curr_ind]
            del starts[curr_ind]
            del ends[curr_ind]
        except ValueError:
            has_next = False
    # if any vertex are in starts when processing finishes,
    # the surface is not manifold in start point
    assert(starts == [])
    return ordered_edges
        
assert(order_edges([[3,4],[1,3],[4,5]])==[(1,3),(3,4),(4,5)])
assert(order_edges([[3,4],[1,3],[4,1]])==[(1,3),(3,4),(4,1)])

# class for entry in priority queue to compare entries by key only
class PQEntry:
    def __init__(self, priority, value):
        self.priority = priority
        self.value = value
    def __cmp__(self, other):
         return cmp(self.priority, other.priority)

# initialize search queue if the starting point is vertex
def initialize_queue_by_vertex(v, f, init_v, target_v):
    # get all triangles where init_v is first, second or third vertex,
    # and separate one vertex
    edges = [(curr_f[1], curr_f[2]) for curr_f in f if curr_f[0]==init_v]
    edges.extend([(curr_f[2], curr_f[0]) for curr_f in f if curr_f[1]==init_v])
    edges.extend([(curr_f[0], curr_f[1]) for curr_f in f if curr_f[2]==init_v])
    if edges==[]:
        return [] # can not process a separate vertex
    # sort edges to make chain or cycle (if possible)
    edges = order_edges(edges)
    # set angles and possible positions starting from 
    # the first point in sorted list
    curr_angle = 0
    # format of variants in queue:
    # (target_dist, [v1 v2 v1_flat(x,y) v2_flat(x,y) angle_start_flat(x,y) angle_end_flat(x,y), edge_stack])
    variants = qu.PriorityQueue()
    for (first_v, second_v) in edges:
        # calculate flat coordinates for two triangle vertices
        v_i1 = v[first_v]-v[init_v]
        v_i2 = v[second_v]-v[init_v]        
        len1 = np.linalg.norm(v_i1)
        len2 = np.linalg.norm(v_i2)
        # calculate angle between 0-v1 and 0-v2
        ort1 = v_i1/len1
        ort2 = v_i2/len2
        angle_12 = math.acos(np.dot(ort1,ort2))
        # calculate flat coordinates of each vertex processed
        v1_flat = np.array([len1*math.cos(curr_angle), len1*math.sin(curr_angle)])
        curr_angle -= angle_12
        v2_flat = np.array([len2*math.cos(curr_angle), len2*math.sin(curr_angle)])
        # calculate measure of distance from v1-v2 to target
        dist = np.linalg.norm((v[first_v]+v[second_v])/2 - v[target_v])
#        dist = np.linalg.norm(v[target_v]-v[init_v], v[first_v]+v[second_v])/2 - v[init_v])
        variants.put(PQEntry(dist, [first_v, second_v, v1_flat, v2_flat, v1_flat, v2_flat, [[first_v, second_v, v1_flat, v2_flat]]]))
    return variants

# calculate angle between vectors v and base_v
# to use base_v as fixed base direction
def rel_angle(v, base_v):
    len_b = np.linalg.norm(base_v)
    len_v = np.linalg.norm(v)
    return math.asin((v[1]*base_v[0]-v[0]*base_v[1])/len_b/len_v)

assert(math.fabs(rel_angle(np.array([1,1]), np.array([2,2])))<1e-8)
assert(math.fabs(rel_angle(np.array([1,1]), np.array([0,1]))+math.pi/4)<1e-8)

# test only function: print main part of priority queue info
def print_queue(q):
    lst = []
    for i in q.queue:
        lst.append((round(i.priority, 2),i.value[0],i.value[1], i.value[4], i.value[5]))
    lst.sort(key=lambda v: v[0])
    for item in lst:
        print(item)

# one step in depth for dfs
# variants: priority queue of 
#   (target_dist, [v1 v2 v1_flat(x,y) v2_flat(x,y) angle_start_flat(x,y) angle_end_flat(x,y) stack])
# init_vertex,target_vertex: positions
# return: (False, new_variants, []) or (True, [], answer)
def dfs_step(v, f, variants, init_vertex, target_vertex, step_num):
    eps = 1e-6
    # if there are no any more variants, return 0
    if variants.empty():
        return (False, variants, [])
    # get top element
    next_data = variants.get()
    v1 = next_data.value[0]
    v2 = next_data.value[1]
    v1_flat = next_data.value[2]
    v2_flat = next_data.value[3]
    a1_flat = next_data.value[4]
    a2_flat = next_data.value[5]
    old_stack = next_data.value[6]
    # check if the target is achieved; if yes, return result
    if v1==target_vertex or v2==target_vertex:
        # if the target point is in angle
        result_flat = v1_flat
        if v2==target_vertex:
            result_flat = v2_flat
        cross1 = a1_flat[0]*result_flat[1]-a1_flat[1]*result_flat[0]
        cross2 = a2_flat[1]*result_flat[0]-a2_flat[0]*result_flat[1]
        if cross1<=eps and cross2<=eps:
            return (True, [], old_stack)
        else:
            return (False, variants, [])
    # if the target is not achieved yet, expang two edges
    # 0. Find new vertex in f
    found_v = [c_f[2] for c_f in f if c_f[0]==v2 and c_f[1]==v1]
    found_v.extend([c_f[0] for c_f in f if c_f[1]==v2 and c_f[2]==v1])
    found_v.extend([c_f[1] for c_f in f if c_f[2]==v2 and c_f[0]==v1])
    if found_v==[]: # there is no such a face
        return (False, variants, []) # variants are changed by get already
    assert(len(found_v)==1) # else the surface is non-manifold
    v3 = found_v[0]    
    # 1. Calculate the new vertex position
    l13 = np.linalg.norm(v[v3] - v[v1])
    l12 = np.linalg.norm(v[v2] - v[v1])
    angle_213 = math.acos(np.dot(v[v3] - v[v1], v[v2] - v[v1])/l13/l12)
    fv12 = v2_flat-v1_flat
    fv12_nrm = fv12 / l12
    fv13_nrm = np.array([fv12_nrm[0]*math.cos(angle_213)-fv12_nrm[1]*math.sin(angle_213),
                         fv12_nrm[0]*math.sin(angle_213)+fv12_nrm[1]*math.cos(angle_213)])
    v3_flat = v1_flat + fv13_nrm*l13
    # 2. Calculate new variants
    # [v1 v2 v1_flat(x,y) v2_flat(x,y) angle_start_flat(x,y) angle_end_flat(x,y), edge_stack]
    # calculate flat angles for corresponding vertices
    anglev3 = 0
    old_angle1 = rel_angle(a1_flat,v3_flat)
    old_angle2 = rel_angle(a2_flat,v3_flat)
    anglev1 = rel_angle(v1_flat,v3_flat)
    anglev2 = rel_angle(v2_flat,v3_flat)
    # add edge 13 if it is possible
    if (v1_flat[0]*v3_flat[1]-v1_flat[1]*v3_flat[0] < -eps
        and anglev3<old_angle1 and anglev1>old_angle2):
        # calculate boarding points for angles
        angle_list = [(old_angle1, a1_flat),
                      (old_angle2, a2_flat),
                      (anglev1, v1_flat),
                      (anglev3, v3_flat)]
        angle_list.sort(key = lambda val: val[0]) # new angles are in angle_list[2][1] and [1][1]
        target_dist_13 = np.linalg.norm((v[v1]+v[v3])/2 - v[target_vertex])
        stack_13 = old_stack[:]
        stack_13.append([v1,v3,v1_flat,v3_flat])
        variants.put(PQEntry(target_dist_13,
                             [v1, v3, v1_flat, v3_flat, 
                              angle_list[2][1], angle_list[1][1], stack_13]))
    # add edge 32 if it is possible
    if (v3_flat[0]*v2_flat[1]-v3_flat[1]*v2_flat[0]<-eps
        and anglev2<old_angle1 and anglev3>old_angle2):
        # calculate boarding points for angles
        angle_list = [(old_angle1, a1_flat),
                      (old_angle2, a2_flat),
                      (anglev3, v3_flat),
                      (anglev2, v2_flat)]
        angle_list.sort(key = lambda val: val[0]) # new angles are in angle_list[2][1] and [1][1]
        target_dist_32 = np.linalg.norm((v[v2]+v[v3])/2 - v[target_vertex])
        stack_32 = old_stack[:]
        stack_32.append([v3,v2,v3_flat,v2_flat])
        variants.put(PQEntry(target_dist_32,
                             [v3, v2, v3_flat, v2_flat,
                              angle_list[2][1], angle_list[1][1], stack_32]))
    return (False, variants, [])

# apply depth-first search (dfs_step) to calculate path
def find_path(v,f,init_vertex,target_vertex):
    variants = initialize_queue_by_vertex(test_v, test_fs, start_v, end_v)
    res = False
    i=0
    while res==False and not variants.empty():
        [res, variants, answ] = dfs_step(test_v, test_fs, variants, start_v, end_v,i)
        i+=1
    return (res,answ)

# get 3D coordinates of path vertices
def trace_path(v, f, init_vertex, target_vertex, path):
    eps = 1e-6
    assert(path!=[])
    # get flat coordinates of the target vertex
    target_f = None
    if path[-1][0]==target_vertex:
        target_f = path[-1][2]
    elif path[-1][1]==target_vertex:
        target_f = path[-1][3]
    else:
        raise ValueError # target is not in last element of the path
    # trace through the path
    dir_f = target_f/np.linalg.norm(target_f)
    curr_point = v[init_vertex]
    trace = [curr_point]
    for ed in path:
        # find flat crossing point of edge and line
        v1_f = ed[2]
        v2_f = ed[3]
        e12 = v2_f-v1_f
        e12 = e12/np.linalg.norm(e12)
        # if edge and line are parallel, add edge to the result 
        if math.fabs(dir_f[0]*e12[1] - dir_f[1]*e12[0])<eps:
            if np.linalg.norm(v[ed[0]]-curr_point)<eps:
                curr_point = v[ed[1]]
                trace.append(curr_point)
            elif np.linalg.norm(v[ed[1]]-curr_point)<eps:
                curr_point = v[ed[0]]
                trace.append(curr_point)
            else:
                raise ValueError # impossible combination?
        else:
            # find crossing point of edge and line
            D = (v1_f[0]-v2_f[0])*dir_f[1] - (v1_f[1]-v2_f[1])*dir_f[0]
            Dx = v1_f[0]*dir_f[1] - v1_f[1]*dir_f[0]
            alpha = Dx/D
            curr_point = (1-alpha)*v[ed[0]]+alpha*v[ed[1]]
            trace.append(curr_point)
    return trace

# plot tracing results using matplotlib
def plot_results(v,f,trace):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot grid
    edge_set = set()
    for face in f: # sort edges
        if face[0]>face[1]:
            edge_set.add((face[1], face[0]))
        else:
            edge_set.add((face[0], face[1]))
        if face[0]>face[2]:
            edge_set.add((face[2], face[0]))
        else:
            edge_set.add((face[0], face[2]))
        if face[2]>face[1]:
            edge_set.add((face[1], face[2]))
        else:
            edge_set.add((face[2], face[1]))
    for ed in edge_set: # plot edges
        ax.plot([v[ed[0]][0], v[ed[1]][0]],
                [v[ed[0]][1], v[ed[1]][1]],
                zs = [v[ed[0]][2], v[ed[1]][2]], color='b')
    # plot trace
    for i in range(len(trace)-1):
        c = 'r'
        if i % 2==0:
            c = 'g'
        ax.plot([trace[i][0], trace[i+1][0]],
                [trace[i][1], trace[i+1][1]],
                zs = [trace[i][2], trace[i+1][2]], color = c)
    plt.show()
# tests

# create test array of points and faces (the simplest form, flat triangles)
def gen_test_array(N):
    # vertices
    xc = np.repeat(np.arange(1.0+10,float(N+1)+10),N)
    yc = np.reshape(np.repeat(np.asmatrix(np.arange(1.0+10,float(N+1)+10)), N, 0), (1, N*N))
    vs = np.column_stack([xc, np.asarray(yc)[0], np.repeat([0], N*N)])
    # faces: temporary manual generation instead of Delaunay generation
    start_v = [i*N+j for j in range(0,N-1) for i in range(0,N-1)]
    fs = [[v, v+1, v+1+N] for v in start_v] 
    fs.extend([[v, v+1+N, v+N] for v in start_v])
    return (vs, fs)

# create test array of points and faces (more sophisticated 3d surface)
def gen_test_3d_array(N):
    # vertices
    xc = np.repeat(np.arange(1.0+10,float(N+1)+10),N)
    yc = np.reshape(np.repeat(np.asmatrix(np.arange(1.0+10,float(N+1)+10)), N, 0), (1, N*N))
    yc = np.asarray(yc)[0]
    zc1 = np.exp(-xc + N/2)*1000
    zc2 = np.exp(-yc + N/2)*1000
    vs = np.column_stack([xc, yc, zc1+zc2])
    # faces: temporary manual generation instead of Delaunay generation
    start_v = [i*N+j for j in range(0,N-1) for i in range(0,N-1)]
    fs = [[v, v+1, v+1+N] for v in start_v] 
    fs.extend([[v, v+1+N, v+N] for v in start_v])
    return (vs, fs)

if __name__=='__main__':
    # generate test surface for tracing
    test_N = 8
    start_v = 7
    end_v = 56
    (test_v, test_fs) = gen_test_3d_array(test_N)
    # find path and trace it if found
    (result, path) = find_path(test_v,test_fs,start_v,end_v)
    if result:
        trace_seq = trace_path(test_v, test_fs, start_v, end_v, path)
        plot_results(test_v,test_fs,trace_seq)
    else:
        print('Path not found')
    # generate test surface for convolutions pairing
    (test_v, test_fs) = gen_test_array(4)
    test_convs = np.array([[0,1,2], [4,5,6], [9,10,11], [12,13,14], [3,7,11], [7,14,15]])
    # calculate pairs with the simplest algorithm (no surface paths)
    res_seq = make_conv_trees(test_v, test_convs, 3, 0.1, 0.1)
    assert(res_seq==[[0,1,3]])
