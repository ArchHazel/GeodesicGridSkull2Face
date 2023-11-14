import numpy as np
import trimesh
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import networkx as nx



# skull_dir = Path('/mnt/h/Dataset/SkullSkinDataset/')
skull_dir = Path('.')
skin_lmk_ids = np.loadtxt(skull_dir / 'skin_ids.txt').astype(np.int32)

out_dir = Path('data')
out_dir.mkdir(exist_ok=True)

skin_obj_list = sorted((skull_dir / 'skullskin_align_skinface').glob('*Skin.ply'))


skull_ids = []


def find_tip_of_nose(skin_mesh):
    # the tip of the nose is the vertex with the maximum z coordinate
    tip_of_nose_idx = skin_mesh.vertices[:,2].argmax()
    nose = skin_mesh.vertices[tip_of_nose_idx]
    return nose, tip_of_nose_idx

def find_boundary_point(skin_mesh):
    # an edge that occurs only once is on the border
    # return vertex indices for unique edges 
    border_edges = skin_mesh.edges[trimesh.grouping.group_rows(skin_mesh.edges_sorted, require_count=1)]
    bp_idx = np.unique(border_edges.flatten())
    boundarypoint =  skin_mesh.vertices[bp_idx] # (# of vertices, 3)

    # add a dimension to the unit boundarypoint to make it (# of vertices, 2, 3)
    boundarypoint_angle = np.zeros((boundarypoint.shape[0], 2, boundarypoint.shape[-1]))
    boundarypoint_angle[:,0] = boundarypoint.copy()
    boundarypoint_angle[:,:,2] = 0
    for i in range(boundarypoint_angle.shape[0]):
        boundarypoint_angle[i,0]  = trimesh.transformations.unit_vector(boundarypoint_angle[i, 0])

    # rotate the vector by 6 degree each time radially
    degree = 30
    
    rotated_vector = np.array([0,1,0])
    bp_select = np.zeros((360//degree, 3))
    bp_select_idx = np.zeros((360//degree, 1))
    for i in range(360//degree):
        boundarypoint_angle[:,1,:] = rotated_vector
        # bp_select = boundarypoint[trimesh.geometry.vector_angle(boundarypoint_angle).argmin()]
        angles = trimesh.geometry.vector_angle(boundarypoint_angle)
        angles_idx = angles.argmin()
        bp_select[i] = boundarypoint[angles_idx]  
        bp_select_idx[i] = bp_idx[angles_idx] 
        # ax.scatter([skin_mesh.vertices[bp_idx[angles_idx]][0]], [skin_mesh.vertices[bp_idx[angles_idx]][1]], [skin_mesh.vertices[bp_idx[angles_idx]][2]], color='g')
        rotated_vector = R.from_euler('z', degree, degrees=True).apply(rotated_vector)

    return boundarypoint,bp_select_idx

def shortest(mesh,start,end):
    
    # edges without duplication
    edges = mesh.edges_unique

    # the actual length of each unique edge
    length = mesh.edges_unique_length

    # create the graph with edge attributes for length
    g = nx.Graph()
    for edge, L in zip(edges, length):
        g.add_edge(*edge, length=L)



    # run the shortest path query using length for edge weight
    path = nx.shortest_path(g, source=start, target=end, weight="length")

    # calculate the path length from source to target
    path_length =np.zeros(len(path))
    for i,p_v in enumerate(path):
        if i == 0:
            continue
        mesh.vertices[path[i-1]] 
        path_length[i] = np.linalg.norm(mesh.vertices[path[i-1]] - mesh.vertices[p_v]) + path_length[i-1]

    return path,path_length

def visualize_path(mesh, path, start, end):
    # VISUALIZE RESULT
    # make the sphere transparent-ish
    mesh.visual.face_colors = [100, 100, 100, 100]
    # Path3D with the path between the points
    path_visual = trimesh.load_path(mesh.vertices[path])
    # visualizable two points
    points_visual = trimesh.points.PointCloud(mesh.vertices[[start, end]])

    # create a scene with the mesh, path, and points
    scene = trimesh.Scene([points_visual, path_visual, mesh])

    scene.show(smooth=False)

for skin_obj_path in tqdm(skin_obj_list):

    skull_ids.append(str(skin_obj_path.stem).replace('Skin', 'Skull'))

    skin_mesh = trimesh.load_mesh(
        skin_obj_path, 
        maintain_order=True, 
        skip_materials=True, 
        process=False
    )




    nose, nose_idx = find_tip_of_nose(skin_mesh)

    boundarypoint, bp_idx = find_boundary_point(skin_mesh)

    skin_mesh.visual.face_colors = [100, 100, 100, 100]
    # Path3D with the path between the points
  
    # create a scene with the mesh, path, and points
    # scene = trimesh.Scene( skin_mesh)
    path_sl = []
    path_length_sl = []
    max_length = []
    for bp_i in bp_idx:
        path,path_length = shortest(skin_mesh,int(nose_idx),int(bp_i))
        path_sl.append(path)
        path_length_sl.append(path_length)
        max_length.append(path_length.max())
        path_visual = trimesh.load_path(skin_mesh.vertices[path])
        # scene.add_geometry(path_visual)

    min_idx = np.array(max_length).argmin()
    path_length_min = max_length[min_idx]
    equ_spaced_length = np.linspace(0,path_length_min,10)

    # query the index of the path that is closest to the equi-spaced length
    
    kp_idx = []
    kp = []


    for p,d in zip(path_sl,path_length_sl):
        for esl in equ_spaced_length:
            kp_idx.append(p[np.abs(d - esl).argmin()])
            kp.append(skin_mesh.vertices[p[np.abs(d - esl).argmin()]])

    kp_idx = np.array(kp_idx)
    kp = np.array(kp).flatten()
    np.save(out_dir / f'{skin_obj_path.stem}_kp.npy', kp)

    # scene.add_geometry(trimesh.points.PointCloud(skin_mesh.vertices[kp_idx]))
    # scene.show(smooth=False)
    v = np.array(skin_mesh.vertices)
    x_min, x_max = v[:, 0].min(), v[:, 0].max()
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    diag = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    skin_size = np.array([dx, dy, dz, diag])
    
