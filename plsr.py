from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import trimesh
from sklearn.decomposition import PCA
from numpy.linalg import inv







def partial_least_squares_regression(X, Y, n_components=2):
    """
    Perform Partial Least Squares Regression.

    Parameters:
    - X: Input data matrix (independent variables)
    - Y: Output data matrix (dependent variable)
    - n_components: Number of components to extract

    Returns:
    - pls_model: Fitted PLS regression model
    - y_pred: Predicted Y values
    """

    # Split the data into training and testing sets
    # X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.05, random_state=42)
    X_train, X_test, Y_train, Y_test = X[:50], X[50:], Y[:50], Y[50:]

    # Create PLS regression model
    pls_model = PLSRegression(n_components=n_components)

    # Fit the model on the training data
    pls_model.fit(X_train, Y_train)


    # Predict Y values on the test set
    y_pred = pls_model.predict(X_test)


    # Calculate the mean squared error on the test set
    mse = mean_squared_error(Y_test, y_pred)
    print(f'Mean Squared Error on Test Set: {mse}')

    return pls_model,  y_pred




if __name__ == "__main__":

    # Load the data Y 100 * 360
    datapath = Path('data')
    skull_dir = Path("skullskin_align_skinface")
    number_of_samples = 100
    number_of_features = 360
    geodesic_grid_list = np.zeros((100, 360))
    for i,geodesic_grid_path in enumerate(os.listdir(datapath)):
        if geodesic_grid_path.endswith('.npy'):
            geodesic_grid = np.load(datapath / geodesic_grid_path)
            geodesic_grid_list[i] = geodesic_grid

    # Load the data X 100 * 123177
    skull_obj_list = sorted(list(skull_dir.glob('*Skull.ply')))
    number_of_samples = 100
    number_of_features = 123177
    skull_vertice = np.zeros((number_of_samples, number_of_features))

    number_of_samples = 100
    number_of_features = 122907
    face_vertice = np.zeros((number_of_samples, number_of_features))
    face_fs = []
    for i,skull_obj_path in enumerate(tqdm(skull_obj_list)):
        skull_mesh = trimesh.load_mesh(
            skull_obj_path, 
            maintain_order=True, 
            skip_materials=True, 
            process=False
        )
        v = np.array(skull_mesh.vertices).flatten()
        skull_vertice[i] = v

        skin_obj_list = str(skull_obj_path).replace('Skull', 'Skin')
        skin_mesh = trimesh.load_mesh(
            skin_obj_list, 
            maintain_order=True, 
            skip_materials=True, 
            process=False
        )
        v= np.array(skin_mesh.vertices).flatten()
        face_vertice[i] = v
        face_fs.append(skin_mesh.faces)


    model, y_pred = partial_least_squares_regression(skull_vertice, geodesic_grid_list, 10)

    print(y_pred.shape)

    y_pred = np.array(y_pred)
    id_face = np.zeros((50,120))

    for i,geodesics in enumerate(y_pred):
        gd_ps = geodesics.reshape((120, 3))
        face_ps = face_vertice[i+50].reshape((-1, 3))
        for j,gd_p in enumerate(gd_ps):
            id = np.linalg.norm(face_ps - gd_p, axis=1).argmin()
            id_face[i][j] = id

    print(id_face) # 50 * 120





    pca = PCA(n_components=10)
    pca.fit(face_vertice[0:50])
    basis = pca.components_ # 10 * 122907
    mean_face = pca.mean_ # 122907

    basis = basis.reshape((10, -1, 3))
    mean_face = mean_face.reshape((-1, 3))

    X_face = np.zeros((50,122907))
    gt = face_vertice[50:]
    for test_person in range(50):
        geo_grid = id_face[test_person]
        mean_geo = mean_face[geo_grid.astype(int)].reshape(-1,1) # 360 * 1
        basis4geo = basis[:, geo_grid.astype(int)] # 10 * 120 * 3
        basis4geo = basis4geo.reshape((10, -1))   # 10*360


        geo_pts = y_pred[test_person ].reshape(-1,1)
        diff = geo_pts - mean_geo
        coeff =inv(basis4geo.dot(basis4geo.T)).dot(basis4geo).dot(diff).reshape(1,-1)
        X_face[test_person]=coeff.dot(pca.components_) + pca.mean_
        err = np.linalg.norm((gt[test_person] - X_face[test_person]).reshape((-1, 3)),axis=1)
        err = err/err.max() * 255
        v = X_face[test_person].reshape(-1,3)
        c = np.zeros((v.shape[0],3))
        c[:,0] = err

        res_mesh = trimesh.Trimesh(vertices=v,faces=face_fs[test_person+50])
        res_name = os.path.basename(skull_obj_list[test_person+50]).replace('Skull', 'Skin')
        print(res_name)
        res_mesh.export(os.path.join('results/', res_name))

        # points_visual = trimesh.points.PointCloud(vertices=v,colors=c)
        # scene = trimesh.Scene([points_visual])
        # scene.show(smooth=False)





   
    # x_new = pca.transform(face_vertice[50:]) # gt cannot get the facial vertice of test person
    # print(pca.inverse_transform(x_new) - face_vertice[50:])