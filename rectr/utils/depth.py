import numpy as np
import scipy.ndimage as nd


def depth2normals(depth, normalize=False):
    """ Compute a surface normals map from a depth map.

    By default, the normals have values in the range [-1, 1]. If `normalize`
    is set to True, the normals are normalized to have values in the range


    Args:
        depth (np.array): Depth map, shape (H, W).
        normalize (bool): Whether to normalize the normals to 0-1 or not.

    Returns:
        np.array: Surface normals, shape (H, W, 3).
    """
    depth_ = nd.gaussian_filter(depth, sigma=0.75)
    zx = nd.sobel(depth_, axis=1) * 16
    zy = nd.sobel(depth_, axis=0) * 16

    normal = np.dstack((-zx, -zy, np.ones_like(depth)))  # (H, W, 3)

    # Divide normals by magnitude to get unit normals
    normal /= np.linalg.norm(normal, axis=2)[:, :, None]

    # Ensure normals are between -1 and 1
    normal = np.clip(normal, -1, 1)

    # Normalize normals to 0-1, if requested
    if normalize:
        normal = (normal + 1) / 2

    return normal


def depth2xyz(depth, K):
    """ Convert depth map to xyz map.
    
    Args:
        depth (numpy array): Depth map, shape (H, W).
        K (dict): Camera parameters, containing the following keys:

            - shift_x: Horizontal shift of the principal point as a fraction of the render size (in pixels). Default: 0.0.
            - shift_y: Vertical shift of the principal point. Default: 0.0.
            - sensor_width: Horizontal size of the image sensor area in millimeters. Default: 36 mm.
            - Focal Length: Focal length of the camera in millimeters.

    Returns:
        numpy array: xyz map, shape (H, W, 3).
    """
    h, w = depth.shape  # Shape of the depth map as (height, width)
    max_dim = np.max([h, w])  # Size of the largest dimension of the depth map

    # Get the camera parameters
    shift_x = 0
    if "shift_x" in K:
        shift_x = (max_dim * K["shift_x"])

    shift_y = 0
    if "shift_x" in K:
        shift_y = (max_dim * K["shift_y"])

    sensor_width = 36.0  # Size of the image in mm
    if "sensor_width" in K:
        sensor_width = K["sensor_width"]

    f = K["Focal Length"]  # Focal length in mm

    # Create a meshgrid
    grid_y = (list(range(h)) - shift_y - h / 2) * sensor_width / max_dim
    grid_y = np.transpose(np.tile(grid_y, (h, 1)))

    grid_x = (list(range(w)) + shift_x - w / 2) * sensor_width / max_dim
    grid_x = np.tile(grid_x, (w, 1))

    R = np.sqrt(grid_x ** 2 + grid_y ** 2 + f ** 2) + 1e-8  # Radius

    # Compute the xyz map
    xyz = np.zeros([h, w, 3], dtype=np.float32)
    xyz[:, :, 2] = depth * f / R
    xyz[:, :, 1] = depth * grid_y / R
    xyz[:, :, 0] = depth * grid_x / R

    return xyz


def depth2pcloud(depth, K):
    """ Generates the point cloud from given depth map `dmap` using intrinsic
    camera matrix `K` with perspective projection.

    Args:
        depth (np.array): Depth map, shape (H, W).
        K (np.array): Camera intrinsic matrix, shape (3, 3).

    Returns:
        np.array: Point cloud, shape (P, 3), P is # non-zero depth values.
    """
    # Get the indices of non-zero depth values
    y, x = np.where(depth != 0.0)
    N = y.shape[0]
    z = depth[y, x]

    pts_proj = np.vstack((x[None, :], y[None, :], np.ones((1, N))) * z[None, :])
    pcloud = (np.linalg.inv(K) @ pts_proj).T

    return pcloud.astype(np.float32)


def procrustes(x_to, x_from, scaling=False, reflection=False, gentle=True):
    """ Finds Procrustes tf of `x_form` to best match `x_to`.

    Args:
        x_to (np.array): Pcloud to which `x_from` will be aligned. Shape
            (V, 3), V is # vertices.
        x_from (np.array): Pcloud to be aligned. Shape (V, 3), V is # vertices.
        scaling (bool): Whether to use scaling.
        reflection (str): Whether to use reflection.
        gentle (bool): Whether to raise Exception when SVD fails
            (`gentle == False`) or rather to print warning and
            continue with unchanged data (`gentle` == True).

    Returns:
        np.array: Aligned pcloud, shape (V, 3).
    """

    n, m = x_to.shape
    ny, my = x_from.shape

    mu_x = x_to.mean(0)
    mu_y = x_from.mean(0)

    x0 = x_to - mu_x
    y0 = x_from - mu_y

    ss_x = (x0 ** 2.).sum()
    ss_y = (y0 ** 2.).sum()

    # Centred Frobenius norm.
    norm_x = np.sqrt(ss_x)
    norm_y = np.sqrt(ss_y)

    # Scale to equal (unit) norm.
    x0 /= norm_x
    y0 /= norm_y

    if my < m:
        y0 = np.concatenate((y0, np.zeros(n, m - my)), 0)

    # Optimum rotation matrix of Y.
    A = np.dot(x0.T, y0)

    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    except:
        if gentle:
            print('WARNING: SVD failed, returning non-changed data.')
            return x_from
        else:
            raise

    V = Vt.T
    T = np.dot(V, U.T)

    # Undo unintended reflection.
    if not reflection and np.linalg.det(T) < 0:
        V[:, -1] *= -1
        s[-1] *= -1
        T = np.dot(V, U.T)

    trace_TA = s.sum()

    if scaling:
        Z = norm_x * trace_TA * np.dot(y0, T) + mu_x
    else:
        Z = norm_y * np.dot(y0, T) + mu_x

    return Z
