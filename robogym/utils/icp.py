# Copy from https://github.com/ClayFlannigan/icp/blob/master/icp.py
# ICP: Iterative Closest Point which is an algorithm to find optimal rotation
# matrix between two set of point cloud. This file implements vanilla ICP using
# Kabsch algorithm with nearest neighbor matching.
# See https://en.wikipedia.org/wiki/Iterative_closest_point for more variants of
# ICP algorithms


import numpy as np
from sklearn.neighbors import NearestNeighbors

from robogym.utils.mesh import get_vertices_bounding_box


class ICP:
    def __init__(self, target_points: np.ndarray, error_threshold: float):
        """
        :param target_points: The target point cloud to match against.
        :param error_threshold: The error threshold to trust ICP result. This is relative
            to bounding box size of target point cloud.
        """
        self.error_threshold = (
            error_threshold * get_vertices_bounding_box(target_points)[-1]
        )
        self.target_points = target_points
        self.knn = None

    def compute(self, points):
        """
        Compute optimal rotation matrix. None if error is above threshold.
        """
        if self.knn is None:
            self.knn = NearestNeighbors(n_neighbors=1)
            self.knn.fit(self.target_points)

        T, max_error = icp(
            points, self.target_points, self.knn, max_iterations=5, tolerance=1e-6,
        )

        if max_error < self.error_threshold:
            return T[:3, :3].T
        else:
            return None


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, neigh):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, knn, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    max_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, knn)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        max_error = np.max(distances)

        if np.abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, max_error
