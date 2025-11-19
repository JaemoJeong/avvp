# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.
import numpy as np

def generate_combined_transform_function(transform_funcs, indices=[0]):
    """
    Create a composite transformation function by composing transformation functions

    Parameters:
        transform_funcs
            list of transformation functions

        indices
            list of indices corresponding to the transform_funcs
            the function is composed by applying 
            function indices[0] -> function indices[1] -> ...
            i.e. f(x) = f3(f2(f1(x)))

    Returns:
        combined_transform_func
            a composite transformation function
    """

    for index in indices:
        print(transform_funcs[index])
    def combined_transform_func(sample):
        for index in indices:
            sample = transform_funcs[index](sample)
        return sample
    return combined_transform_func

def noise_transform_vectorized(X, sigma=0.1):
    """
    Adding random Gaussian noise with mean 0
    """
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

def scaling_transform_vectorized(X, sigma=0.25):
    """
    Scaling by a random factor
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
    return X * scaling_factor

def rotation_transform_vectorized(X):
    """
    Applying a random 3D rotation
    """
    # switch last 2 axes of X
    X = np.swapaxes(X, 1, 2)
    
    axes = np.random.uniform(low=-1, high=1, size=(X.shape[0], X.shape[2]))
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(X.shape[0]))
    matrices = axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)
    acc = np.matmul(X[:,:,0:3], matrices)
    gyr = np.matmul(X[:,:,3:6], matrices)
    X = np.concatenate([acc, gyr], axis=-1)

    X = np.swapaxes(X, 1, 2)
    return X

def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    """
    Get the rotational matrix corresponding to a rotation of (angle) radian around the axes

    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    Formula: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    m = np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])
    matrix_transposed = np.transpose(m, axes=(2,0,1))
    return matrix_transposed

def negate_transform_vectorized(X):
    """
    Inverting the signals
    """
    return X * -1

def time_flip_transform_vectorized(X): # Modified to work with shape (batch, time, channels)!!
    """
    Reversing the direction of time
    """
    return X[:,:,::-1] #X[:, ::-1, :]

def time_segment_permutation_transform_improved(X, num_segments=4): # Modified to work with shape (batch, time, channels)!!
    """
    Randomly scrambling sections of the signal
    """
    print(X.shape, num_segments)
    segment_points_permuted = np.random.choice(X.shape[-1], size=(X.shape[0], num_segments), replace=False)
    segment_points = np.sort(segment_points_permuted, axis=1)
    X_transformed = np.empty(shape=X.shape)
    for i, (sample, segments) in enumerate(zip(X, segment_points)): 
        splitted = np.split(sample, segments, axis=-1)
        np.random.shuffle(splitted)
        concat = np.concatenate(splitted, axis=-1)
        X_transformed[i] = concat

    return X_transformed