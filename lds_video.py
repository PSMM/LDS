#
# lds_video.py.
#
# Written by: Pascal Mettes.
#
# This file contains the Python implementation of a sub-optimal algorithm for
# parameter learning in Linear Dynamical Systems (LDS). This implementation is
# based on the paper by Saisan et al. [Saisan'01] on dynamic texture recognition
# from video using LDS.
#
# This implementation contains both the implementation of the formulation by
# Saisan et al., as well as an optimized version using the well-known covariance
# trick. For more info on the optimization, please refer to my Master thesis.
#

import math
import numpy as np
from numpy.linalg import pinv, svd, eig, inv
import cv
import sys
from matplotlib.pylab import *

#
# Learn the main parameters of a Linear Dynamical System (LDS) [Saisan'01].
#
# Input : The videofile (string), number of latent components (int), number of
#         frames (int), and scale factor (float).
# Output: The main LDS parameters, resp. A,C,X (all 2D numpy arrays).
#
def learn_lds(videofile, n, f, scale):
    # Open the video file.
    capture = cv.CaptureFromFile(sys.argv[1])
    nr_frames = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
    frames_used = min(f, nr_frames)

    # Initialize the observability matrix.
    Y = np.zeros((int((800 * scale) * (600 * scale)), frames_used))
    # Retrieve the frames, scale them, and add them to the matrix.
    for i in xrange(frames_used):
        frame  = cv.QueryFrame(capture)
     
        gframe = cv.CreateImage(cv.GetSize(frame), 8, 1)
        cv.CvtColor(frame, gframe, cv.CV_BGR2GRAY)
        tframe = cv.CreateImage((int(800 * scale), int(600 * scale)), 8, 1)
        cv.Resize(gframe, tframe)

        Y[:,i] = np.asarray(tframe[:,:]).ravel()

    # Compute and subtract the mean.
    Ymean = Y.mean(axis=1)
    NY = Y - np.tile(Ymean, (frames_used, 1)).T

    # Traditional procedure. Uncomment this part and comment out the next part
    # to use this approach.
    #U,S,V = svd(NY)
    #C = U[:,0:n]
    #X = np.dot(np.diag(S[0:n]), V[0:n,:])
    #A = np.dot(X[:,1:frames_used], pinv(X[:,0:(frames_used-1)]))

    # Optimized LDS parameter learning using the covariance trick.
    U,S,V = svd(np.dot(NY.T, NY))
    X = np.dot(np.diag(np.sqrt(S[0:n])), V[0:n,:])
    C = np.dot(np.dot(NY, V.T[:,0:n]), inv(np.diag(np.sqrt(S[0:n]))))
    A = np.dot(X[:,1:frames_used], pinv(X[:,0:(frames_used-1)]))
    
    return A,C,X

#
# Entry point of the application when run directly.
#
if __name__ == "__main__":
    videofile = sys.argv[1]
    # Hard-codec values for #latent states, #frames, and scale factor.
    n, f, scale = 20, 150, 0.2
    
    # Learn the parameters.
    A,C,X = learn_lds(videofile, n, f, scale)
    
    # Show the latent mappings as a matrix.
    matshow(X)
    show()
