# Linear Dynamical Systems in mutiple programming languages
This repository contains the implementation of the sub-optimal parameter learning algorithm of Linear Dynamical Systems in 3 programming languages: Python, C++, and Matlab.

## Optimized parameter learning
Since the LDS parameter learning algorithm for videos (Saisan et al., 2001) can not handle large videos, these implementations are altered by exploiting the well-known covariance trick. This trick, e.g. used in the seminal work on Eigenfaces (Turk and Pentland, 1991), is both memory and time efficient and can therefore handled HD-videos and/or vidoes with a high number of frames.
