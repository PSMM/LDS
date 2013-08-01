function [A,C,X] = lds_video(videofile, n, f)
% Perform optimized LDS on a single video.
%
% Written by: Pascal Mettes.
%
%This file contains the Matlab implementation of a sub-optimal algorithm for
% parameter learning in Linear Dynamical Systems (LDS). This implementation is
% based on the paper by Saisan et al. [Saisan'01] on dynamic texture recognition
% from video using LDS.
%
% This implementation contains both the implementation of the formulation by
% Saisan et al., as well as an optimized version using the well-known covariance
% trick. For more info on the optimization, please refer to my Master thesis.

% Load the video.
vobj   = mmreader(videofile);
width  = get(vobj, 'Width');
height = get(vobj, 'Height');

% Initialize the video matrix, where the frames become column vectors.
Y = zeros(width * height, f);
% Yield the frames and add to the video matrix
for i=1:f,
    frame     = rgb2gray(read(vobj, i));
    Y(:,i) = reshape(frame, size(frame,1) * size(frame,2), 1);
end

% Subtract the mean from the video matrix/
Ymean = mean(Y, 2);
NY    = Y - repmat(Ymean, 1, f);

% Perform SVD on covariance matrix.
NYT     = NY' * NY;
[U,S,V] = svd(NYT);

% Compute primary LDS parameters
X = sqrt(S(1:n,1:n)) * V(:,1:n)';
C = (NY * V(:,1:n)) / sqrt(S(1:n,1:n));
A = X(:,2:f) * pinv(X(:,1:(f-1)));