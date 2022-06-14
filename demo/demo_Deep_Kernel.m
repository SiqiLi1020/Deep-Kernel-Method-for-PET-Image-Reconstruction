% This is a MATLAB demo showing how to use the deep kernel method for 
% high-temporal resolution dynamic PET image reconstruction. The method is 
% described in 
%
%   S. Q.  Li, and G. B. Wang, Deep Kernel Representation for Image Reconstruction in PET. 
%   IEEE Transactions on Medical Imaging, accepted, May 2022.
%
% 
%
% Siqi Li and Guobao Wang @ UC Davis Medical Center (5-26-2022)

clear; clc;

%% add path
run('D:/Guobao_work/fessler/irt/setup');  
run('D:/Guobao_work/KER_v0.2/setup');

%% simulate PET data

% load true emission and atteunation images
imgsiz = [111 111];
load('htrzubal_data');

% system matrix G 
sysflag = 1;  % 1: using Fessler's IRT; 
              % 0: using your own system matrix G 
disp('--- Generating system matrix G ...')
if sysflag
    % require Fessler's IRT matlab toolbox to generate a system matrix
    ig = image_geom('nx', imgsiz(1), 'ny', imgsiz(2), 'fov', 33.3);

    % field of view
    ig.mask = ig.circ(16, 16) > 0;

    % system matrix G
    prjsiz = [249 210];
    sg = sino_geom('par', 'nb', prjsiz(1), 'na', prjsiz(2), 'dr', 70/prjsiz(1), 'strip_width', 'dr');
    G  = Gtomo2_strip(sg, ig, 'single', 1);
    Gopt.mtype  = 'fessler';
    Gopt.ig     = ig;
    Gopt.imgsiz = imgsiz;
    Gopt.prjsiz = prjsiz;
    Gopt.disp   = 0; % no display of iterations
else
    Gfold = ''; % where you store your system matrix G;
    load([Gfold, 'G']);	% load matrix G
    Gopt.mtype  = 'matlab';
    Gopt.imgsiz = imgsiz;
end

% noise-free projection data
dt = scant(:,2)-scant(:,1); % scan duration of dynamic frames
cc = decaycoef(scant/60, log(2)/109.8); % decay coefficients
nt = repmat(dt'./cc',[prod(Gopt.prjsiz),1]); % multiplicative factor
for m = 1:size(xt,2) % frame-wise
    proj    = nt(:,m).*proj_forw(G, Gopt, xt(:,m)); % noise-free geometric projection
    at(:,m) = exp(- proj_forw(G, Gopt, ut(:,m)) ); % attenuation
    rt(:,m) = repmat(mean(at(:,m).*proj,1)*0.2,[size(proj,1) 1]); % 20% mean background (scatter and randoms)
    y0(:,m) = at(:,m).*proj + rt(:,m); % total noise-free projection
end

% count level
count = 20e6; % a total of 20 million events

% normalized sinograms
cs = count / sum(y0(:));
y0 = cs * y0;
rt = cs * rt;
nt = cs * (nt.*at); % final normalization factor

% noisy realization
load ('../proj/proj20');


%% Conventional MLEM reconstruction (i.e. K = I)
disp('--- Doing conventional MLEM reconstruction ...')

% maximum iteration number
maxit = 100;

% number of frames
numfrm = size(yt,2);

% select frames to reconstruct
mm = 1:numfrm; 

% MLEM of noisy data
tic;
for m = 1:numfrm
    [x, out] = eml_kem(yt(:,m), nt(:,m), G, Gopt, [], rt(:,m), maxit);
    for it = 1:size(out.xest,2)
        xi = out.xest(:,it);
        X{1}(:,m,it) = xi;
    end
end
toc;

%% Conventional Kernel Method (KEM) 

disp('--- Doing conventional kernel reconstruction (KEM) ...')

% Constructing composite image prior (CIP) for the spatial kernel
M = {[1:48], [49:53], [54:58], [59:63]};
imgsiz = Gopt.imgsiz;
for m = 1:length(M)
    y_m = sum(yt(:,M{m}),2);
    n_m = sum(nt(:,M{m}),2);
    r_m = sum(rt(:,M{m}),2);
    [x, out] = eml_kem(y_m, n_m, G, Gopt, [], r_m, 100); 
    x = gaussfilt(x,imgsiz,3);
    U(:,m) = x(:);
end
U = U * diag(1./std(U,1)); % normalization

% spatial kernel by KNN
sigma = 1;
[N, W] = buildKernel(imgsiz, 'knn', 48, U, 'radial', sigma);
Ks = buildSparseK(N, W);

% temporal kernel Kt = I;
Kt = speye(numfrm);

% reconstruction of kernel coefficient a
tic;
[a, out] = eml_kem4D(yt(:,mm), nt(:,mm), G, Gopt, [], rt(:,mm), maxit, Ks, Kt); 
toc;

% converted to activity image x
for it = 1:size(out.xest,3)
    ai = out.xest(:,:,it);
    X{2}(:,:,it) = Ks * ai * Kt'; 
end

%% Proposed Deep Kernel Method  

disp('--- Doing deep kernel reconstruction ...')

% Build kernel matrix obtained from deep kernel model 
load ('../training data/index_300_20e6_n_20.mat'); % load neighbor index
iter = 300; % choose deep kernel training iteration 
load (sprintf('../trained models/W_%diter.mat',iter)); % load pairwise weight matrix W from deep kernel method
W = double(W);
Ks = buildSparseK(N, W, [], [], 0); % reshape W as kernel matrix K using neighbor index

% temporal kernel Kt = I;
Kt = speye(numfrm);

% reconstruction of kernel coefficient a
tic;
[a, out] = eml_kem4D(yt(:,mm), nt(:,mm), G, Gopt, [], rt(:,mm), maxit, Ks, Kt); 
toc;

% converted to activity image x
for it = 1:size(out.xest,3)
    ai = out.xest(:,:,it);
    X{3}(:,:,it) = Ks * ai * Kt'; 
end


%% Comparison for image quality

disp('--- Comparing different reconstruction methods ...')

% Algorithms to compre
algs = {'ML-EM','Conventional kernel','Proposed deep kernel'};

% the iteration number to show results
it = 10+1;

% select a frame to compare image quality
M = 15; % frame 15

% image display and MSE (in dB)
figure;
mask = model>=0;
for k = 1:length(algs)
    Xk = X{k};
    for i = 1:size(Xk,3)
        for m = 1:size(Xk,2)
            xi = Xk(mask,m,i);
            x0 = xt(:,m);
            MSE{k}(i,m) = 10*log10(sum((xi-x0(mask)).^2)/sum(x0(mask).^2));
        end
    end
   
    maxval = 1.2e4;
    subplot(2,2,k); imagesc(reshape(Xk(:,M,it),imgsiz),[0 maxval]); axis image;
    set(gca,'FontSize',10); axis off; colormap(hot); % colorbar;
    title(sprintf('%s, %3.2fdB',algs{k}, MSE{k}(it,M))); 
end
% show true image
subplot(2,2,4); imagesc(reshape(xt(:,M),imgsiz),[0 maxval]); axis image;
set(gca,'FontSize',10); axis off; colormap(hot); % colorbar;
title(sprintf('True image')); 
print('')
 

