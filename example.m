% Written by Emtiyaz, EPFL
% Modified on March 8, 2014
clear all;
close all;

% synthetic data
setSeed(1);

%dataset_name = 'dataset1';
%N = 1000; % number of data examples
%D = 20; % feature dimensionality

%dataset_name = 'dataset2';
%N = 500; % number of data examples
%D = 10; % feature dimensionality

%dataset_name = 'dataset3';
%N = 800; % number of data examples
%D = 40; % feature dimensionality

%dataset_name = 'dataset4';
%N = 200; % number of data examples
%D = 100; % feature dimensionality

%dataset_name = 'dataset5';
%N = 100; % number of data examples
%D = 5; % feature dimensionality

dataset_name = 'dataset6';
N = 20; % number of data examples
D = 5; % feature dimensionality



X = [5*rand(N/2,D); -5*rand(N/2,D)]; 
Sigma = X*X' + eye(N); % linear kernel
mu = zeros(N,1); % zero mean
y = mvnrnd(mu, Sigma, 1);
y = (y(:)>0);

x_file = sprintf('X_%s',dataset_name);
fileID = fopen(x_file,'w');
for c = 1:size(X,2)
	for r = 1:size(X,1)
		fprintf(fileID,'%.10f\t',X(r,c));
	end
	fprintf(fileID,'\n');
end
fclose(fileID);

y_file = sprintf('y_%s',dataset_name);
fileID = fopen(y_file,'w');
fprintf(fileID,'%.10f\n',y);
fclose(fileID);

% optimizers options
optMinFunc = struct('Display', 1,...
    'Method', 'lbfgs',...
    'DerivativeCheck', 'off',...
    'LS', 2,...
    'MaxIter', 1000,...
    'MaxFunEvals', 1000,...
    'TolFun', 1e-4,......
    'TolX', 1e-4);

% load bound
load('llp.mat'); 
bound_file = sprintf('bounds');
fileID = fopen(bound_file,'w');
for r = 1:size(bound,1)
	for c = 1:size(bound,2)
		fprintf(fileID,'%.10f\t',bound(r,c));
	end
	fprintf(fileID,'\n');
end
fclose(fileID);

% optimize wrt m (see function simpleVariational.m for details)
m0 = mu; % initial value all zero
v = ones(N,1); % fix v to 1
Omega = inv(Sigma);
[m, logLik] = minFunc(@simpleVariational, m0, optMinFunc, y, X, mu, Omega, v, bound);

m_file = sprintf('m_%s',dataset_name);
fileID = fopen(m_file,'w');
fprintf(fileID,'%.10f\n',m);
fclose(fileID);

logLik_file = sprintf('logLik_%s',dataset_name);
fileID = fopen(logLik_file,'w');
fprintf(fileID,'%.10f\n',logLik);
fclose(fileID);

% plot
%figure(1)
%imagesc(Sigma); colorbar;
%title('GP Kernel matrix');

%figure(2)
%stem(y);
%hold on
%plot(1./(1+exp(-m)), '*r','markersize', 10);
%ylim([-0.05 1.05]);
%ylabel('Prediction for training data');

