%% set path
global MGP_path;
MGP_path='/Users/litao/本地文稿/research/Mixture of Robust Gaussian Processes/release code'; % root path of this project
GPML_path = '/Users/litao/本地文稿/research/Mixture of Robust Gaussian Processes/gpml-matlab-v4.2-2018-06-11/';
run([GPML_path,'startup.m']);% gpml toolbox
addpath(genpath(MGP_path));

%% synthetic datasets
clear
load('data/synthetic_datasets/S1_outlier_ratio_0.1_outlier_level_2.5.mat');

%%% parameters of the algorithm
opts.alg = 'hard-cut'; 
opts.lik = @likT;% @likLaplace
opts.inf = @infVB; % @infLaplace @infEP 
opts.covfunc = @covSEiso;
opts.meanfunc = @meanZero;
opts.maxiter = 20;
opts.average = true;
opts.verbose = true;
K = 3; % set to 5 if you are using S7-S12
[y_pred,z_test,z_train,RMSE,record] = MGP_learn_and_pred( x_train,y_train,K,x_test,y_test,opts);
 
%%%% Draw predictive function and learned components
draw_pred(x_train,y_train,z_train,mix_para,hyp,record);

%% realworld datasets
clear
load('data/realworld_datasets/Boston.mat');

%%% parameters of the algorithm
opts.alg = 'hard-cut'; 
opts.lik = @likT;% @likLaplace
opts.inf = @infVB; % @infLaplace @infEP 
opts.covfunc = @covSEiso;
opts.meanfunc = @meanZero;
opts.maxiter = 20;
opts.average = true;
opts.verbose = true;

K=3;

[ z_train,record] = MGP_learn(K,x_train,y_train,opts);
record.y_test = y_test;
[y_pred,z_test,RMSE] = MGP_pred(x_train,y_train,z_train,x_test,record);