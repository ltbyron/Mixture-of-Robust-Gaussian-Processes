function [ opts] = MGP_opts(opts)
    if ~exist('opts','var') opts=struct();end
    
    %%% Choose 'MCMC' 'hard-cut' or 'v-hard-cut','MWMC'
    IMPLEMENTED = {'hard-cut'};
    if ~isfield(opts,'alg') opts.alg='hard-cut';end
    if ~any(strcmp(IMPLEMENTED,opts.alg))
        error('Algorithm not implemented');
    end
    
    %%% Gaussian Process options
    if ~isfield(opts,'inf') opts.inf=@infGaussLik;end
    if ~isfield(opts,'meanfunc') opts.meanfunc=@meanZero;end
    if ~isfield(opts,'covfunc') opts.covfunc=@covSEard;end
    if ~isfield(opts,'lik') opts.lik=@likGauss;end
    
    %%% maximum number of EM iterations
    if ~isfield(opts,'maxiter') opts.maxiter=10;end
    
    %%% Use average or single expert to predict
    if ~isfield(opts,'average') opts.average=true;end
    if ~isfield(opts,'verbose') opts.verbose=true;end
    
end

