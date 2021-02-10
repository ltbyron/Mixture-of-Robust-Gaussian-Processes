function [ hyp] = hyp_init( mean,cov,lik,D,func)
    if ~exist('func','var') func =@zeros;end    
    
    if ~iscell(mean) mean={mean};end
    hyp.mean = func(eval(feval(mean{:})),1);
    if iscell(cov)
        hyp.cov  = func(eval(feval(cov{:})),1);
    else
        hyp.cov  = func(eval(feval(cov)),1);
    end
    %hyp.cov  = [-1;.5]+func(eval(feval(cov)),1);
    hyp.lik  = func(eval(feval(lik)),1);

end

