function [loglikelihood] = MGP_loglikelihood(x,y,z,hyp,inf,meanfunc,covfunc,lik)
    K = length(hyp);
    loglikelihood = 0;
    for k=1:K
        idx = (z==k);
        loglikelihood = loglikelihood - gp(hyp{k},inf,meanfunc,covfunc,lik,x(idx,:),y(idx));
    end
end

