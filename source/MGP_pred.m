function  [y_pred,z_test,RMSE] =MGP_pred(x_train,y_train,z_train,x_test,record)
    t1 = clock();
    mix_para = record.mix_para;
    hyp = record.hyp;
    inf = record.inf;
    meanfunc = record.meanfunc;
    covfunc = record.covfunc;
    lik = record.lik;
    average = record.average;
    if isfield(record,'y_test') y_test=record.y_test;end
    verbose = record.verbose;
    K = length(unique(z_train));
    N = length(x_test);     
     
    [mu,cov_,pi]=mix_para2gm(mix_para);
    GM = gmdistribution(mu,cov_,pi);
    [z_test,~,alpha] =cluster(GM,x_test);
    
    
    if ~average
        alpha = double(alpha == repmat(max(alpha,[],2),[1,K]));
    end
   
    y_pred = zeros(length(x_test),K);
    if average
        for k=1:K
            idx = find(z_train==k);
            y_pred(:,k) =y_pred(:,k)+gp(hyp{k},inf,meanfunc,covfunc,lik,x_train(idx,:),y_train(idx),x_test);
        end
    else
        for k=1:K
            idx1 = find(z_train==k); idx2=find(z_test==k);
            y_pred(idx2,k)=y_pred(idx2,k)+gp(hyp{k},inf,meanfunc,covfunc,lik,x_train(idx1,:),y_train(idx1),x_test(idx2,:));
        end
    end
    y_pred = sum(alpha.*y_pred,2);
   
    
    if exist('y_test','var')
        RMSE = sqrt(mse(y_test,y_pred));
        if verbose fprintf('RMSE:%.4f\n',RMSE);end
    else
        RMSE=[];
    end
    
    t2=clock();
    record.time_test = etime(t2,t1);

end

function [mu,cov_,pi]=mix_para2gm(mix_para)
    K=length(mix_para);D=length(mix_para{1}.mu);
    mu = zeros(K,D);
    pi = zeros(1,K);
    cov_=zeros(D,D,K);
    for k=1:K
        mu(k,:)=mix_para{k}.mu;
        pi(k) = mix_para{k}.pi;
        cov_(:,:,k)  = mix_para{k}.sigma2;
    end
end