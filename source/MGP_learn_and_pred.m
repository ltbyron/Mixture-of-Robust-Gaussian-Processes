function [y_pred,z_test,z_train,RMSE,record] = MGP_learn_and_pred( x_train,y_train,K,x_test,y_test,opts)
    if ~exist('opts','var') opts=struct();end
    opts = MGP_opts(opts);
    
    if exist('y_test','var') && ~isempty(y_test) opts.y_test=y_test;end
    t1= clock();
    [ z_train,record] = MGP_learn(K,x_train,y_train,opts);
    [y_pred,z_test,RMSE] =MGP_pred(x_train,y_train,z_train,x_test,record);
    t2 = clock();
    record.time_total = etime(t2,t1);
    
end

