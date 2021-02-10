function [ z,record] = MGP_learn(K,x_train,y_train,opts)
    t1 = clock;
    if ~exist('opts','var') opts=struct();end
    opts = MGP_opts(opts);
    
    record=opts;
    record.K =K;
    
    alg = opts.alg;
    inf = opts.inf;
    meanfunc = opts.meanfunc;
    covfunc = opts.covfunc;
    lik = opts.lik;
    maxiter=opts.maxiter;
    num_opt_iter = -100;
    verbose = opts.verbose;
    
    [N,D] = size(x_train); % number of training samples
    
    % mix_para{k} has three fields: pi,mu,sigma2
    % initialized by k-means
    if verbose fprintf('Initializing...\n');end
    Z = EM_init(x_train,K);
    W_old = -2.5*ones(K,N);
    z_old=Z;Q_old=0;
    
    mix_para = cell(K,1);hyp=cell(K,1);u=cell(K,1);
    
    record.Q = [];
    for iter=1:maxiter
        %%% M-step: updating parameters based on Z
        if iter>1
            mix_para_old = mix_para;
        end
        
        Q_ =0;Q_old=0;
        for k=1:K
            if verbose fprintf('Iter:%d M-step update expert %d\n',iter,k);end
            % Update mixture parameters pi,mu,sigma
            idx = find(Z==k);
                mix_para{k}.pi = length(idx)/N;
                if length(idx)>1
                    mix_para{k}.mu = mean(x_train(idx,:));
                    mix_para{k}.sigma2=cov(x_train(idx,:))+1e-6*eye(length(mix_para{k}.mu));
                else
                    mix_para{k}.mu = x_train(idx,:);
                    mix_para{k}.sigma2=1e-6*eye(length(mix_para{k}.mu));
                end

                % Update GP parameters
                if iter ==1, hyp{k}=hyp_init(meanfunc,covfunc,lik,D);end
                [hyp{k},fvalue] = minimize(hyp{k},@gp,num_opt_iter,inf,meanfunc,covfunc,lik,x_train(idx,:),y_train(idx));
                Q_old = Q_old+fvalue(1);
                Q_ = Q_+fvalue(end);
        end
        
        % Q-function calculation
        Q  = Q_function(x_train,Z,mix_para,Q_);
        record.Q = [record.Q;Q];
        if iter>1 Q_old  = Q_function(x_train,Z,mix_para_old,Q_old);end
        relative_change = abs(Q-Q_old)/(eps+abs(Q_old));
        if verbose, fprintf('Q-function value:%.4f-------------->%.4f relative change:%.6f\n',Q_old,Q,relative_change);end
        if iter>1 && relative_change<1e-4
            break
        end
        
        %%% E-step
        if verbose, fprintf('Iter:%d E-step\n',iter);end
        Z = hardcut_Z(x_train,y_train,K,mix_para,hyp,covfunc,lik);
        %%%% Remove empty components
        k=1;
        while k<=K
            idx = find(Z==k);
            if isempty(idx)
               mix_para(k)=[];hyp(k)=[];
               K =K-1;
               Z(Z>=k)=Z(Z>=k)-1;Z(Z>=k)=Z(Z>=k)-1;
            else
                k=k+1;
            end
        end
        
        z = mode(Z,2);
        if verbose fprintf('number of components:%d\n',length(unique(Z)));end
        if verbose fprintf('%d ',sum(repmat(Z,[1,K])==repmat(1:K,[numel(Z),1])));fprintf('\n');end
        if verbose fprintf('%.2f ',sum(repmat(Z,[1,K])==repmat(1:K,[numel(Z),1]))/numel(Z));end
        vz = sum(z~=z_old);
        if verbose fprintf('\nvariation in z:%d\n',vz);end
        z_old=z;
    end
    
    record.u=u;
    record.mix_para=mix_para;
    record.hyp=hyp;
    
    t2=clock();
    record.time_train = etime(t2,t1);
end

