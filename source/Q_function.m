function [Q ] = Q_function(x,Z,mix_para,Q_)
    K = length(mix_para);
    Q=0; 
    for k=1:K
        if mix_para{k}.mu>eps
            idx =find(Z==k);
            res = x(idx,:)-repmat(mix_para{k}.mu,[length(idx),1]);
            Q = Q + length(idx)*log(mix_para{k}.pi);
            Q = Q - 0.5*log(det(mix_para{k}.sigma2))...
                -0.5*trace(res*(mix_para{k}.sigma2\res'));
        end
    end
    Q = Q-Q_;
end

