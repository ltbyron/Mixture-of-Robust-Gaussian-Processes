function [Z] = hardcut_Z(x_train,y_train,K,mix_para,hyp,covfunc,lik)
    [N,~]=size(x_train);
    Z = zeros(N,1);
    for i=1:N 
        pz = zeros(K,1);
        for k=1:K
            if abs(mix_para{k}.pi)<1e-16
                pz(k)=0;
            else
                pz(k)=mix_para{k}.pi*mvnpdf(x_train(i,:),mix_para{k}.mu,mix_para{k}.sigma2);
                pz(k)=pz(k)*exp(lik(hyp{k}.lik,y_train(i),0,covfunc(hyp{k}.cov,x_train(i,:)))); 
            end
        end
        [~,Z(i)]=max(pz);
    end
end

