function [h] = draw_pred(x_train,y_train,z_train,mix_para,hyp,record)
    % mainly for visualization of posterior predictive function
    % x_hat is generated uniformly and calculated in this function
    color_list = [166,86,40;
                55,126,184;
                77,175,74;
                255,127,0;
                152,78,163;]/255;
            
    K = max(z_train);
    %K = record.K;
    x_hat = linspace(min(x_train),max(x_train),1000)';
    if isempty(mix_para)
        y_hat = gp(hyp,record.inf,record.meanfunc,record.covfunc,record.lik,x_train,y_train,x_hat);
        z_hat = ones(size(x_hat));
    else
        if isfield(record,'z_test') record=rmfield(record,'z_test');end
        if isfield(record,'y_test') record=rmfield(record,'y_test');end
        [y_hat,z_hat] = MGP_pred(x_train,y_train,z_train,x_hat,record);
    end
    
%     figure
    hold on
    for k=1:K
        idx1 = (z_train==k);
        color = color_list(mod(k-1,5)+1,:);
        plot(x_train(idx1),y_train(idx1),'.','MarkerSize',10,'Color',color);
        
        %idx2=(z_test==k);
        %plot(x_test(idx2),y_test(idx1),'.','MarkerSize',10,'Color',color_list(mod(k-1,5)+1,:));
        
        if isfield(record,'color') color=record.color;end
        idx3=(z_hat==k);
        if k==1
            h=plot(x_hat(idx3),y_hat(idx3),'Color',color,'LineWidth',2);
        else
            plot(x_hat(idx3),y_hat(idx3),'Color',color,'LineWidth',2);
        end
    end
    xlabel('Input');
    ylabel('Output');
    %plot(x_hat,y_hat);
    
end

