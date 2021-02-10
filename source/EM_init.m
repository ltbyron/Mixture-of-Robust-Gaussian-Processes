function [z] = EM_init(x,K)
    z = kmeans(x,K);
end

