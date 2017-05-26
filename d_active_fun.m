function y = d_active_fun(x , type)
% input is net.layers{tt}.a, which has passed the activation function.
switch type
    case 'relu'
        y = (x>0 );
    case 'sigmoid'
        y = x .* (1-x);
    case 'linear'
        y =ones(size(x));
        
end


end

