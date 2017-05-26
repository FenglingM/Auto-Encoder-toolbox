

function y = activefunc(x, type)
    switch type
        case 'sigmoid'
            y = sigmoid(x);
        case 'relu'
            y = relu(x);
        case 'linear'
            y = x;
    end

end