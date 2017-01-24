function net = net(inputs,targets,nodelayers, numEpochs, batchSize, eta)

% randomly assign the weights
weights_cell = cell(1,size(nodelayers,2)-1);
for l = 1:size(nodelayers,2)-1
    weights_cell{l} = randn(nodelayers(l+1),nodelayers(l));
end

% randomly assign the biases
bias_cell = cell(1,size(nodelayers,2)-1);
for l = 1:size(nodelayers,2)-1
    bias_cell{l} = randn(nodelayers(l+1),1);
end

% BEGIN BACKPROPOGATION ALGORITHM

% initiate outer loop for the total number of epochs
epoch = 1;
while epoch <= numEpochs
    % initialize the correct guesses and MSE for the epopch to zero (used for
    % calculating accuracy)
    correct_guesses = 0;
    MSE = 0;
    
    % create an empty cell to hold the activations
    activation_cell = cell(1,size(nodelayers,2));

    % create an empty cell to hold the errors
    errors_cell = cell(1,size(nodelayers,2)-1);

    % create an empty cell to hold the weighted inputs
    z_cell = cell(1,size(nodelayers,2)-1);
    
    % BEGIN RUNNING MINIBATCHES WITHIN THE EPOCH
    
    % initiate the initial minibatch
    minibatch_start = 1;
    minibatch_end = batchSize; 

    while minibatch_end <= size(inputs,2)
        % set the initial activation matrix (one column for each instance in
        % the minibatch)
        a = inputs(:,minibatch_start:minibatch_end);
        activation_cell{1} = a;
        
        % feed forward
        for l = 1:size(nodelayers,2)-1
            % turn the bias vector into a matrix consistent with the number of
            % instances in the minibatch
            bias_matrix = repmat(bias_cell{l},1,batchSize);

            % compute the weighted input for the next layer
            z = weights_cell{l}*a + bias_matrix;
            z_cell{l} = z;

            % compute activation for that layer
            a = logsig(z);
            activation_cell{l+1} = a;
        end
        
        % compute the error at the output layer
        delta_L = (a - targets(:,minibatch_start:minibatch_end)) .* (logsig(z) .* (1 - logsig(z)));
        
        % add the errors at the output layer to the errors cell
        errors_cell{l} = delta_L;
        
        % compute the accuracy
        diff = sum(abs(targets(:,minibatch_start:minibatch_end) - round(a)),1);
            % find all intances where all the guesses are correct
        all_correct = diff == 0;
            % add up all the total correct responses
        correct_guesses = correct_guesses + sum(all_correct);
        
        % calculate the contribution to MSE
        MSE = MSE + norm((targets(:,minibatch_start:minibatch_end) - a)) ^2;
        
        
        % backpropogate the error
        for back_layer = l-1:-1:1
            errors_cell{back_layer} = weights_cell{back_layer+1}' * errors_cell{back_layer+1} .* (logsig(z_cell{back_layer}) .* (1 - logsig(z_cell{back_layer})));
        end

        % gradient descent
        for layer = l:-1:1
            % update weights
            weights_cell{layer} = weights_cell{layer} - (eta / batchSize) * (errors_cell{layer} * activation_cell{layer}');

            % update biases
            bias_cell{layer} = bias_cell{layer} - (eta / batchSize) * sum(errors_cell{layer},2);
        end
        
        % increment the mini batch counters
        minibatch_start = minibatch_start + batchSize;
        minibatch_end = minibatch_end + batchSize;
    end
    
    % define the total instances
    total = size(targets,2);
    
    % compute the MSE
    MSE = MSE / (2 * total);
    
    % write the results
    fprintf('Epoch %i, MSE: %f, Correct: %i / %i  Acc: %f\n',epoch, MSE, correct_guesses,total,correct_guesses/total)
    
    % check for termination criteria
    if correct_guesses == total 
        break
    end
    
    % increase the epoch counter
    epoch = epoch + 1;
end 