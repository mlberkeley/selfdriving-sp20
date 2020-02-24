import torch
import torch.nn as nn
import numpy as np
import os

# Fully Connected Network Class (a custom subclass of torch's nn module)
class fcn(nn.Module):

    '''
    Initialize Fully Connected Network.  Model will take in num_inputs inputs and
    output a single number.  This number will be unbounded by default but if an out_range
    is specified, (a, b), then the output will be within this interval.

    The model_name parameter is also the file path to the saved weights from the working
    directory.  To load a model in a different file make sure the name accounts for the
    extended filepath.
    
    By default we attempt to load previously trained weights.  If for whatever reason
    one would like to retrain the model from scratch the retain parameter may be used.
    '''
    def __init__(self, model_name, num_inputs, out_range = None, retrain = False):
        super().__init__()

        #Set the weights file path as the model name
        self.weights_fp = model_name + ".pt"
        #out range will should either be None or of the form (a, b)
        self.out_range = out_range

        # Number of hidden units in first hidden layer
        self.H_1 = 30
        # Number of hidden units in second hidden layer
        self.H_2 = 20
        
        '''
        Weights generally [input dim, output dim] so when we multiply a matrix
        of size [batch size, input dim] by a matrix of size [input dim, output dim] 
        we get a matrix of size [batch size, output dim].  The bias will have 
        shape [output dim] so we can add it to the result of the matrix multiplication.
        '''
        
        #Weights and Biases for computing input -> first hidden layer
        self.W_1 = nn.Parameter(torch.randn([num_inputs, self.H_1]))
        self.B_1 = nn.Parameter(torch.randn([self.H_1]))

        #Weights and Biases for computing first -> second hidden layer
        self.W_2 = nn.Parameter(torch.randn([self.H_1, self.H_2]))
        self.B_2 = nn.Parameter(torch.randn([self.H_2]))
        
        #Weights and Biases for computing second hidden layer -> output
        self.W_3 = nn.Parameter(torch.randn([self.H_2, 1]))
        self.B_3 = nn.Parameter(torch.randn([1]))

        '''
        Adam is a gradient descent based algorithm that incorporates a couple
        other nice features which encourage fast, robust model performance.

        For one, Adam is robust to noisy gradient as it updates according to
        a exponential moving average.  Further, it recognizes the relative
        magnitudes of gradients an makes updates accordingly (i.e. a parameter that
        consistenly has relatively low loss gradient will be assigned a greater
        leraning rate than one that consistently has greater loss gradients.)
        '''
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)

        #Load previously saved weights if not retraining.
        if (not retrain):
            self.load_saved_weights()

    '''
    Forward propogation algorithm. Note calling model(inp) is equivalent to calling
    model.forward(x = inp).  
    The additional clamp parameter is only relevant if the output range is specified.
    In this case while training we would like to use unbounded output to allow the gradient
    to exists everywhere (clamp = False), but during usage we want to cutoff the output whenever
    it exceeds either boundary (clamp = True).
    '''
    def forward(self, x, clamp = True):
        
        # x will be a matrix of dimension [batch size, num_inputs] (note batch size can vary during usage).
        x = torch.tensor(x, dtype = torch.float32)

        # first hidden layer computation with tanh activation. [batch size, num_inputs] -> [batch size, H_1]
        h_1 = torch.tanh(torch.matmul(x, self.W_1) + self.B_1)
    
        # second hidden layer computation with tanh activation. [batch size, H_1] -> [batch size, H_2]
        h_2 = torch.tanh(torch.matmul(h_1, self.W_2) + self.B_2)
        
        #output computation with no activation.  [batch size, H_2] - (MatMul) > [batch size, 1] - (Squeeze) > [batch size]
        out = torch.squeeze(torch.matmul(h_2, self.W_3) + self.B_3)

        '''
        If the output has no specified range or it does but we are not clamping the output
        right now (generally during training), then we simply return the unmodified out

        If the output range is specified and we are clamping then we clamp the input to be
        between out_range[0] and out_range[1].  For example a very large number would be
        mapped to out_range[1].
        '''
        if (self.out_range is None or not clamp):
            return out
        else:
            return torch.clamp(out, self.out_range[0], self.out_range[1])

    '''
    Trains the Fully Connected Network using the provided data for
    num_epochs epochs using a parameter batch size (default 10).

    input_data: A matrix of dimensions [NUM DATA POINTS, num_inputs]
    targets: A vector of dimension [NUM DATA POINTS] (since singular output)
    '''
    def train(self, input_data, targets, num_epochs, batch_size = 10):
        #Number of data points
        NUM_DATA = len(input_data)

        #Number of data batches for training purposes
        NUM_BATCHES = int(NUM_DATA / batch_size)

        #Converts input data and targets to np arrays for easy indexing
        input_data = np.array(input_data)
        targets = np.array(targets)


        #Train with all data once num_epochs times.
        for e in range(num_epochs):

            '''
            In case data was generated in some non-random fashion, we randomize
            the order in which we process the data points.  This prevents, for
            example, the model optimizing for a certain situation that is more
            prevalent later in the data set than earlier or vice versa.
            '''

            #Numbers 0 to NUM_DATA - 1 (data point indices)
            random_indices = np.arange(NUM_DATA)
            #Shuffles indices in place
            np.random.shuffle(random_indices)

            #Reorders the input data and targets in a consistent manner.
            input_data = input_data[random_indices]
            targets = targets[random_indices]

            #Epoch Loss
            e_loss = 0

            #Go through each of the batches and make updates
            for b in range(NUM_BATCHES):

                #Get the appropriate batch_size inputs and targets.
                inp = input_data[b * batch_size:(b + 1) * batch_size]
                targ = targets[b * batch_size:(b + 1) * batch_size]

                '''
                self(inp, clamp = False) runs self.forward(x = inp, clamp = False)
                This gets the output of the current model for the given inputs.

                We don't clamp so that the gradient exists everywhere (y = x, dy/dx = 1)
                If we did clamp, whenever the model's output was out of the desired range
                we would have 0 gradient and gradient would not know which way to move in.
                '''
                out = self(inp, clamp = False)

                '''
                Batch Loss = Mean Squared Error.  Convert the targets into a torch tensor
                so the calculated MSE remains a valid tensor.  This allows us to use torchs
                automatic gradient calculation features and thus we know how to adjust the
                model parameters to minimize the loss.
                '''
                b_loss = torch.mean((torch.tensor(targ, dtype = torch.float32) - out) ** 2)

            
                #Zero the current gradients so we don't accumulate the previous gradients.
                self.optimizer.zero_grad()
                #In a single backward pass populate d(batch loss)/d(parameter) for each parameter.
                b_loss.backward()
                #Make a gradient descent update.  Note this is uses the Adam procedure from above.
                self.optimizer.step()

                #Now that we are done with torch's automatic gradient we can convert to a normal number.
                b_loss = b_loss.detach().numpy()
                #Increment the epoch loss
                e_loss += b_loss

            #Normalization to give us MSE over the whole dataset.
            e_loss /= NUM_BATCHES
            print(f"EPOCH {e + 1} {e_loss:.6f}", end = "\r")

            '''
            This print methodology allows us to constantly get updates for the epoch loss, and jump to
            a new line 5 times over the full training procedure, so we can monitor model progress.
            '''
            if (e % int(num_epochs / 5) == 0):
                print()

            #Save the weights after training in the file path that corresponds to the model name.
            torch.save(self.state_dict(), open(self.weights_fp, "wb"))

    '''
    Run forward propogation on the data to get predictions.
    Then detach to remove automatic gradient dependencies.
    Then convert to numpy from torch tensor and return
    '''
    def predict(self, x):
        return self(x).detach().numpy()

    '''
    Attempts to load saved weights from the appropriate file path.  If the weights file is not at
    the specified location then random weights will be used to initialize the model and a warning
    will be issued.
    '''
    def load_saved_weights(self):
        if (os.path.exists(self.weights_fp)):
            self.load_state_dict(torch.load(open(self.weights_fp, "rb")))
        else:
            print("WARNING: Weights file not found, initializing fcn with random weights.")
