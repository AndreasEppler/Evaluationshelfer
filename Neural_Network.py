import math
import os
from datetime import datetime
import numpy as np
import random as rd

class NeuralNetwork:
    """
    this class models a sequential neural network model
    up to this version, it is only possible to create fully connected (dense) models
    """
    def __init__ (self):
        self._layers = []
        self._saving_loc = "Saved Models"
        # self._cost = Cost(cost_fcn)

    def add (self, layer):
        """
        function enables the creation of dense, fully connected neural networks by consecutively adding layers
        """
        if type(layer) == Layer:
            self._layers.append(layer)
            # Initialisation of the weight matrix requires at least two layers
            if len(self._layers)>1:
                try:
                    self._layers[-1]._kernel_initializer(self._layers[-2])
                except WeightInitNameError as e:
                    print(f"layer no {len(self._layers)-1}: {e}")
                    print("using standard normal distribution instead" )
                    self._layers[-1].set_init("sn")
                    self._layers[-1]._kernel_initializer(self._layers[-2])   
                
        else: 
            raise NotALayerError("you can only add layers")

    def predict (self, X):
        #TODO: für mehrere Daten erweitern
        """
        function determines output values of the neural network to corresponding input values
        """
        if type(X) == list:
            a = []
            for x in X:
                a.append(self.predict(x))
            return a
        else:
            tmp = None      # tmp = a = output of each neuron in the neural net
            for no, layer in enumerate(self._layers[1:]):
                if no == 0: 
                    tmp = layer.act_func()(layer.get_weights().dot(X) + layer.get_bias())
                else:
                    tmp = layer.act_func()(layer.get_weights().dot(tmp) + layer.get_bias())
                # print(tmp)
            return tmp

    def train (self, training_data, epochs = 30, batch_size = 50, eta = 0.1, optimizer="sgd"):
        #TODO implement using class Opimizer
        w, b = self.get_network_weights_biases()
        # print(f"before Optimization: w = {w}; b = {b}")
        if optimizer=="sgd":
            self.sgd(training_data, epochs, batch_size, eta)
        else:
            print(f"Optimizer {optimizer} not found")

        w, b = self.get_network_weights_biases()
        # print(f"after Optimization: w = {w}; b = {b}")
    
    def get_network_weights_biases(self):
        """
        Return weights and biases for entire network (in case there are more
                                                      than two layers)
        """
        weights_network = [w.get_weights() for w in self._layers[1:]]
        biases_network = [b.get_bias() for b in self._layers[1:]]
        return [weights_network, biases_network]
    
    def show_progress(self, training_data):
        cost_obj = Cost("cross entropy")
        weights, biases = self.get_network_weights_biases()
        C = 0
        for x, y in training_data:
            A, Z = self.feedforward(weights, biases, x, y)
            C += cost_obj.cost()(A[-1], y)
        print(C)
    
    # Stochastic gradient descent
    def sgd (self, training_data, epochs, batch_size, eta):
        # training_data = list of (x,y) tuples, where x is the input (1600 element vector) and y the actual output (0,1) or (1,0)
        n = len(training_data)
        for _ in range(epochs):
            self.show_progress(training_data)
            rd.shuffle(training_data)   # shuffle after every epoch (in order to create "new" batches)
            # batches is a list of list of tuples (x,y) of training_data
            batches = [training_data[j:j+batch_size] for j in range(0, n, batch_size)]
            for batch in batches:
                self.update_weight_bias(batch, eta) 
            
        
    def update_weight_bias(self, batch, eta):
        """
        function to update weights and biases for the current batch using 
        gradient descent with backpropagation
        """
        weights, biases = self.get_network_weights_biases()
        # nabla w and b are lists of weights and biases where the list index represents
        # the repsective layer (i.e. 2 layers in total means only one list entry)
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights] 
        for x, y in batch:
            nabla_b_1, nabla_w_1 = self.backpropagate(x,y) # nabla_b and nabla_w for one data point
            nabla_b = [nb + nb1 for nb, nb1 in zip(nabla_b, nabla_b_1)]
            nabla_w = [nw + nw1 for nw, nw1 in zip(nabla_w, nabla_w_1)]
        # Using sgd means averaging the gradients over the batch size calculated in the for loop
        nabla_b = [nb/len(batch) for nb in nabla_b]
        nabla_w = [nw/len(batch) for nw in nabla_w]
        
        # Update rule for w and b
        weights_updated = [w-eta*nw for w, nw in zip(weights, nabla_w)]
        biases_updated = [b-eta*nb for b, nb in zip(biases, nabla_b)]
        
        for w, b, l in zip(weights_updated, biases_updated, self._layers[1:]):
            l.set_weights(w)
            l.set_bias(b)
    
    def backpropagate(self, x, y):      
        """
        Return the gradient of the cost function with respect to w and b (nabla_w, nabla_b)
        for a single training datapoint ((x_training, y_training))
        """
        no_layers = len(self._layers)
        weights, biases = self.get_network_weights_biases()
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        A, Z = self.feedforward(weights, biases, x, y)  # Given Network -> A = list of [(1600x1), (2x1)] vectors
        
        # calculate delta for last layer
        cost_obj = Cost("cross entropy")
        cost_der = cost_obj.cost()(A[-1], y, derivative=True)  # cost derivative (at last layer)
        act_func_der = self._layers[-1].act_func()(Z[-1], derivative=True)  # activation function derivative (at last layer)
        # act_func_der1 = A[1]*(1-A[-1])  # should be the same as above
        
        delta = (cost_der * act_func_der)     # Given Network -> delta = (2,1) vector
        nabla_b[-1] = delta
        # nabal_w is computed by using the input activavation (Layer l-1) and the delta from Layer l
        nabla_w[-1] = delta @ A[-2].transpose()  # so that (2x1) x (1x1600)
         
        # backwards loop through each layer to compute delta for each layer 
        # starts at second last layer (for which delta has not been calculated yet)
        if no_layers > 2: 
            for j in range(2, no_layers):
                z = Z[-j]
                act_func_der = self._layers[-j].act_func()(Z[-j], derivative=True)
                delta = (weights[-j+1].transpose()@ delta) * act_func_der
                nabla_b[-j] = delta
                # nabal_w is computed by using the input activavation (Layer l-1) and the delta from Layer l
                nabla_w[-j] = np.dot(delta, A[-j-1].transpose())
                return [nabla_b, nabla_w]
        else:
            return [nabla_b, nabla_w]
        
        # # try with while loop
        # j = 2
        # while True:
        #     z = Z[-j]
        #     act_func_der = self._layers[-j].act_func()(Z[-j], derivative=True)
        #     delta = (weights[-j+1].transpose()@ delta) * act_func_der
        #     nabla_b[-j] = delta
        #     # nabal_w is computed by using the input activavation (Layer l-1) and the delta from Layer l
        #     nabla_w[-j] = np.dot(delta, A[-j-1].transpose())
        #     return [nabla_b, nabla_w]
            
        
    def feedforward(self, weights, biases, x, y):
        """
        Return the activations and weighted inputs of the network for a single
        training datapoint
        """
        a = x           # activation of each layer
        A = [x]         # activations of the network (starting with input x)
        Z = []          # weighted input (z) of the network (first layer has no weights and therefore no z)
        
        for b, w, l in zip(biases, weights, self._layers[1:]):
            # z = np.dot(w, a)+b      # list of weights starts at Layer2, while a starts at Layer1
            z = (w @ a)+b      # list of weights starts at Layer2, while a starts at Layer1
            Z.append(z)
            a = l.act_func()(z)
            A.append(a)
        return [A, Z]
            
        
    def save_model (self, saving_loc = None):
        """ function enables saving the model for later use """
        if saving_loc == None:
            saving_loc = self._saving_loc
        if os.path.exists(saving_loc) == False:
            os.mkdir(saving_loc)
        with open(f"{saving_loc}/{datetime.now().strftime('%Y_%m_%d %Hh %M')}.ann", 'w') as file: 
            for no, layer in enumerate(self._layers):
                file.write(f"Layer {no}:*neurons: {len(layer)}*")
                file.write(f"activation_function: {layer._act_func._function_type}*")
                if no > 0:
                    file.write(f"weights: {layer.get_weights().flatten().tolist()}*")
                    file.write(f"bias: {layer.get_bias().flatten().tolist()}\n")
                else:
                    file.write("\n")

    def load_model (self, file_loc, verbose = False):
        try:
            file = open(f"{file_loc}", 'r') 
        except FileNotFoundError:
            print("model file cannot be found!")
        else:
            self._layers = []
            lines = file.readlines()
            prev_layer_neurons = 0
            for line in lines:
                line = line.replace("[","").replace("]","").replace(",","")
                elements = line.split("*")
                layer = elements[0]
                neurons = int(elements[1][len("neurons: "):])
                act_func = elements[2][len("activation_function: "):]
                self.add(Layer(neurons, act_func))
                if not layer == "Layer 0:":
                    weights = np.array(list(map(float, elements[3][len("weights: "):].split())))
                    self._layers[-1].set_weights(weights.reshape(neurons, prev_layer_neurons))
                    self._layers[-1].set_bias(np.array(list(map(float,elements[4][len("bias: "):].split()))).reshape(neurons,1))
                prev_layer_neurons = neurons  
            if verbose: 
                print(f"Neural network import successful, imported following network:")  
                print(self)
            

    def __str__ (self):
        """
        Method returns neural network structure wenn using the print() method on the network
        """
        str = ""
        for no, layer in enumerate(self._layers):
            str += f"Layer {no}: Neurons: {layer._neurons}"
            if no > 0:
                str += f" Activation Function: {layer._act_func._function_type}\n"
                str += f"weights:\n{layer._weights}\nbias:\n{layer._bias}\n"
            else:
                str += "\n"
        return str
        

class Layer:
    """
    this class is used to create layers that can be added to the model
    """
    def __init__ (self, neurons, act_func, own_act_func = None, init = "sn"):
        self._neurons = neurons
        self._init = init
        self._weights = None
        self._bias = None
        self._act_func = None
        try:
            self._act_func = ActivationFunction(act_func, own_act_func)
        except ActivationFunctionNameError as e:
            print(e)
            print("using linear function instead")
            self._act_func = ActivationFunction("linear")

    def _kernel_initializer (self, previous_layer):
        """
        this class can be used by esternal classes to set the weights and biases of each layer
        """
        self._init = Initializer(self, previous_layer)
        self._weights = self._init.get_weights()
        self._bias = self._init.get_bias()

    def act_func (self):
        return self._act_func.activation()

    def get_weights (self):
        return self._weights
    
    def set_weights (self, weights):
        self._weights = weights

    def set_bias (self, bias):
        self._bias = bias

    def get_bias (self):
        return self._bias

    def set_init (self, init):
        self._init = init
    
    def __len__ (self):
        return self._neurons


class Initializer:
    """
    this class is responsible for the kernel initialization of a layer when 
    a layer plus its successor layer are passed
    """
    def __init__ (self, layer, previous_layer):
        self._previous_layer = previous_layer
        self._layer = layer
        self._weights = None
        self._bias = None
        if layer._init == "standard_normal" or layer._init == "sn":
            self._standard_normal()
        elif layer._init == "ones":    
            self._ones()
        else:
            raise WeightInitNameError(f"weight initialisation {layer._init} not known")

    def _standard_normal (self):
        """ internal method, sets the weights and biases of the class as a standard normal distribution"""
        self._weights = np.array([[rd.gauss(0, 1)/math.sqrt(len(self._previous_layer)) for _ in range(len(self._previous_layer))] for _ in range(len(self._layer))])
        # TODO: Aufgabenstellung unklar, Bias auch durch sqrt(N_0) teilen?
        self._bias = np.array([[rd.gauss(0, 1)/math.sqrt(len(self._previous_layer))] for _ in range(len(self._layer))])
    
    # Hilfsmethode, nicht gefordert jedoch zur Überprüfung von Methoden hilfreich
    def _ones (self):
        """ internal method, sets the weights and biases of the class as ones"""
        self._weights = np.array([[1 for _ in range(len(self._previous_layer))] for _ in range(len(self._layer))])
        self._bias = np.array([[1] for _ in range(len(self._layer))]) 

    def get_weights (self):
        return self._weights

    def get_bias (self):
        return self._bias


# class Optimizer:
#     """
#     this class provides different optimizers for neural network model training
#     """
#     def __init__ (self, optimizer):
#         self._optimizer = optimizer
        
#     # Stochastic gradient descent
#     def sgd (self, training_data, epochs):
#         pass

class Cost:
    """
    this class provides different cost functions
    """
    def __init__ (self, cost_function):
        self._name_space = {"cross entropy" : self.cross_entropy, "ce" : self.cross_entropy, "mse" : self.mse}
        if cost_function in self._name_space:
            self._cost_function = cost_function
        else:
            raise CostFunctionNameError(f"cost function {cost_function} is not known")
    
    def cost (self):
        """ this class returns the cost functions of a Cost object """
        return self._name_space[self._cost_function]
    
    # TODO: anders als in der PDF wird die Cross entropy cost function in der Literatur noch durch N_S geteilt?
    def cross_entropy (self, a, y, derivative = False):
        """ cross entropy cost function, can be used externally """
        #avoids that eps a=1 or a=0 (then due to cross entropy cost fcn, div/0 error occurs)
        eps = 1e-10
        for i in range(len(a)):
            if (a[i][0]> 1-eps):
                a[i][0] = 1-eps
            if (a[i][0]< 0+eps):
                a[i][0] = 0+eps
                         
        c = 0
        if not derivative:  #(returns scalar)
            for j in range(len(y)):
                c -= (y[j]*math.log(a[j])+(1-y[j])*math.log(1-a[j]))
        else:   # (returns vector)
            c = np.zeros((len(y),1))
            for j in range(len(y)):
                c[j] = (a[j]-y[j])/(a[j]*(1-a[j]))
        return c

    def mse (self, y, a, derivative = False):
        """ mean squared error cost function, can be used externally """
        c = 0
        if not derivative:
            for i in range(len(y)):
                c += (y[i]-a[i])**2
        else:
            for i in range(len(y)):
                c += 2*(y[i]-a[i])
        return c

class ActivationFunction:
    """
    this class provides different activation functions that can be passed to a layer
    """
    
    def __init__ (self, function_type, function = None):
        self._name_space = {"sigmoid" : self.sigmoid, "linear" : self.linear}
        if function == None:
            if function_type in self._name_space:
                self._function_type = function_type
            else:
                raise ActivationFunctionNameError(f"activation function {function_type} is not known")
        else:
            self._function_type = function_type
            self._add_activation(function_type, function)

    # derivative: bool, if False: original activation function, if True: first order derivative
    def activation (self):
        """ this class returns the activation functions of an ActivationFunction object """
        return self._name_space[self._function_type]

    def _add_activation (self, activation_name, activation):
        self._name_space[activation_name] = activation

    def sigmoid (self, z, derivative = False):
        """ sigmoid activation function, can be used externally"""
        sigmoid = lambda z : 1/(1+np.exp(-z))
        if not derivative:
            return sigmoid(z)
        else:
            return sigmoid(z)*(1-sigmoid(z))
    
    def linear(self, z, derivative = False):
        """ linear activation function, can be used externally """
        if not derivative:
            return z
        else:
            return 1

class NameError(Exception): pass
class ActivationFunctionNameError(NameError): pass
class WeightInitNameError(NameError): pass
class CostFunctionNameError(NameError): pass
class FunctionalErrors(Exception): pass
class NotALayerError(FunctionalErrors):pass

if __name__ == "__main__":
    def tanh (z, derivative = False):
            """ tangens hyperbolicus activation function"""
            if not derivative:
                return np.tanh(z)
            else:
                return 1 - np.pow(np.tanh(z),2)

    neural_net = NeuralNetwork()
    
    
    n_l1 = 1600            # neurons layer 1
    # n_l2 = 5            # neurons layer 2
    n_l3 = 2            # neurons layer 3
    training_size = 50   # fictional size of training data
    neural_net.add(Layer(n_l1, "sigmoid", init="sn"))
    # neural_net.add(Layer(n_l2, tanh, init = "sn"))
    neural_net.add(Layer(n_l3, "sigmoid", init = "sn"))
    
    # neural_net.save_model()
    # model_name = "2021_09_23 18h 24.ann"
    # neural_net.load_model(f"./Saved Models/{model_name}")

    # Example Training data
    x = [np.random.randn(n_l1, 1) for _ in range (0, training_size)]
    y = [np.array([[0],[1]]) for _ in range (0, training_size)]
    training_data = [(X, Y) for X, Y in zip(x,y)]
    
    neural_net.train(training_data, epochs=100, batch_size=1, eta=0.1)
    # print(neural_net)
    
    x_test = x[0]      
    # print("Prediction: ") 
    # print(neural_net.predict(x_test))
    
    # Example Training data

