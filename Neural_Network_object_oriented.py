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
        self._optimizer = None
        self._cost = None
        self._saving_loc = "Saved Models"

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
        """
        function determines output values of the neural network to corresponding input values
        """
        if type(X) == list:
            a = []
            for x in X:
                a.append(self.predict(x))
            return a
        else:
            self._layers[0].set_a(X)
            self._layers[0].set_z(X)
            for no, layer in enumerate(self._layers[1:]):
                if no == 0: 
                    layer.set_z((layer.get_weights().dot(X) + layer.get_bias()))
                    layer.set_a(layer.act_func()(layer.get_z()))
                else:
                    layer.set_z(layer.get_weights().dot(self._layers[no].get_a()) + layer.get_bias())
                    layer.set_a(layer.act_func()(layer.get_z()))
            return self._layers[-1].get_a()

    def train (self, training_data, epochs = 30, batch_size = 1, verbose = 0):
        batches = list(range(0, len(training_data), batch_size))
        batches.append(len(training_data))
        for epoch in range(epochs):
            #TODO: batch nach jeder Epoche aktualisieren, Gradientenabstieg lediglich für ein Datenpaar
            rd.shuffle(training_data)
            X = list(list(zip(*training_data))[0])
            y = list(list(zip(*training_data))[1])
            weight_grad = []
            bias_grad = []
            for s in range(len(self._layers)-1):
                weight_grad.append(np.zeros_like(self._layers[s+1].get_weights()))
                bias_grad.append(np.zeros_like(self._layers[s+1].get_bias()))
            # print(f"weight grad:{weight_grad}")
            if verbose == 1:
                print(f"Epoch no: {epoch} cost: {self._cost.cost()(self.predict(X), y)}")
            for n in range(len(batches)-1):
                if batches[n+1]-batches[n] > 1:
                    for batch in range(batches[n], batches[n+1]):
                        weight_grad[n], bias_grad[n] = self._backprop(X[batch], y[batch], weight_grad, bias_grad)
                    # average of the gradients over the batches
                    for j in range(len(self._layers)-1):
                        weight_grad[j] /= len(batches[j])
                        bias_grad[j] /= len(batches[j])
                else:
                    weight_grad, bias_grad = self._backprop(X[batches[n]], y[batches[n]], weight_grad, bias_grad)
                for j in range(len(self._layers)-1):
                    weights, bias = self._optimizer.optimize()(self._layers[1+j].get_weights(), self._layers[1+j].get_bias(), weight_grad[j], bias_grad[j], batch_size)
                    self._layers[1+j].set_weights(weights)
                    self._layers[1+j].set_bias(bias)
    
    def _backprop (self, X, y, weight_grad, bias_grad):
        self.predict(X)
         
        layer = self._layers[-1]
        prev_layer = self._layers[-2]
        # tmp_(S) = delta_C/delta_a_(S) * delta_a_(S)/delta_z_(S) 
        tmp = self._cost.cost()(layer.get_a(), y, True) * layer.act_func()(layer.get_z(), True)
        # delta_C/delta_w_(S) = tmp_w_(S) * delta_z_(S)/delta_w_(S)
        weight_grad[-1] += tmp @ prev_layer.get_a().T
        # delta_C/delta_b_(S) = tmp_b_(S) * delta_z_(S)/delta_b_(S)
        bias_grad[-1] += tmp
        if len(self._layers) > 2:
            for j in range(1, len(self._layers)-1):
                next_layer = self._layers[-j]
                layer = self._layers[-j-1]
                prev_layer = self._layers[-j-2]
                # tmp_(S-j) = tmp_w_(S-j+1) * delta_z_(S-j+1)/delta_a_(S-j) * delta_a_(S-j)/delta_z_(S-j) 
                tmp = (next_layer.get_weights().T @ tmp) * layer.act_func()(layer.get_z(), True)
                # delta_C/delta_w_(S-j) * delta_z_(S-j)/delta_w_(S-j)
                weight_grad[-1-j] += tmp * prev_layer.get_a().T
                # delta_C/delta_b_(S-j) * delta_z_(S-j)/delta_b_(S-j)
                bias_grad[-1-j] += tmp
        return weight_grad, bias_grad
    
    def compile (self, optimizer, learning_rate, loss):
        try:
            self._optimizer = Optimizer(optimizer, learning_rate)
        except OptimizerNameError as e:
            print(e)
            print("using stochastic gradient descent instead")
            self._optimizer = Optimizer("sgd", learning_rate)
        try:
            self._cost = Cost(loss)
        except CostFunctionNameError as e:
            print(e)
            print("using cross entropy instead")
            self._cost = Cost("ce")

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

    def load_model (self, file_loc):
        try:
            file = open(f"{file_loc}", 'r') 
        except FileNotFoundError:
            print("file not available!")
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
                    bias = np.array(list(map(float,elements[4][len("bias: "):].split())))
                    self._layers[-1].set_weights(weights.reshape(neurons, prev_layer_neurons))
                    self._layers[-1].set_bias(bias.reshape(neurons, 1))
                    print(neurons, act_func, weights)
                prev_layer_neurons = neurons             
            

    def __str__ (self):
        """
        Method returns weights and biasas of the network as a string
        """
        str = ""
        for no, layer in enumerate(self._layers[1:]):
            str += f"weights layer {no+1}:\n{layer._weights}\n"
            str += f"bias layer {no+1}:\n{layer._bias}\n"
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
        self._z = None
        self._activation = None 
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

    def set_z (self, z):
        self._z = z

    def get_z (self):
        return self._z

    def get_a (self):
        return self._activation
    
    def set_a (self, a):
        self._activation = a

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
        self._bias = np.array([[rd.gauss(0, 1)] for _ in range(len(self._layer))])
    
    # Hilfsmethode, nicht gefordert jedoch zur Überprüfung von Methoden hilfreich
    def _ones (self):
        """ internal method, sets the weights and biases of the class as ones"""
        self._weights = np.array([[1 for _ in range(len(self._previous_layer))] for _ in range(len(self._layer))])
        self._bias = np.array([[1] for _ in range(len(self._layer))]) 

    def get_weights (self):
        return self._weights

    def get_bias (self):
        return self._bias


class Optimizer:
    """
    this class provides different optimizers for neural network model training
    """
    def __init__ (self, optimizer_type, learning_rate = 0.01):
        self._name_space = {"sgd" : self.sgd}
        self._learning_rate = learning_rate
        if optimizer_type in self._name_space:
            self._optimizer_type = optimizer_type
        else:
            raise OptimizerNameError(f"optimizer {optimizer_type} is not known")

    def optimize(self):
        return self._name_space[self._optimizer_type]
        
    # Stochastic gradient descent
    def sgd (self, old_weights, old_biases, weight_grad, bias_grad, batch_size):
        biases = old_biases + self._learning_rate/batch_size*bias_grad
        weights = old_weights + self._learning_rate/batch_size*weight_grad
        return weights, biases

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
        if not derivative:
            for i in range(len(y)):
                c -= (y[i]*np.log(a[i])+(1-y[i])*np.log(1-a[i]))
        else:
            # for i in range(len(y)):
            #     c -= (a[i]-y[i])/(a[i]*(1-a[i]))
            c = np.zeros((len(y),1))
            for i in range(len(y)):
                c[i] = (a[i]-y[i])/(a[i]*(1-a[i]))
        return c
    
    def mse (self, y, a, derivative = False):
        """ cross entropy cost function, can be used externally """
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
        """ this method returns the activation functions of an ActivationFunction object """
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
class OptimizerNameError(NameError): pass
class FunctionalErrors(Exception): pass
class NotALayerError(FunctionalErrors):pass

if __name__ == "__main__":
    def tanh (z, derivative = False):
            """ tangens hyperbolicus activation function"""
            if not derivative:
                return np.tanh(z)
            else:
                return 1 - np.pow(np.tanh(z),2)

    n_l1 = 1600         # neurons layer 1
    n_l2 = 5            # neurons layer 2
    n_l3 = 1            # neurons layer 3
    training_size = 2   # fictional size of training data

    # Example Training data
    x = [np.random.randn(n_l1, 1) for _ in range (0, training_size)]
    y = [np.array([[0],[1]]) for _ in range (0, training_size)]
    training_data = [(X, Y) for X, Y in zip(x,y)]

    neural_net = NeuralNetwork()
    neural_net.add(Layer(n_l1, "sigmoid", init="sn"))
    # neural_net.add(Layer(n_l2, tanh, init = "sn"))
    neural_net.add(Layer(n_l3, "sigmoid", init = "sn"))
    print(neural_net)
    neural_net.compile("sgd", 0.1, "mse")
    print(neural_net.predict(list(list(zip(*training_data))[0])))
    neural_net.train(training_data, epochs = 100, batch_size = 1, verbose = 1)
    print(neural_net.predict(x))
    print(neural_net)
    # neural_net.save_model()
    

    # model_name = "2021_09_23 18h 24.ann"
    # neural_net.load_model(f"./Saved Models/{model_name}")

   


