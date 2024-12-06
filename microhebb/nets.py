import numpy as np

#====================================================================
# activations
#====================================================================

def softmax(x):
        
    exp = np.exp(x - x.max()).T
    
    return (exp/exp.sum(axis=0)).T
    
def d_softmax(x):
        
    p = softmax(x)
        
    return p*(1 - p)

def sigmoid(x):
    
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    
    return sigmoid(x)*(1 - sigmoid(x))

#====================================================================
# Layers
#====================================================================

class layer:
    
    def __init__(self, X, Y, f, f_=None, w_scale=0.001):
        
        self.X = X
        
        self.Y = Y
        
        self.f = f
        
        self.f_ = f_
        
        self.W = np.random.uniform(-w_scale, w_scale, (X, Y))
        
        #self.b = np.random.uniform(-w_scale, w_scale, (Y, 1))
        
        self.eta = 0
        
    def forward(self, x, y=None):
        
        u = x@self.W# + self.b
        
        o = self.f(u)
        
        return o
    
class classifier(layer):
    
    def train(self, x, y, u, o):
        
        e = (y - o.T)*self.f_(u).T
        c = (x@e.T).T
        deltaW = 2*self.eta*c
        self.W += deltaW.T
        
        
        
        #self.b += deltaB
    
    def train_forward(self, x, y):
        
        u = x@self.W# + self.b
        
        o = self.f(u)
        
        self.train(x.T, y, u, o)
        
        return o

class hebb(layer):
    
    def train(self, x, y):
        
        deltaW = x@y.T*self.eta
        
        self.W += deltaW
    
    def train_forward(self, x, y=None):
        
        if np.any(y):
            
            self.train(x, y)
            
            return y
        
        u = x@self.W
        
        o = self.f(u)

        self.train(x, o.T)
        
        return o
    
class softhebb(layer):
    
    def __init__(self, X, Y, f, f_=None, w_scale=0.001):
        
        self.X = X
        
        self.Y = Y
        
        self.f = f
        
        self.f_ = f_
        
        self.W = np.random.uniform(-w_scale, w_scale, (X, Y))
        
        self.W_norm()
        
        self.eta = 0
    
    def train(self, x, o, u):
        
        e = (x - (u@self.W.T).T)
        
        deltaW = self.eta*e@o
        
        self.W += deltaW
        
        #self.W_norm()
    
    def train_forward(self, x):
        
        u = x@self.W
        
        o = self.f(u)
        
        self.train(x.T, o, u)
        
        return o

    def W_norm(self):
        
        n = np.sqrt((self.W**2).sum(axis=1))
        
        self.W = self.W/np.linalg.norm(self.W, axis=1)[:,np.newaxis]
        
#====================================================================
# nets
#====================================================================

class net:
    
    W = None
    
    eta = 0
    
    layers = []
    
    def unsupervised_forward(X):
        
        pass
    
    def supervised_forward(x, y):
        
        return None
    
    def unsupervised_sgd(X, T):
        
        for t in range(T):
        
            i = np.random.randint(len(X))
            z = unsupervised_forward(X[i].reshape((28*28, 1)).T)
        
    
    def supervised_sgd(X, Y, T):
        
        sqreE = []
        
        for t in range(T):
        
            i = np.random.randint(len(X))
            z = supervised_forward(X[i].reshape((28*28, 1)).T, Y[i].reshape((10,1)))
            sqreE += [((Y[i] - z[-1])**2).sum()/len(z[-1])]
        
        return sqreE

class softmax_classifier(net):
    
    def __init__(self, arch, w_scale=0.001):
        
        self.depth = len(arch)
        
        self.layers = [softhebb(X=arch[i-1], Y=arch[i], f=softmax, w_scale=w_scale) for i in range(1, self.depth-1)] + [classifier(X=arch[-2], Y=arch[-1], f=softmax, f_=d_softmax)]
        
        self.eta = 0
    
    def forward(self, x):
        
        vals = [np.copy(x)]
        
        for l in self.layers:
            
            l.eta = 0
            
            x = l.forward(x)
            
            vals += [np.copy(x)]
        
        return vals
    
    def unsupervised_forward(self, x):
        
        vals = [np.copy(x)]
        
        for l in self.layers[:-1]:
            
            l.eta = self.eta
            
            x = l.train_forward(x)
            
            vals += [np.copy(x)]
        
        return vals
    
    def supervised_forward(self, x, y):
        
        for l in self.layers[:-1]:
            
            l.eta = self.eta
            
            x = l.forward(x)
        
        l = self.layers[-1]
        
        l.eta = self.eta
        
        x = l.train_forward(x, y)
        
        return [x]
    
    '''
    def train(self, x, y):
        
        return self.unsupervised_forward(x) + self.supervised_forward(x, y)
        '''