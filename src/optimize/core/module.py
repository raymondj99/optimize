from optimize.core.layer import Layer
from optimize.core.functional import log_prob

class Module(Layer):
    '''
    Module inherits from a layer and does not requires a backward function
    However you can override it if you want.
    The Module class provides an interface to create your own models with trainable parameters
    You can register it in the init function and pass them to the optimizers by parameters() method
    '''
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Module'
        self.params = {}

    def register_param(self, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v
    
    def parameters(self):
        return self.params
    

class MSELoss(Module):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.MSE'

    def forward(self, y_true, y_pred):
        diff = y_true - y_pred
        diff_squared = diff ** 2
        mse = diff_squared.mean()  
        return mse
        

class BCELoss(Module):
    def __init__(self):
        super().__init__()
        self.desc = 'optimize.Layer.CE'

    def forward(self, y_true, y_pred):
        log_positive_prob = log_prob(y_pred)
        log_negative_prob = log_prob(1 - y_pred)
        nll = y_true * log_positive_prob + (1 - y_true) * log_negative_prob
        nll = -nll.mean()
        return nll