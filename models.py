import torch
from torch import nn
import numpy as np
import math
from torch.nn import init


## PEMLP
class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs) 
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)
    
class PEMLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, N_freqs=10):
        super().__init__()
        self.enconding = PositionalEncoding(in_channels=in_features, N_freqs=N_freqs)
        
        self.net = []
        self.net.append(nn.Linear(self.enconding.out_channels, hidden_features))
        self.net.append(nn.ReLU(True))

        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(True))

        final_linear = nn.Linear(hidden_features, out_features)                
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(self.enconding(coords))
        return output


##    
def init_weights(m, omega=1, c=1, is_first=False): # Default: Pytorch initialization
    if hasattr(m, 'weight'):
        fan_in = m.weight.size(-1)
        if is_first:
            bound = 1 / fan_in # SIREN
        else:
            bound = math.sqrt(c / fan_in) / omega
        init.uniform_(m.weight, -bound, bound)
    
def init_weights_kaiming(m):
    if hasattr(m, 'weight'):
        init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_bias(m, k):
    if hasattr(m, 'bias'):
        init.uniform_(m.bias, -k, k)

'''Used for SIREN, FINER, Gauss, Wire, etc.'''
def init_weights_cond(init_method, linear, omega=1, c=1, is_first=False):
    init_method = init_method.lower()
    if init_method == 'sine':
        init_weights(linear, omega, 6, is_first)    # SIREN initialization
    ## Default: Pytorch initialization

def init_bias_cond(linear, fbs=None, is_first=True):
    if is_first and fbs != None:
        init_bias(linear, fbs)
    ## Default: Pytorch initialization

''' 
    FINER activation
    TODO: alphaType, alphaReqGrad
'''
def generate_alpha(x, alphaType=None, alphaReqGrad=False):
    """
    if alphaType == ...:
        return ...
    """
    with torch.no_grad():
        return torch.abs(x) + 1
    
def finer_activation(x, omega=1, alphaType=None, alphaReqGrad=False):
    return torch.sin(omega * generate_alpha(x, alphaType, alphaReqGrad) * x)


'''
    Gauss. & GF(FINER++Gauss.) activation
'''
def gauss_activation(x, scale):
    return torch.exp(-(scale*x)**2)

def gauss_finer_activation(x, scale, omega, alphaType=None, alphaReqGrad=False):
    return gauss_activation(finer_activation(x, omega, alphaType, alphaReqGrad), scale)

    
'''
    Wire & WF activation
'''
def wire_activation(x, scale, omega_w):
    return torch.exp(1j*omega_w*x - torch.abs(scale*x)**2)

def finer_activation_complex_sep_real_imag(x, omega=1):
    with torch.no_grad():
        alpha_real = torch.abs(x.real) + 1
        alpha_imag = torch.abs(x.imag) + 1
    x.real = x.real * alpha_real
    x.imag = x.imag * alpha_imag
    return torch.sin(omega * x)

def wire_finer_activation(x, scale, omega_w, omega, alphaType=None, alphaReqGrad=False):
    if x.is_complex():
        return wire_activation(finer_activation_complex_sep_real_imag(x, omega), scale, omega_w)
    else:
        return wire_activation(finer_activation(x, omega), scale, omega_w)


## FINER 
class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega=30, 
                 is_first=False, is_last=False, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.omega = omega
        self.is_last = is_last ## no activation
        self.alphaType = alphaType
        self.alphaReqGrad = alphaReqGrad
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
        # init bias
        init_bias_cond(self.linear, fbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return finer_activation(wx_b, self.omega)
        return wx_b # is_last==True
      
class Finer(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 first_omega=30, hidden_omega=30, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, 
                                   omega=first_omega, 
                                   init_method=init_method, init_gain=init_gain, fbs=fbs,
                                   alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, 
                                       omega=hidden_omega, 
                                       init_method=init_method, init_gain=init_gain, hbs=hbs,
                                       alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        self.net.append(FinerLayer(hidden_features, out_features, is_last=True, 
                                   omega=hidden_omega, 
                                   init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)



## SIREN 
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)    
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        return out

class Siren(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output



## WIRE
class ComplexGaborLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=10, omega_w=20, 
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1):
        super().__init__()
        self.scale = scale
        self.omega_w = omega_w
        self.is_last = is_last ## no activation
        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        # init weights
        init_weights_cond(init_method, self.linear, omega_w, init_gain, is_first)
        
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return wire_activation(wx_b, self.scale, self.omega_w)
        return wx_b # is_last==True

class Wire(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 scale=10, omega_w=20,
                 init_method='sine', init_gain=1):
        super().__init__()
        hidden_features = int(hidden_features / np.sqrt(2))
        
        self.net = []
        self.net.append(ComplexGaborLayer(in_features, hidden_features, is_first=True, 
                                          scale=scale, omega_w=omega_w, 
                                          init_method=init_method, init_gain=init_gain))

        for i in range(hidden_layers):
            self.net.append(ComplexGaborLayer(hidden_features, hidden_features, 
                                              scale=scale, omega_w=omega_w, 
                                              init_method=init_method, init_gain=init_gain))

        self.net.append(ComplexGaborLayer(hidden_features, out_features, is_last=True, 
                                          scale=scale, omega_w=omega_w, 
                                          init_method=init_method, init_gain=init_gain)) # omega_w: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output.real

 

## WFINER
class WFLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=10, omega_w=20, omega=1,
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.scale = scale
        self.omega_w = omega_w
        self.omega = omega
        self.is_last = is_last ## no activation
        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        
        # init weights
        init_weights_cond(init_method, self.linear, omega*omega_w, init_gain, is_first)
        
        # init bias 
        init_bias_cond(self.linear, fbs, is_first)
        
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return wire_finer_activation(wx_b, self.scale, self.omega_w, self.omega)
        return wx_b # is_last==True

class WF(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 scale=10, omega_w=20, omega=1,
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        hidden_features = int(hidden_features / np.sqrt(2))
        
        self.net = []
        self.net.append(WFLayer(in_features, hidden_features, is_first=True,
                                    omega=omega, scale=scale, omega_w=omega_w, 
                                    init_method=init_method, init_gain=init_gain, fbs=fbs,
                                    alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net.append(WFLayer(hidden_features, hidden_features, 
                                        omega=omega, scale=scale, omega_w=omega_w,
                                        init_method=init_method, init_gain=init_gain, hbs=hbs,
                                        alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        self.net.append(WFLayer(hidden_features, out_features, is_last=True, 
                                    omega=omega, scale=scale, omega_w=omega_w,
                                    init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output.real



## Gauss
class GaussLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0,
                 is_first=False, is_last=False,
                 init_method='Pytorch', init_gain=1):
        super().__init__()
        self.scale = scale
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # init weights
        init_weights_cond(init_method, self.linear, None, init_gain, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return gauss_activation(wx_b, self.scale)
        return wx_b # is_last==True
    
class Gauss(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 scale=30,
                 init_method='Pytorch', init_gain=1):
        super().__init__()
        self.net = []
        self.net.append(GaussLayer(in_features, hidden_features, is_first=True, 
                                   scale=scale,
                                   init_method=init_method, init_gain=init_gain))

        for i in range(hidden_layers):
            self.net.append(GaussLayer(hidden_features, hidden_features, 
                                       scale=scale,
                                       init_method=init_method, init_gain=init_gain))
            
        self.net.append(GaussLayer(hidden_features, out_features, is_last=True, 
                                   scale=scale,
                                   init_method=init_method, init_gain=init_gain))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output



## GFINER: Gauss-Finer
class GFLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=3, omega=1,
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.scale = scale
        self.omega = omega
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features, bias=bias)
                
        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
            
        # init bias 
        init_bias_cond(self.linear, fbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            gauss_finer_activation(wx_b, self.scale, self.omega)
        return wx_b # is_last==True

class GF(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, 
                 scale=3, omega=1, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.net = []
        self.net.append(GFLayer(in_features, hidden_features, is_first=True, 
                                scale=scale, omega=omega, 
                                init_method=init_method, init_gain=init_gain, fbs=fbs,
                                alphaType=alphaType, alphaReqGrad=alphaReqGrad))
        
        for i in range(hidden_layers):
            self.net.append(GFLayer(hidden_features, hidden_features, 
                                     scale=scale, omega=omega, 
                                     init_method=init_method, init_gain=init_gain, hbs=hbs,
                                     alphaType=alphaType, alphaReqGrad=alphaReqGrad))
         
        self.net.append(GFLayer(hidden_features, out_features, is_last=True, 
                                scale=scale, omega=omega, 
                                init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output
    