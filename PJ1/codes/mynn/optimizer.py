from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.model = model
        lr0 = float(init_lr)
        self.init_lr = lr0  # fixed reference (never changed by schedulers)
        self.lr = lr0  # effective step size; learning-rate schedulers update this

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    g = layer.grads[key]
                    # L2 weight decay as gradient term (λ w), same idea as PyTorch SGD weight_decay.
                    if layer.weight_decay:
                        g += layer.weight_decay_lambda * layer.params[key]
                    layer.params[key] -= self.lr * g
                # Keep direct attributes synchronized with params dict.
                if hasattr(layer, 'W') and 'W' in layer.params:
                    layer.W = layer.params['W']
                if hasattr(layer, 'b') and 'b' in layer.params:
                    layer.b = layer.params['b']


class MomentGD(Optimizer):
    """
    SGD with momentum: L2-style weight decay is added to the gradient (``g + λ w``) **before**
    the velocity update (same ordering as common PyTorch SGD + weight_decay + momentum).
    Then ``v <- μ v + g_eff`` and ``param <- param - lr * v``.
    """

    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = float(mu)
        self._velocities = {}
        for layer in self.model.layers:
            if not getattr(layer, "optimizable", False):
                continue
            lid = id(layer)
            self._velocities[lid] = {
                k: np.zeros_like(layer.params[k]) for k in layer.params.keys()
            }

    def step(self):
        for layer in self.model.layers:
            if not getattr(layer, "optimizable", False):
                continue
            lid = id(layer)
            vel = self._velocities[lid]
            for key in layer.params.keys():
                g = layer.grads[key]
                if layer.weight_decay:
                    g += layer.weight_decay_lambda * layer.params[key]
                vel[key] = self.mu * vel[key] + g
                layer.params[key] -= self.lr * vel[key]
            if hasattr(layer, "W") and "W" in layer.params:
                layer.W = layer.params["W"]
            if hasattr(layer, "b") and "b" in layer.params:
                layer.b = layer.params["b"]