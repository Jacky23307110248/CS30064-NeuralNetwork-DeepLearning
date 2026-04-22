from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, initialize_method=np.random.normal):
        self.size_list = size_list
        self.act_func = act_func
        self.initialize_method = initialize_method

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(
                    in_dim=size_list[i],
                    out_dim=size_list[i + 1],
                    initialize_method=initialize_method,
                )
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    CNN matching a common PyTorch-style stack (MNIST 28x28, grayscale):
      Conv(1->16, k=3, p=1) -> ReLU -> MaxPool(2,2)
    -> Conv(16->32, k=3, p=1) -> ReLU -> MaxPool(2,2)
    -> Flatten -> Linear(32*7*7 -> hidden) -> ReLU -> Linear(hidden -> num_classes)
    """
    def __init__(
        self,
        size_list=None,
        act_func='ReLU',
        lambda_list=None,
        conv_kernel_size=3,
        conv_stride=1,
        conv_padding=1,
        input_height=28,
        input_width=28,
        initialize_method=np.random.normal,
        fc_hidden=128,
        c1_out=16,
        c2_out=32,
    ):
        if act_func == 'Logistic':
            raise NotImplementedError
        elif act_func != 'ReLU':
            raise ValueError(f"Unsupported act_func for Model_CNN: {act_func}")

        # size_list (optional): [in_channels, num_classes] or [in_channels, _, num_classes]; middle entry ignored.
        # If None: MNIST defaults in_channels=1, num_classes=10.
        if size_list is None:
            in_channels, num_classes = 1, 10
            self.size_list = [in_channels, num_classes]
        elif len(size_list) == 2:
            in_channels, num_classes = int(size_list[0]), int(size_list[1])
            self.size_list = [in_channels, num_classes]
        elif len(size_list) == 3:
            in_channels, num_classes = int(size_list[0]), int(size_list[2])
            self.size_list = [in_channels, size_list[1], num_classes]
        else:
            raise ValueError("Model_CNN size_list must be None, length 2, or length 3.")

        self.act_func = act_func
        self.conv_kernel_size = int(conv_kernel_size)
        self.conv_stride = int(conv_stride)
        self.conv_padding = int(conv_padding)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.initialize_method = initialize_method
        self.fc_hidden = int(fc_hidden)
        self.c1_out = int(c1_out)
        self.c2_out = int(c2_out)
        self.num_classes = int(num_classes)

        h, w = self.input_height, self.input_width
        if h % 4 != 0 or w % 4 != 0:
            raise ValueError("Model_CNN expects input H,W divisible by 4 (two 2x2 pools).")

        conv1 = conv2D(
            in_channels=in_channels,
            out_channels=self.c1_out,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
            initialize_method=initialize_method,
        )
        pool1 = MaxPool2d(kernel_size=2, stride=2)
        conv2 = conv2D(
            in_channels=self.c1_out,
            out_channels=self.c2_out,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
            initialize_method=initialize_method,
        )
        pool2 = MaxPool2d(kernel_size=2, stride=2)
        flat = Flatten()
        fc_in = self.c2_out * (h // 4) * (w // 4)
        fc1 = Linear(in_dim=fc_in, out_dim=self.fc_hidden, initialize_method=initialize_method)
        fc2 = Linear(in_dim=self.fc_hidden, out_dim=self.num_classes, initialize_method=initialize_method)

        self.layers = [
            conv1, ReLU(), pool1,
            conv2, ReLU(), pool2,
            flat, fc1, ReLU(), fc2,
        ]

        self.lambda_list = lambda_list
        if lambda_list is not None:
            if len(lambda_list) != 4:
                raise ValueError("Model_CNN lambda_list must contain 4 values (conv1, conv2, fc1, fc2).")
            opt = [layer for layer in self.layers if layer.optimizable]
            if len(opt) != 4:
                raise RuntimeError("Model_CNN: expected 4 optimizable layers.")
            for layer, lam in zip(opt, lambda_list):
                layer.weight_decay = True
                layer.weight_decay_lambda = lam

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X.ndim == 2:
            expected = self.input_height * self.input_width
            if X.shape[1] != expected:
                raise ValueError(
                    f"Flattened input dim {X.shape[1]} does not match "
                    f"input_height*input_width={expected}."
                )
            X = X.reshape(X.shape[0], 1, self.input_height, self.input_width)
        elif X.ndim != 4:
            raise ValueError(f"Unexpected input shape for Model_CNN: {X.shape}")

        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        if isinstance(param_list, str):
            with open(param_list, 'rb') as f:
                param_list = pickle.load(f)

        if not isinstance(param_list, list) or len(param_list) < 3:
            raise ValueError("Invalid checkpoint format for Model_CNN.")

        meta = param_list[0]
        if meta != 'Model_CNN':
            raise ValueError(f"Checkpoint model type mismatch: {meta}")

        cfg = param_list[1]
        if not isinstance(cfg, dict):
            raise ValueError(
                "Unsupported Model_CNN checkpoint: expected a config dict at index 1; retrain if needed."
            )

        self.__init__(
            cfg.get('size_list'),
            cfg.get('act_func', 'ReLU'),
            cfg.get('lambda_list'),
            conv_kernel_size=cfg.get('conv_kernel_size', 3),
            conv_stride=cfg.get('conv_stride', 1),
            conv_padding=cfg.get('conv_padding', 1),
            input_height=cfg.get('input_height', 28),
            input_width=cfg.get('input_width', 28),
            fc_hidden=cfg.get('fc_hidden', 128),
            c1_out=cfg.get('c1_out', 16),
            c2_out=cfg.get('c2_out', 32),
        )
        saved_layers = param_list[2:]

        opt_layers = [layer for layer in self.layers if layer.optimizable]
        if len(saved_layers) != len(opt_layers):
            raise ValueError("Checkpoint parameter count mismatch for Model_CNN.")

        for layer, saved in zip(opt_layers, saved_layers):
            layer.W = saved['W']
            layer.b = saved['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = saved['weight_decay']
            layer.weight_decay_lambda = saved['lambda']

    def save_model(self, save_path):
        param_list = [
            'Model_CNN',
            {
                'size_list': self.size_list,
                'act_func': self.act_func,
                'conv_kernel_size': self.conv_kernel_size,
                'conv_stride': self.conv_stride,
                'conv_padding': self.conv_padding,
                'input_height': self.input_height,
                'input_width': self.input_width,
                'lambda_list': self.lambda_list,
                'fc_hidden': self.fc_hidden,
                'c1_out': self.c1_out,
                'c2_out': self.c2_out,
            },
        ]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)