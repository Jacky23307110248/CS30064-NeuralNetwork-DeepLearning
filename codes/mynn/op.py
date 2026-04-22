from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # Default np.random.normal uses std=1, which is far too large for deep nets; use Kaiming He for ReLU-style training.
        if initialize_method is np.random.normal:
            scale = np.sqrt(2.0 / max(in_dim, 1))
            self.W = np.random.normal(0.0, scale, size=(in_dim, out_dim))
            self.b = np.zeros((1, out_dim), dtype=self.W.dtype)
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None, "Linear.backward called before forward."
        assert grad.ndim == 2 and self.input.ndim == 2
        assert grad.shape[0] == self.input.shape[0]

        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if initialize_method is np.random.normal:
            fan_in = in_channels * kernel_size * kernel_size
            scale = np.sqrt(2.0 / max(fan_in, 1))
            self.W = np.random.normal(0.0, scale, size=(out_channels, in_channels, kernel_size, kernel_size))
            self.b = np.zeros((1, out_channels), dtype=self.W.dtype)
        else:
            self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
            self.b = initialize_method(size=(1, out_channels))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.input_padded = None
        self.input_cols = None
        self.out_h = None
        self.out_w = None
        self.col_k = None
        self.col_i = None
        self.col_j = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        assert X.ndim == 4
        batch, channels, H, W = X.shape
        assert channels == self.in_channels

        k = self.kernel_size
        s = self.stride
        p = self.padding

        if p > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        else:
            X_pad = X

        H_pad, W_pad = X_pad.shape[2], X_pad.shape[3]
        out_h = (H_pad - k) // s + 1
        out_w = (W_pad - k) // s + 1
        assert out_h > 0 and out_w > 0

        # im2col: [N, C, H_pad, W_pad] -> [N*out_h*out_w, C*k*k]
        col_k, col_i, col_j = self._get_im2col_indices(
            channels, H_pad, W_pad, k, s, out_h, out_w
        )
        cols = X_pad[:, col_k, col_i, col_j]  # [N, C*k*k, out_h*out_w]
        cols = cols.transpose(0, 2, 1).reshape(batch * out_h * out_w, -1)

        W_col = self.W.reshape(self.out_channels, -1)  # [out_c, C*k*k]
        out = cols @ W_col.T + self.b  # [N*out_h*out_w, out_c]
        out = out.reshape(batch, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        self.input = X
        self.input_padded = X_pad
        self.input_cols = cols
        self.out_h = out_h
        self.out_w = out_w
        self.col_k = col_k
        self.col_i = col_i
        self.col_j = col_j
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.input is not None and self.input_padded is not None
        assert grads.ndim == 4

        X = self.input
        X_pad = self.input_padded
        batch, _, H, W = X.shape
        _, _, H_out, W_out = grads.shape
        k = self.kernel_size
        p = self.padding

        assert self.input_cols is not None
        assert self.out_h == H_out and self.out_w == W_out

        grad_col = grads.transpose(0, 2, 3, 1).reshape(batch * H_out * W_out, self.out_channels)
        W_col = self.W.reshape(self.out_channels, -1)

        # dW, db
        dW_col = grad_col.T @ self.input_cols  # [out_c, C*k*k]
        dW = dW_col.reshape(self.W.shape)
        db = np.sum(grad_col, axis=0, keepdims=True)

        # dX via col2im
        dX_cols = grad_col @ W_col  # [N*H_out*W_out, C*k*k]
        dX_pad = self._col2im(
            dX_cols, X_pad.shape, self.col_k, self.col_i, self.col_j, batch, H_out, W_out
        )

        if p > 0:
            dX = dX_pad[:, :, p:p + H, p:p + W]
        else:
            dX = dX_pad

        self.grads['W'] = dW
        self.grads['b'] = db
        return dX

    @staticmethod
    def _get_im2col_indices(channels, H_pad, W_pad, kernel_size, stride, out_h, out_w):
        k = kernel_size
        i0 = np.repeat(np.arange(k), k)
        i0 = np.tile(i0, channels)
        i1 = stride * np.repeat(np.arange(out_h), out_w)

        j0 = np.tile(np.arange(k), k)
        j0 = np.tile(j0, channels)
        j1 = stride * np.tile(np.arange(out_w), out_h)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        c = np.repeat(np.arange(channels), k * k).reshape(-1, 1)
        return c, i, j

    @staticmethod
    def _col2im(cols, X_pad_shape, c_idx, i_idx, j_idx, batch, out_h, out_w):
        dX_pad = np.zeros(X_pad_shape, dtype=cols.dtype)
        cols_reshaped = cols.reshape(batch, out_h * out_w, -1).transpose(0, 2, 1)
        np.add.at(
            dX_pad,
            (
                np.arange(batch)[:, None, None],
                c_idx[None, :, :],
                i_idx[None, :, :],
                j_idx[None, :, :],
            ),
            cols_reshaped,
        )
        return dX_pad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class MaxPool2d(Layer):
    """
    Non-overlapping max pooling (typical: kernel=2, stride=2).
    Input: [N, C, H, W] with H, W divisible by kernel size.
    """
    def __init__(self, kernel_size=2, stride=2) -> None:
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.input_shape = None
        self.mask = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        assert (kh, kw) == (sh, sw), "MaxPool2d: only equal stride to kernel is supported."
        n, c, h, w = X.shape
        assert h % kh == 0 and w % kw == 0, f"MaxPool2d: spatial size ({h},{w}) not divisible by kernel {kh}x{kw}."
        oh, ow = h // kh, w // kw
        x_view = X.reshape(n, c, oh, kh, ow, kw)
        out = x_view.max(axis=(3, 5))
        self.input_shape = X.shape
        self.mask = (x_view == out[:, :, :, np.newaxis, :, np.newaxis])
        return out

    def backward(self, grad):
        assert self.mask is not None and self.input_shape is not None
        kh, kw = self.kernel_size
        n, c, oh, ow = grad.shape
        dout = grad[:, :, :, np.newaxis, :, np.newaxis]
        denom = np.maximum(self.mask.sum(axis=(3, 5), keepdims=True), 1e-12)
        d_in = self.mask * (dout / denom)
        return d_in.reshape(self.input_shape)


class Flatten(Layer):
    """Flatten [N, C, H, W] -> [N, C*H*W]."""
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert X.ndim == 4
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        assert self.input_shape is not None
        assert grad.ndim == 2 and grad.shape[0] == self.input_shape[0]
        return grad.reshape(self.input_shape)


class GlobalAvgPool2D(Layer):
    """
    Global average pooling over spatial dimensions.
    Input: [N, C, H, W] -> Output: [N, C]
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert X.ndim == 4
        self.input = X
        return X.mean(axis=(2, 3))

    def backward(self, grad):
        assert self.input is not None
        assert grad.ndim == 2
        n, c, h, w = self.input.shape
        assert grad.shape == (n, c)
        scale = 1.0 / (h * w)
        return np.broadcast_to(
            grad[:, :, np.newaxis, np.newaxis] * scale,
            (n, c, h, w),
        ).copy()


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.labels = None
        self.probs = None
        self.grads = None
        self.has_softmax = True
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        assert predicts.ndim == 2
        assert labels.ndim == 1
        assert predicts.shape[0] == labels.shape[0]

        self.labels = labels.astype(np.int64)
        probs = softmax(predicts) if self.has_softmax else predicts
        self.probs = probs

        batch_size = predicts.shape[0]
        chosen = probs[np.arange(batch_size), self.labels]
        chosen = np.clip(chosen, 1e-12, 1.0)
        loss = -np.mean(np.log(chosen))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        assert self.probs is not None and self.labels is not None
        batch_size = self.probs.shape[0]
        one_hot = np.zeros_like(self.probs)
        one_hot[np.arange(batch_size), self.labels] = 1.0
        self.grads = (self.probs - one_hot) / batch_size
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition