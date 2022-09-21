import math
import torch
from utils.compressor.compressor import Compressor


class TopkCompressor(Compressor):
    """ Compressor for federated communication
        Top-k gradient or weights selection
        Args:
            compress_ratio (float): compress ratio
    """

    def __init__(self, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.index_dtype = torch.int64
        self.value_dtype = torch.float32

    def compress_tensor(self, tensor):
        """compress tensor into (values, indices)
        Args:
            tensor (torch.Tensor): tensor
        Returns:
            tuple: (values, indices)
        """
        if torch.is_tensor(tensor):
            tensor = tensor.detach()
        else:
            raise TypeError(
                "Invalid type error, expecting {}, but get {}".format(
                    torch.Tensor, type(tensor)))

        numel = tensor.numel()
        top_k_samples = int(math.ceil(numel * self.compress_ratio))

        tensor = tensor.view(-1)
        importance = tensor.abs()

        _, indices = torch.topk(importance,
                                top_k_samples,
                                0,
                                largest=True,
                                sorted=False)
        values = tensor[indices]

        values = values.to(dtype=self.value_dtype)
        indices = indices.to(dtype=self.index_dtype)

        return values, indices

    def decompress_tensor(self, values, indices, shape):
        """decompress tensor"""
        de_tensor = torch.zeros(size=shape, dtype=self.value_dtype).view(-1)
        de_tensor = de_tensor.index_put_([indices], values,
                                         accumulate=True).view(shape)
        return de_tensor

    def compress(self, parameters):
        """compress model
        Args:
            parameters (torch.nn.module): PyTorch module.
        Returns:
            tuple: list(values) and list(indices).
        """
        values_list = []
        indices_list = []
        for param in parameters:
            values, indices = self.compress_tensor(param)
            values_list.append(values)
            indices_list.append(indices)

        return values_list, indices_list

    def decompress(self, shape_list, values_list, indices_list):
        """decompress model
        Args:
            shape_list (list[tuple]): The shape of every corresponding tensor.
            values_list (list[torch.Tensor]): list(values).
            indices_list (list[torch.Tensor]): list(indices).
        """
        parameters_layer_list = []
        for shape, values, indices in zip(shape_list, values_list,
                                          indices_list):
            de_tensor = self.decompress_tensor(values, indices, shape)
            parameters_layer_list.append(de_tensor.view(-1))

        parameters = torch.cat(parameters_layer_list)

        return parameters


class QSGDCompressor(Compressor):
    """Quantization compressor.

    A implementation for paper https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf.

    Alistarh, Dan, et al. "QSGD: Communication-efficient SGD via gradient quantization and encoding." Advances in Neural Information Processing Systems 30 (2017): 1709-1720.
    Thanks to git repo: https://github.com/xinyandai/gradient-quantization

    Args:
        n_bit (int): the bits num for quantization. Bigger n_bit comes with better compress precision but more communication consumption.
        random (bool, optional): Carry bit with probability. Defaults to True.
        cuda (bool, optional): use GPU. Defaults to False.
    """

    def __init__(self, n_bit, random=True, cuda=False):
        super().__init__()
        self.random = random
        self.bit = n_bit
        self.cuda = cuda
        self.s = 2 ** self.bit
        self.code_dtype = torch.int32

    def compress(self, tensor):
        """Compress a tensor with quantization
        Args:
            vec ([type]): [description]
        Returns:
            norm (torch.Tensor): The normalization number.
            signs (torch.Tensor): Tensor that indicates the sign of coresponding number.
            quantized_intervals (torch.Tensor): Quantized tensor that each item in [0, 2**n_bit -1].
        """
        shape = tensor.shape
        vec = tensor.view(-1)
        # norm = torch.norm(vec, dim=1, keepdim=True)
        norm = torch.max(torch.abs(vec), dim=0, keepdim=True)[0]
        normalized_vec = vec / norm

        scaled_vec = torch.abs(normalized_vec) * self.s
        l = torch.clamp(scaled_vec, 0, self.s - 1).type(self.code_dtype)

        if self.random:
            # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
            probabilities = scaled_vec - l.type(torch.float32)
            r = torch.rand(l.size())
            if self.cuda:
                r = r.cuda()
            l[:] += (probabilities > r).type(self.code_dtype)

        signs = torch.sign(vec) > 0
        return [norm, signs.view(shape), l.view(shape)]

    def decompress(self, signature):
        """Decompress tensor
        Args:
            signature (list): [norm, signs, quantized_intervals], returned by :func:``compress``.
        Returns:
            torch.Tensor: Raw tensor represented by signature.
        """
        [norm, signs, l] = signature
        assert l.shape == signs.shape
        shape = l.shape
        scaled_vec = l.type(
            torch.float32) * (2 * signs.type(torch.float32) - 1)
        compressed = (scaled_vec.view((-1))) * norm / self.s
        return compressed.view(shape)
