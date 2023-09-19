import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
from torch_xla import tf_saved_model_integration, stablehlo
import numpy

class M(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(1000, 10000)
        self.b = torch.nn.Linear(10000, 10)


    def forward(self, x):
        x = self.a(x)
        return self.b(x)


def main():
    import torch_xla.runtime
    torch_xla.runtime.use_spmd()
    num_devices = 4
    output_path = '/mnt/hanq/linear_stablehlo'
    device_ids = numpy.arange(4)
    col_mesh = xs.Mesh(device_ids, (1, num_devices))
    row_mesh = xs.Mesh(device_ids, (num_devices, 1))

    device = xm.xla_device()

    model = M().to(device)
    x = torch.randn(10, 1000).to(device)
    args = (x, )
    print(model(*args))

    xs.mark_sharding(x, row_mesh, (0, None))
    stablehlo.save_torch_model_as_stablehlo(
            model, (x, ), output_path)


if __name__ == '__main__':
    main()

