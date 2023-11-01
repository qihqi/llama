from torch_xla import stablehlo
from torch_xla import tf_saved_model_integration


def create_empty_cache(model_args, mode):
    pass

def load_stablehlo_model(path, mode):
    shlo = stablehlo.StableHLOGraphModule.load(path)
    tfmodel = tf_saved_model_integration.make_tf_function(path)
    if mode == 'jax':
        from jax.experimental import jax2tf
        return jax2tf.call_tf(tfmodel)
    return tfmodel

