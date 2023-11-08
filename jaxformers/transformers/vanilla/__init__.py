"""
A Vanilla Transformer

![Overview](./src/vanilla_transformer/full.png "Transformer")

"""


# Fake devices only on CPU
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'