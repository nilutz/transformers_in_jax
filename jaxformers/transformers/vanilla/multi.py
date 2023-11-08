# The following is run in parallel on each host on a GPU cluster or TPU pod slice.
import jax
jax.distributed.initialize(coordinator_address="192.168.0.1:1234",
                           num_processes=2,
                           process_id=0)  # On GPU, see above for the necessary arguments.
print(jax.device_count())  # total number of accelerator devices in the cluster

print(jax.local_device_count())  # number of accelerator devices attached to this host

# The psum is performed over all mapped devices across the pod slice
xs = jax.numpy.ones(jax.local_device_count())
a = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
print(a)