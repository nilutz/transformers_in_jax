import optax


def create_custom_optimizer(learning_rate, weight_decay_rate, beta_1, beta_2, epsilon):
    optimizer_def = optax.chain(
        optax.scale_by_adam(b1=beta_1, b2=beta_2, eps=epsilon),
        optax.scale(-learning_rate),
        optax.scale_by_schedule(optax.linear_schedule(1.0)),
    )

    # Apply weight decay to all parameters except "LayerNorm", "layer_norm", and "bias"
    exclude_fn = (
        lambda path: "LayerNorm" in path or "layer_norm" in path or "bias" in path
    )
    optimizer = optax.tree_multistep(optimizer_def, exclude_fn=exclude_fn)

    return optimizer
