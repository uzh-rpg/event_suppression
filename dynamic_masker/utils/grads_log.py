import torch
import wandb
from dynamic_masker.models.arch import ConvLayer, ResidualBlock, UpsampleConvLayer, TransposedConvLayer

def log_gradient_flow(model, step):
    """Logs gradient norms, zero gradients, and weight updates to Weights & Biases."""
    
    grad_norms = []
    zero_grad_layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

            # Log to wandb
            wandb.log({f"Grad Norm/{name}": grad_norm}, step=step)

            # Check for zero gradients
            if param.grad.abs().sum().item() == 0:
                zero_grad_layers.append(name)
        else:
            zero_grad_layers.append(name)
    
    if grad_norms:
        wandb.log({
            "Grad Norm/Min": min(grad_norms),
            "Grad Norm/Max": max(grad_norms),
            "Grad Norm/Mean": sum(grad_norms) / len(grad_norms),
        }, step=step)

    if zero_grad_layers:
        wandb.log({"Zero Grad Layers": len(zero_grad_layers)}, step=step)
        # print(f"⚠️ Warning: {len(zero_grad_layers)} layers have zero gradients:", zero_grad_layers)
        print(f"⚠️ Warning: {len(zero_grad_layers)} layers have zero gradients")
    

def log_activation_distributions(model, inputs, step):
    """Logs activations from key layers to Weights & Biases."""
    activations = {}

    def hook_fn(module, input, output):
        activations[module] = output.detach()

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (ConvLayer, ResidualBlock, UpsampleConvLayer, TransposedConvLayer)):
            hooks.append(module.register_forward_hook(hook_fn))

    _ = model(inputs)  # Forward pass to get activations

    for module, activation in activations.items():
        mean_act = activation.mean().item()
        std_act = activation.std().item()

        # Log to wandb
        wandb.log({
            f"Activation/{module} Mean": mean_act,
            f"Activation/{module} Std": std_act
        }, step=step)

    # Remove hooks after logging
    for hook in hooks:
        hook.remove()


def log_optimizer_state(optimizer, step):
    """Logs optimizer state (learning rate, weight updates) to Weights & Biases."""
    for i, param_group in enumerate(optimizer.param_groups):
        wandb.log({f"LR/Group_{i}": param_group["lr"]}, step=step)


def check_unused_layers(model, inputs):

    activations = {}
    def hook_fn(module, input, output):
        # If output is a tuple, extract the first tensor
        if isinstance(output, tuple):
            output = output[-1]  # Use last element if it's a tuple

        if isinstance(output, torch.Tensor):  
            activations[module] = output.detach()  

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    used_layers = list(activations.keys())
    all_layers = list(model.modules())

    unused_layers = [layer for layer in all_layers if layer not in used_layers]

    if unused_layers:
        print(f"⚠️ Unused Layers in Forward Pass: {[str(layer) for layer in unused_layers]}")
    else:
        print("✅ All layers are used in the forward pass.")


def debug_training_step(model, optimizer, step):
    """Runs all debugging checks and logs to Weights & Biases."""

    # Log everything
    log_gradient_flow(model, step)
    # log_activation_distributions(model, inputs, step)
    log_optimizer_state(optimizer, step)
    # check_unused_layers(model, inputs)
