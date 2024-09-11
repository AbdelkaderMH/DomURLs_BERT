import torch

def trace_jit_model(model, loader):
    """
    Trace a PyTorch model using JIT.

    Args:
        model: The PyTorch model to trace.
        loader: The data loader to provide input for tracing.

    Returns:
        torch.jit.ScriptModule: The traced JIT model.
    """
    
    model = model.eval().cpu()
    for batch in loader:
        if len(batch) == 2:
            example_input, y1 = batch
        else:
            example_input, y1, y2 = batch
        break
    
    scripted_model = torch.jit.script(model) #torch.jit.trace(model, example_inputs=example_input)
    return scripted_model