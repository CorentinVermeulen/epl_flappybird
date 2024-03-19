import torch

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_cached())
