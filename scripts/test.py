import torch
import ctranslate2

print(f"PyTorch sees GPU: {torch.cuda.is_available()}")
print(f"CTranslate2 supports CUDA: {ctranslate2.get_cuda_device_count() > 0}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")