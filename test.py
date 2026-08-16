from torch.nn import Module
a = Module()
print(a.call_super_init)

# class sophia:
#     name = "shuang"
#     def __init__(self):
#         self.name = "wu"
#         # super().__setattr__("name", "wu")
# a = sophia()
# print(sophia.name, a.name, sep='\n')
# print(a.__dict__)

# print(bool(None))

# import torch
# print(hasattr(torch._C, '_log_api_usage_once'), type(torch._C._log_api_usage_once), sep='\n')
# torch._C._log_api_usage_once("sophia_event")

# from dataclasses import dataclass
# @dataclass
# class sophia:
#     name = "shuang"
# a = sophia()
# print(sophia.name, a.name)
# sophia.name = "colorful clouds"
# print(sophia.name, a.name)
# a.name = "wu"
# print(sophia.name, a.name)
# sophia.name = "caiyun"
# print(sophia.name, a.name)


# print(5 // 2 * 2)
# print((32768 + 64 - 1) // 64 * 64)

# l = [1,2,3,4]
# print(l[-1])

# print(3 % 1)

# print(type((512, 0)))

# print(-5//2, 5//2)

# print(-(-512 // 4 // 128) * 128)

# print("l".upper())

# import dataclasses
# print(dataclasses.__file__)


# print(4/3, 283//128)

# import torch
# device = torch.device('cuda')
# token_bytes_path = "/home/shuang/.cache/nanochat/tokenizer/token_bytes.pt"
# with open(token_bytes_path, "rb") as f:
#     token_bytes = torch.load(f, map_location=device)
#     print(token_bytes.device)
# with open("/home/shuang/.cache/nanochat/tokenizer/tokenizer.pkl", 'rb') as f:
#     print(type(f))

# from nanochat.tokenizer import RustBPETokenizer
# from nanochat.common import print0
# tokenizer = RustBPETokenizer.from_directory("/home/shuang/.cache/nanochat/tokenizer")
# vocab_size = tokenizer.enc.n_vocab
# print0(f"Vocab size: {vocab_size:,}")

# import pickle
# with open('/home/shuang/.cache/nanochat/tokenizer/tokenizer.pkl', 'rb') as f:
#     header = f.read(10)
# print(header)

# import os
# a = "/home/shuang/sophia"
# b = os.path.join(a, "2026")
# c = os.path.join(b, "tokenizer.pkl")
# with open(c, 'rb') as f:
#     pass
# print(os.path.isfile(c))

# import os
# a = "/home/shuang/.sophia/2026/Aug"
# os.makedirs(a, exist_ok=False)

# from kernels import has_kernel, get_kernel
# hf_kernel = "varunneal/flash-attention-3"
# print(get_kernel(hf_kernel).flash_attn_interface)

# a = 'a'
# b = False
# c = a == 'b' or not b
# print(c)
# print((8, 5) >= (8, 6))

# import torch
# capability = torch.cuda.get_device_capability()
# print(capability)

# import os
# env = os.environ.get("NANOCHAT_DTYPE")
# print(env)

# print(1)
# import torch
# print(torch.backends.mps.is_available())

# import torch
# import time

# def measure_fp32_peak(device='cuda'):
#     """
#     通过大量矩阵乘法测量 GPU 的 FP32 峰值算力
#     """
#     # 检查 CUDA 是否可用
#     if not torch.cuda.is_available():
#         print("CUDA 不可用，请检查显卡驱动")
#         return None
    
#     device = torch.device(device)
#     print(f"使用设备: {torch.cuda.get_device_name(0)}")
#     print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
#     print("-" * 50)
    
#     # 设置矩阵大小
#     # 选这个尺寸是为了让矩阵足够大，能喂饱 GPU，但又不超过显存
#     # 对于 T600 的 4GB 显存，这个尺寸比较合适
#     size = 8192  # 8192 x 8192 的矩阵
    
#     print(f"测试矩阵大小: {size} x {size}")
#     print("正在准备数据...")
    
#     # 创建两个随机矩阵，使用 FP32 类型
#     # 让它们连续跑多次，取平均值
#     num_iterations = 20
#     warmup_iterations = 5
    
#     # 预热 GPU，让频率稳定下来
#     print("预热中...")
#     for _ in range(warmup_iterations):
#         a = torch.randn(size, size, device=device, dtype=torch.float32)
#         b = torch.randn(size, size, device=device, dtype=torch.float32)
#         c = torch.mm(a, b)
#         torch.cuda.synchronize()  # 确保运算完成
    
#     print("开始正式测试...")
    
#     # 记录开始时间
#     torch.cuda.reset_peak_memory_stats()
#     start_time = time.time()
    
#     # 正式测试
#     total_flops = 0
#     for i in range(num_iterations):
#         # 每次重新生成矩阵，避免缓存命中影响结果
#         a = torch.randn(size, size, device=device, dtype=torch.float32)
#         b = torch.randn(size, size, device=device, dtype=torch.float32)
        
#         # 执行矩阵乘法 C = A @ B
#         # 对于两个 n×n 的矩阵相乘，运算次数 = 2 * n^3 - n^2（或者近似为 2*n^3）
#         c = torch.mm(a, b)
        
#         # 同步等待运算完成
#         torch.cuda.synchronize()
        
#         # 计算本次运算的浮点操作数
#         # 矩阵乘法的 FLOPs = 2 * n^3（标准计算）
#         flops_per_matmul = 2 * size * size * size
#         total_flops += flops_per_matmul
        
#         # 每5次打印进度
#         if (i + 1) % 5 == 0:
#             print(f"完成 {i+1}/{num_iterations} 次")
    
#     end_time = time.time()
#     elapsed_time = end_time - start_time
    
#     # 计算平均算力（TFLOPS）
#     avg_tflops = (total_flops / elapsed_time) / 1e12
    
#     # 打印结果
#     print("=" * 50)
#     print(f"测试结果:")
#     print(f"  总运算次数: {total_flops:.2e} 次浮点运算")
#     print(f"  总耗时: {elapsed_time:.3f} 秒")
#     print(f"  平均算力: {avg_tflops:.3f} TFLOPS")
#     print(f"  理论峰值: 1.709 TFLOPS")
#     print(f"  效率: {avg_tflops / 1.709 * 100:.1f}%")
    
#     # 打印显存使用情况
#     max_memory = torch.cuda.max_memory_allocated() / 1024**3
#     print(f"  峰值显存占用: {max_memory:.2f} GB")
#     print("=" * 50)
    
#     return avg_tflops

# if __name__ == "__main__":
#     measure_fp32_peak()


# from nanochat.common import get_peak_flops
# import torch
# device =  torch.cuda.get_device_name(0)
# print(get_peak_flops("h100"))

# a = "abcdefg"
# print("cde" in a)

# import torch
# a = torch.cuda.get_device_name(1)
# print(type(a))
# print(a)

# a = lambda: None
# print(a)
# print(type(a))
# print(a())

# a = 1
# b = a == 5
# print(b)

# import logging

# logger = logging.getLogger(__name__)
# logger.info("I'm Sophia")
# logger.warning("Don't look back")

# print(logger)

# a = ('a', 'b', 'c')
# b = ['a', 'b', 'c']
# print(all(k in b for k in a))

# import torch
# import time

# size = 1
# a = torch.randn(size, size, device='cuda')

# print(a.dtype, a.element_size(), sep='\n')

# device = 'cuda'
# print(f"graphics card: {torch.cuda.get_device_name(0)}")
# print(f"total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# size = 2048
# a = torch.randn(size, size, device=device)
# b = torch.randn(size, size, device=device)

# def test_speed(a, b, name, repeat=200):
#     for _ in range(20):
#         torch.mm(a, b)
#     torch.cuda.synchronize()

#     start = time.time()
#     for _ in range(repeat):
#         torch.mm(a, b)
#     torch.cuda.synchronize()
#     elapsed = time.time() - start
#     print(f"{name:20s} | {elapsed:.4f} s")
#     return elapsed

# a_fp32 = a
# b_fp32 = b
# for _ in range(10):
#     t_fp32 = test_speed(a_fp32, b_fp32, "FP32 standard")
#     print(t_fp32)


# torch.set_float32_matmul_precision("high")
# t_high = test_speed(a_fp32, b_fp32, "FP32 high")
# torch.set_float32_matmul_precision("highest") 

# a_bf16 = a.bfloat16()
# b_bf16 = b.bfloat16()
# t_bf16 = test_speed(a_bf16, b_bf16, "bfloat16")




# print(torch.cuda.is_bf16_supported())
# print(f"cuDNN 可用: {cudnn.is_available()}")
# print(f"cuDNN 版本: {cudnn.version()}")
# print(f"cuDNN 启用: {cudnn.enabled}")
# print(f"cuDNN 确定性: {cudnn.deterministic}")
# print(f"cuDNN 基准测试: {cudnn.benchmark}")

# 查看 FP16 是否被支持（在某些旧版本中可能为 False）
# try:
#     print(f"cuDNN FP16 支持: {cudnn.is_acceptable(torch.randn(1, 1, 1, 1, device='cuda').half())}")
# except:
#     print("cuDNN FP16 支持: 无法检测")


# print(f"PyTorch 版本: {torch.__version__}")
# print(f"CUDA 版本: {torch.version.cuda}")
# print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
# print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# import torch
# import time
# torch.set_float32_matmul_precision("high")

# a = torch.rand(4096, 4096, device='cuda')
# b = torch.rand(4096, 4096, device='cuda')

# for _ in range(10): torch.mm(a, b)
# torch.cuda.synchronize()

# start = time.time()
# for _ in range(100): torch.mm(a, b)
# torch.cuda.synchronize()
# print(f"time cost: {time.time() - start:.3f} s")


# import torch
# print(torch.cuda.is_available())

# print(torch.cuda.manual_seed(42))
# print(torch.rand(3, device='cuda'))
# print(torch.rand(3, device='cuda'))
# print(torch.rand(3, device='cuda'))
# print(torch.cuda.manual_seed(42))
# print(torch.rand(3, device='cuda'))
# print(torch.rand(3, device='cuda'))
# print(torch.rand(3, device='cuda'))

# text = """
#   █████████                     █████       ███           
#  ███░░░░░███                   ░░███       ░░░            
# ░███    ░░░   ██████  ████████  ░███████   ████   ██████  
# ░░█████████  ███░░███░░███░░███ ░███░░███ ░░███  ░░░░░███ 
#  ░░░░░░░░███░███ ░███ ░███ ░███ ░███ ░███  ░███   ███████ 
#  ███    ░███░███ ░███ ░███ ░███ ░███ ░███  ░███  ███░░███ 
# ░░█████████ ░░██████  ░███████  ████ █████ █████░░████████
#  ░░░░░░░░░   ░░░░░░   ░███░░░  ░░░░ ░░░░░ ░░░░░  ░░░░░░░░ 
#                       ░███                                
#                       █████                               
#                      ░░░░░                                
# """
# print(text)

# from nanochat.common import autodetect_device_type, compute_init
# device_type = autodetect_device_type()
# print(device_type, compute_init(device_type))

# import argparse
# parser = argparse.ArgumentParser(description="Pretrain base model")
# output =parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# #args = parser.parse_args()
# print(output)
# output = parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
# print(output)
# print(parser)
# args2 = parser.parse_args()
# print(args2.model_tag)

# import os
# print("PYTORCH_ALLOC_CONF" in os.environ.keys())