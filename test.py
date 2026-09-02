# initiate the model
from nanochat.gpt import GPTConfig, GPT, norm
import torch
def build_model_meta(depth):
    base_dim = depth * 64
    model_dim = ((base_dim + 128 - 1) // 128) * 128
    num_heads = model_dim // 128
    config = GPTConfig(
        sequence_len=512, vocab_size=32768,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern="L",
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta
model = build_model_meta(4)
model.to_empty(device=torch.device("cuda")) 
model.init_weights()

# load tokenizer
from nanochat.tokenizer import get_tokenizer
tokenizer = get_tokenizer()

# generate input and target
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, 1, 512, split="train", device=torch.device("cuda"), resume_state_dict=None)
idx, idy, dataloader_state_dict = next(train_loader) 

# forward and backward
x = model.transformer.wte(idx)
print(x.grad_fn)
print(x.grad_fn.next_functions)

# x = x.sum()
# x.backward()
# print(model.transformer.wte.weight.grad[32,:])
# print((model.transformer.wte.weight.grad>0).any())








# # first eval
# from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
# build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, 1, 512, split="val", device=torch.device("cuda"))
# batch_iter = build_val_loader()
# total_nats = torch.tensor(0.0, dtype=torch.float32, device=torch.device("cuda"))
# total_bytes = torch.tensor(0, dtype=torch.int64, device=torch.device("cuda"))
# model.eval()
# # loop
# for _ in range(1):
#     x, y = next(batch_iter)
#     loss2d = model(x, y, loss_reduction='none')
#     y = y.view(-1)
#     num_bytes2d = token_bytes[y]
#     total_nats += (loss2d * (num_bytes2d > 0)).sum()
#     total_bytes += num_bytes2d.sum()
# total_nats = total_nats.item()
# total_bytes = total_bytes.item()
# import math
# bpb = total_nats / (math.log(2) * total_bytes)
# min_val_bpb = bpb
# model.train()

# #
# synchronize = torch.cuda.synchronize
# synchronize()
# import time
# t0 = time.time()


# x = model.transformer.wte(idx)
# print(x[:, :-1].shape)
# print(model.smear_gate.weight.shape)
# y = model.smear_gate(x[:, 1:, :24])
# print(y.shape)
# gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
# print(gate.shape)



# print(model.transformer.wte.weight.grad)

# loss = model(idx, idy)
# loss.backward()

# print((model.transformer.wte.weight.grad>0).any())

# print(model.smear_lambda.grad)

# print(model.smear_lambda.grad)

# print(model.transformer.h[0].attn.c_proj.weight.grad)





# import torch
# row_buffer = torch.empty((1, 513), dtype=torch.long)
# print(row_buffer.requires_grad)






# print(x.requires_grad, y.requires_grad, sep='\n')

# print(x.grad, y.grad, sep='\n')

# loss.backward()

# print(x.grad, y.grad, sep='\n')






# print(x.grad_fn, y.grad_fn, sep='\n')

# print(torch.autograd.variable.Variable._execution_engine.run_backward((loss,), 
#                                                                 (torch.ones_like(loss, memory_format=torch.preserve_format),),
#                                                                 False, False, (), True, True))

# import logging
# log = logging.getLogger(__name__)

# print(log.getEffectiveLevel())
# print(logging.DEBUG)

# print(torch.autograd.graph._engine_run_backward((loss,), (torch.ones_like(loss, memory_format=torch.preserve_format),),
#                                                 False, False, (), True, True))

# print(torch.autograd._make_grads((loss,), (None,), False))

# print(torch.ones_like(loss, memory_format=torch.preserve_format))

# print(loss.dtype.is_floating_point)

# import torch
# a = torch.empty(2,2,3,5)
# print(a.numel())


# print(loss.numel())


# print(loss.requires_grad)

# import torch
# print(isinstance(None, torch.Tensor))


# print(loss.shape)

# print(loss.is_nested)

# print(isinstance(loss, torch.autograd.graph.GradientEdge))

# print(zip((loss,), (None,)))

# a = (loss,)
# print(torch.autograd._tensor_or_tensors_to_tuple(None, len(a)))

# torch.autograd.backward(loss)


# print(loss, x, y, sep='\n')
# print(len(loss))

# a = torch.tensor(0)
# print(a)


# a = (None,)
# print(a * 2)


# a = None
# print(id(a))
# import sys
# print(sys.getsizeof(a))

# a = tuple[torch.Tensor]
# print(type(a))

# from torch.overrides import is_tensor_like
# print(is_tensor_like(loss))
# print(torch._C._are_functorch_transforms_active())
# from torch.overrides import has_torch_function_unary
# print(has_torch_function_unary(loss))
# loss.backward()


# print(num_bytes2d[-1])
# x = x.view(-1)
# num_bytes2d = token_bytes[x]
# print(num_bytes2d[0])

# import torch

# a = torch.tensor([1.33,2.44,3.55])
# b = torch.tensor([False,True,True])
# print(a*b)

# cos = model.cos[:, :512]
# sin = model.sin[:, :512]
# # transfer token id sequence to embedding sequence
# x = model.transformer.wte(idx)
# x = norm(x)
# x0 = x
# # block 0
# x = model.resid_lambdas[0] * x + model.x0_lambdas[0] * x0
# # block 0 CausalSelfAttention
# q = model.transformer.h[0].attn.c_q(norm(x)).view(1,512,2,128)
# k = model.transformer.h[0].attn.c_k(norm(x)).view(1,512,2,128)
# v = model.transformer.h[0].attn.c_v(norm(x)).view(1,512,2,128)
# x1, x2, x3, x4 = q[..., :64], q[..., 64:], k[..., :64], k[..., 64:]
# y1, y3 = x1 * cos + x2 * sin, x3 * cos + x4 * sin 
# y2, y4 = x1 * (-sin) + x2 * cos, x3 * (-sin) + x4 * cos
# q, k = torch.cat([y1, y2], 3), torch.cat([y3, y4], 3)
# q, k = norm(q), norm(k)
# q, k = q * 1.2, k * 1.2
# q, k, v = q.transpose(1, 2), k.transpose(1,2), v.transpose(1, 2)
# import torch.nn.functional as F
# y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=False)
# y.transpose(1,2)
# y = y.contiguous().view(1, 512, -1)
# y = model.transformer.h[0].attn.c_proj(y)
# x += y
# # block 0 MLP
# x += model.transformer.h[0].mlp.c_proj(F.relu(model.transformer.h[0].mlp.c_fc(norm(x))).square())
# # block 1
# x = model.resid_lambdas[1] * x + model.x0_lambdas[1] * x0
# ve = model.value_embeds["1"](idx)
# # block 1 CausalSelfAttention
# q = model.transformer.h[1].attn.c_q(norm(x)).view(1,512,2,128)
# k = model.transformer.h[1].attn.c_k(norm(x)).view(1,512,2,128)
# v = model.transformer.h[1].attn.c_v(norm(x)).view(1,512,2,128)
# ve = ve.view(1, 512, 2, 128)
# gate = 3 * torch.sigmoid(model.transformer.h[1].attn.ve_gate(x[..., :12]))
# v = v + gate.unsqueeze(-1) * ve
# x1, x2, x3, x4 = q[..., :64], q[..., 64:], k[..., :64], k[..., 64:]
# y1, y3 = x1 * cos + x2 * sin, x3 * cos + x4 * sin 
# y2, y4 = x1 * (-sin) + x2 * cos, x3 * (-sin) + x4 * cos
# q, k = torch.cat([y1, y2], 3), torch.cat([y3, y4], 3)
# q, k = norm(q), norm(k)
# q, k = q * 1.2, k * 1.2
# q, k, v = q.transpose(1, 2), k.transpose(1,2), v.transpose(1, 2)
# y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=False)
# y.transpose(1,2)
# y = y.contiguous().view(1, 512, -1)
# y = model.transformer.h[1].attn.c_proj(y)
# x += y
# # block 1 MLP
# x += model.transformer.h[1].mlp.c_proj(F.relu(model.transformer.h[1].mlp.c_fc(norm(x))).square())
# # block 2
# x = model.resid_lambdas[2] * x + model.x0_lambdas[2] * x0
# # block 2 CausalSelfAttention
# q = model.transformer.h[2].attn.c_q(norm(x)).view(1,512,2,128)
# k = model.transformer.h[2].attn.c_k(norm(x)).view(1,512,2,128)
# v = model.transformer.h[2].attn.c_v(norm(x)).view(1,512,2,128)
# x1, x2, x3, x4 = q[..., :64], q[..., 64:], k[..., :64], k[..., 64:]
# y1, y3 = x1 * cos + x2 * sin, x3 * cos + x4 * sin 
# y2, y4 = x1 * (-sin) + x2 * cos, x3 * (-sin) + x4 * cos
# q, k = torch.cat([y1, y2], 3), torch.cat([y3, y4], 3)
# q, k = norm(q), norm(k)
# q, k = q * 1.2, k * 1.2
# q, k, v = q.transpose(1, 2), k.transpose(1,2), v.transpose(1, 2)
# y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=False)
# y.transpose(1,2)
# y = y.contiguous().view(1, 512, -1)
# y = model.transformer.h[2].attn.c_proj(y)
# x += y
# # block 2 MLP
# x += model.transformer.h[2].mlp.c_proj(F.relu(model.transformer.h[2].mlp.c_fc(norm(x))).square())
# x_backout = x
# # block 3
# x = model.resid_lambdas[3] * x + model.x0_lambdas[3] * x0
# ve = model.value_embeds["3"](idx)
# # block 3 CausalSelfAttention
# q = model.transformer.h[1].attn.c_q(norm(x)).view(1,512,2,128)
# k = model.transformer.h[1].attn.c_k(norm(x)).view(1,512,2,128)
# v = model.transformer.h[1].attn.c_v(norm(x)).view(1,512,2,128)
# ve = ve.view(1, 512, 2, 128)
# gate = 3 * torch.sigmoid(model.transformer.h[3].attn.ve_gate(x[..., :12]))
# v = v + gate.unsqueeze(-1) * ve
# x1, x2, x3, x4 = q[..., :64], q[..., 64:], k[..., :64], k[..., 64:]
# y1, y3 = x1 * cos + x2 * sin, x3 * cos + x4 * sin 
# y2, y4 = x1 * (-sin) + x2 * cos, x3 * (-sin) + x4 * cos
# q, k = torch.cat([y1, y2], 3), torch.cat([y3, y4], 3)
# q, k = norm(q), norm(k)
# q, k = q * 1.2, k * 1.2
# q, k, v = q.transpose(1, 2), k.transpose(1,2), v.transpose(1, 2)
# y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=False)
# y.transpose(1,2)
# y = y.contiguous().view(1, 512, -1)
# y = model.transformer.h[3].attn.c_proj(y)
# x += y
# # block 3 MLP
# x += model.transformer.h[3].mlp.c_proj(F.relu(model.transformer.h[3].mlp.c_fc(norm(x))).square())
# # End Block
# x = x - model.backout_lambda * x_backout
# x = norm(x)
# # Logits
# logits = model.lm_head(x)
# logits = 15 * torch.tanh(logits / 15)
# loss = torch._C._nn.cross_entropy_loss(logits.view(-1, logits.size(-1)), idy.view(-1), None, 0, -1, 0.0)
# # Compare two results
# loss2d = model(idx, idy, loss_reduction='none')
# print(loss2d - loss)



# import torch
# print(torch.sigmoid(torch.tensor([8,9,10])))

# print(torch.tensor([2,3]).square())

# print(F.relu(torch.tensor([2,2,-1,3,2,1.1,-9,3])))

# a = q[0,0,1,:]
# b = k[0,0,:2,:]
# c = torch.mv(b,a)
# c /= 128**0.5
# w = torch.softmax(c, dim=0)
# o = torch.mv(v[0,0,:2,:].transpose(0,1), w)
# print(o - r[0,0,1,:])

# print(cos.shape, sin.shape, q.shape, x1.shape, x2.shape, y1.shape, sep='\n')

# from nanochat.gpt import norm
# import torch
# a = torch.empty(2,2,3,5)
# a = a.view(2, 2, -1)
# print(a.shape)
# a = torch.tensor([[1.,1.,1.],[2.,2.,2.]])
# a = norm(a)
# print(a)

# a = torch.tensor([[1,2,3]])
# b = torch.tensor([[1,2,3], [4,5,6]])
# c = a * b
# print(c)



# from torch.nn import Linear
# import torch
# a = torch.arange(6)
# b = a.view(2,3)
# print(a,b,sep='\n')
# a[5] = 6
# print(a,b,sep='\n')
# a = Linear(2,2,bias=False)
# print(a.weight)
# b = torch.tensor([[[1.,0.],[1.,0.]]])
# print(a(b))

# print(model.transformer.h[0].attn.c_q)

# print(model.resid_lambdas, model.x0_lambdas, sep='\n')
# print(model.value_embeds)
# print(model.smear_gate(x[:, 1:, :24]))
# print(torch.sigmoid(model.smear_gate(x[:, 1:, :24])).size())
# print(model.smear_lambda)
# gate = model.smear_lambda * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
# print(gate.size(), x[:, :-1].size(),sep='\n')
# print(gate * x[:, :-1])
# x = torch.cat([x[:, :1], x[:, 1:]],dim=1)
# print(x.size())
# a = torch.empty(2,2,3)
# print(a)
# print(a[:,:,0])
# print(a[:,:,:1])
# b = torch.cat([a[:,:,:1],a[:,:,1:]],dim=2)
# print(a == b)
# print(x,x[:, :1], x[:, 1:],sep='\n')
# x=torch.rms_norm(x, [256])
# x = norm(x)
# print(torch.rms_norm(x, [256]) == norm(x))
# print(sum(_**2 for _ in x[0,0,:]))
# print(model.transformer.wte.weight, model.transformer.wte.padding_idx, model.transformer.wte.max_norm, 
#       model.transformer.wte.norm_type, model.transformer.wte.scale_grad_by_freq, model.transformer.wte.sparse, sep='\n')
# print(torch.embedding(model.transformer.wte.weight, x, -1, False, False) == model.transformer.wte(x))
# print(model.transformer.wte(torch.arange(1,device=torch.device("cuda"))) == model.transformer.wte.weight[0,:])
# print(model.transformer.wte(torch.tensor([32767],device=torch.device("cuda"))))
# print(model.transformer.wte.padding_idx, model.transformer.wte.max_norm, sep='\n')
# from torch.overrides import has_torch_function_variadic
# print(has_torch_function_variadic(x, model.transformer.wte.weight))

# print(model.cos)
# print(model.cos.size())
# print(model.cos[:,0:512])

# import torch
# a = torch.empty(4,4,4,4)
# c = torch.empty(4,4,4,4)
# print(a.shape)
# b = a[:,0:2], c[:,0:2]
# print(b)
# b = a[:,:,0:2]
# print(b.shape)
# b = a[:,:,:,0:2]
# print(b.shape)
# a = torch.outer(torch.arange(4), torch.arange(3))
# print(a)
# a = a[None, :, None, :]
# print(a)
# b = a[:,2]
# print(b)

# print(torch.empty(2,2).size())
# print(torch.arange(2))
# for name, module in model.named_modules():
#     print(type(module).__name__)

# print(model.eval() is model)




# print(x,y,sep='\n')
# print(not (model._backward_hooks or model._backward_pre_hooks or model._forward_hooks or model._forward_pre_hooks))

# print(torch._C._get_tracing_state())
# loss2d = model(x, y, loss_reduction='none')
# print(loss2d)
# b = build_val_loader()
# print(a,b,sep='\n')


# d12 = build_model_meta(12)

# def get_scaling_params(m):
#     params_counts = m.num_scaling_params()
#     scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
#     return scaling_params
# target_tokens = int(12 * get_scaling_params(model))
# D_REF = int(12 * get_scaling_params(d12))
# print(D_REF)


# a = [1,2,3,4]
# print(a.pop(2))


# from nanochat.dataloader import _document_batches
# from nanochat.tokenizer import get_tokenizer
# tokenizer = get_tokenizer()
# bos_token = tokenizer.get_bos_token_id()
# batches = _document_batches("train", None, 128)
# doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
# print(len(doc_batch), doc_batch[1], pq_idx, rg_idx, epoch, sep='\n')
# token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)
# print(token_lists[1])
# print(tokenizer.decode(token_lists[1]))

# print(0 % 250)

# a = 4
# b = 3
# c = 2
# print(a or b % c == 0)


# a = 1 * 512
# b = 2 * 256
# print(id(a) == id(b) == id(512))


# import torch
# gpu_buffer = torch.empty(2 * 1 * 2, dtype=torch.long, device=torch.device("cuda")) # on-device buffer
# inputs = gpu_buffer[:1 * 2].view(1, 2)
# targets = gpu_buffer[1 * 2:].view(1, 2)
# gpu_buffer[0] = 1
# print(gpu_buffer, inputs,targets,sep='\n')

# a = [1,2,3,4,5]
# print(a[:-1], a[1:], sep='\n')

# a = [[1,2], [1,2,3], [1]]
# b = min(range(len(a)), key=lambda i: len(a[i]))
# print(b)


# import torch
# a = torch.tensor([1,2,3], dtype=torch.long)
# print(a, a.dtype,sep='\n')

# import torch
# a = torch.empty(2*2*2)
# print(a)

# from nanochat.common import compute_init, autodetect_device_type
# device_type = autodetect_device_type()
# ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# print(device == "cuda")

# from nanochat.dataloader import _document_batches
# batches = _document_batches("train", None, 128)
# print(batches)

# import pyarrow.parquet as pq
# filepath = "/home/shuang/.cache/nanochat/base_data_climbmix/shard_00036.parquet"
# pf = pq.ParquetFile(filepath)
# # print(pf.num_row_groups)
# rg = pf.read_row_group(0)
# batch = rg.column('text').to_pylist()
# print(len(batch))

# from nanochat.common import get_base_dir
# import os
# base_dir = get_base_dir()
# data_dir = os.path.join(base_dir, "base_data_climbmix")
# print(data_dir)
# print(os.listdir(data_dir))
# parquet_files = sorted([
#     f for f in os.listdir(data_dir)
#     if f.endswith('.parquet') and not f.endswith('.tmp')
# ])
# print(parquet_files[:-1])
# print(parquet_files[-1:])
# print(parquet_files[-1])

# a = "11.sophia"
# print(a.endswith(".sophia"))

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--fp8", action="store_true", help="enable FP8 training")
# args = parser.parse_args()
# print(args.fp8)

# from nanochat.gpt import GPTConfig, GPT
# import torch
# config = GPTConfig(
#     sequence_len=512, vocab_size=32768,
#     n_layer=4, n_head=2, n_kv_head=2, n_embd=256,
#     window_pattern="L",
# )
# with torch.device("meta"):
#     model = GPT(config)

# class Sophia:
#     pass
# a = Sophia()
# print(a.__class__.__name__)

# from torch.optim import Optimizer
# hooked = getattr(Optimizer.step, "hooked", None)
# print(hooked)

# a = [1]
# a = list(a)
# print(a)

# import torch
# a = torch.empty(1,2)
# print(a.dtype)
# b = torch.empty(2,1)
# c = torch.empty(1,1)
# print(sorted({a.shape, b.shape, c.shape}))

# print(type({1,2}))
# import math
# weight_decay_scaled = 0.28 * math.sqrt(512 / 2**19 ) * (D_REF / target_tokens)    
# print(weight_decay_scaled)

# a = dict(kind='adamw')
# print(a["kind"])

# print(model._modules, model._parameters, sep='\n')
# model.to_empty(device=torch.device("cuda"))
# model.init_weights() 
# orig_model = model
# model = torch.compile(model, dynamic=False)

# batch_ratio = 512 / 524288
# batch_lr_scale = batch_ratio ** 0.5
# print(f"Scaling LRs by {batch_lr_scale:.4f}")
# param_counts = model.num_scaling_params()
# scaling_params = param_counts['transformer_matrices'] + param_counts['lm_head']
# print(scaling_params)
# target_tokens = int(12 * scaling_params)
# print(f"{target_tokens:,}", sep='\n')
# print(orig_model is model._orig_mod)
# print(orig_model.config is model.config)
# print(model.window_sizes)



# print(model.num_scaling_params())
# for p in model.transformer.h.parameters():
#     print(p.numel())

# print(model.num_scaling_params().items())

# print(dir(model))
# print(model.num_scaling_params())
# print(model.transformer.wte)
# print(model.get_compiler_config())
# print(model.dynamo_ctx)
# print(dir(model))
# print(model._orig_mod == orig_model)

# def print_module(module):
#     if not module._modules:
#         print(module)
#     for _ in module._modules:
#         print_module(module._modules[_])
# print_module(model)
# def print_parameters(module):
#     if not module._modules:
#         print(module._parameters)
#         return
#     for _ in module._modules:
#         print_parameters(module._modules[_])
#     print(module._parameters)
# print_parameters(model)
# def print_buffers(module):
#     if not module._modules:
#         print(module._buffers)
#         return
#     for _ in module._modules:
#         print_buffers(module._modules[_])
#     print(module._buffers)
# print_buffers(model)

    
# print(model.resid_lambdas.device)
# for _ in model.transformer.h[0].attn.c_q._modules:
#     print("m", _)
# for _ in model._parameters:
#     print(_)
# for _ in model.transformer.h[0].attn.c_q._buffers:
#     print("b", _)
# model.to_empty(device=torch.device("cuda"))
# device = torch.cuda.current_device()
# allocated = torch.cuda.memory_allocated(device)
# print(f"{allocated / 1024**2:.2f} MB")
# print(model)
# print(model.resid_lambdas.device)
# model.init_weights()
# for block in model.transformer.h:
#     print(block.attn.ve_gate)
# print(model.value_embeds)
# torch.nn.init.constant_(model.backout_lambda, 0.2)
# print(model.backout_lambda)
# weight = model.transformer.wte.weight
# print(weight.device, torch.overrides.has_torch_function_variadic(weight), sep='\n')
# torch.nn.init.normal_(weight, mean=0.0, std=0.8)
# print(model.transformer.h[0].attn.c_q.weight.size())
# n_embd = config.n_embd
# s = 3**0.5 * n_embd**-0.5
# torch.nn.init.uniform_(model.transformer.h[0].attn.c_q.weight, -s, s)
# print(model.transformer.h[0].attn.c_q.weight)
# torch.nn.init.zeros_(model.smear_lambda)
# print(model.smear_lambda)

# param = model.transformer.h[0].attn.c_q._parameters["weight"]
# print(param.requires_grad)
# with torch.no_grad():
#     param_applied = torch.empty_like(param, device=torch.device("cuda"))
# print(param_applied, id(param_applied), param_applied.data_ptr(), sep='\n')
# print(torch._has_compatible_shallow_copy_type(param, param_applied))
# print(torch.__future__.get_swap_module_params_on_conversion())
# print(hasattr(param_applied, "__tensor_flatten__"))
# from torch._subclasses.fake_tensor import FakeTensor
# print(isinstance(param, FakeTensor))
# buff = model.transformer.h[0].attn.c_q._buffers
# print(buff)

# f = lambda t: t+1
# print(f(1))


# from nanochat.gpt import GPTConfig
# config = GPTConfig()
# print(config.__dict__)

# a = [("1",1), ("2",2)]
# print(dict(a))


# from dataclasses import dataclass
# @dataclass
# class sophia:
#     pass
# print(hasattr(sophia, '__dataclass_fields__'))


# from nanochat.gpt import GPTConfig
# config = GPTConfig()
# _FIELDS = '__dataclass_fields__'
# fields = getattr(config, _FIELDS)
# print(fields)
# print(tuple(f.name for f in fields.values()))
# print(getattr(config, 'sequence_len'))

# import torch
# a = torch.arange(3)
# b = torch.arange(2)
# freqs = torch.outer(a,b)
# # print(freqs)
# cos, sin = freqs.cos(), freqs.sin()
# # print(cos, sin, sep='\n')
# print(hasattr(cos, "__torch_function__"))

# import torch
# channel_range = torch.arange(0, 128, 2, dtype=torch.float32, device="cpu")
# inv_freq = 1.0 / (100000 ** (channel_range / 128))
# print(inv_freq)

# import torch
# with torch.device("meta"):
#     channel_range = torch.arange(0, 128, 2, dtype=torch.float32, device="meta")
#     print(channel_range)

# import torch
# a = torch.empty(1)
# print(a.dtype)

# import torch
# from torch.nn.parameter import Parameter
# a = torch.empty(3,3)
# b = Parameter(a)
# print(b.device)
# print(b)
# with torch.device("meta"):
#     a = torch.empty(3,3)
#     b = Parameter(a)
#     print(b.device)
#     print(b)

# from torch.nn.modules.sparse import Embedding
# from torch.nn.modules.container import ModuleDict
# a = ModuleDict({"sophia": Embedding(3, 3)})
# print(a._modules)

# from torch.nn.modules.container import ModuleDict
# a = ModuleDict({"a":1, "b":2})
# print(a.a)

# import torch
# print(torch.ones(4), torch.zeros(1), sep='\n')

# a = {"wte": 1, "h": 2}
# print(a.items())

# from collections import abc as container_abcs, OrderedDict
# from torch.nn.modules.container import ModuleDict
# a = {}
# print(isinstance(a, (OrderedDict, ModuleDict, container_abcs.Mapping)))

# class sophia:
#     pass
# a = sophia()
# print(a._parameter)

# import math
# import torch
# from torch.nn.parameter import Parameter
# from torch.nn import init
# a = torch.empty(3,3)
# b = Parameter(a)
# print(b)
# init.kaiming_normal_(b, a=math.sqrt(5))
# print(b)

# import torch
# from torch.nn.parameter import Parameter
# a = torch.empty(3,3)
# print(a.requires_grad)
# print(a.uniform_(0,1))
# b = Parameter(a)
# print(b.requires_grad)
# print(b)

# import torch
# print(torch._jit_internal.is_scripting())

# import math
# param = math.sqrt(5)
# print(not isinstance(param, bool)
# and isinstance(param, int)
# or isinstance(param, float)
# )

# import torch
# m = torch.empty(2,3)
# print(m.size(1))
# print(m.dim())
# print(m)
# print(2 in m.shape)

# a = [1,2,3]
# print(a[-1])

# for i in range(10):
#     print(i % 1)
# print(5//2, -5//2)

# from dataclasses import dataclass
# @dataclass
# class sophia:
#     # name = "shuang"
#     name: str = "shuang"
# a = sophia()
# print(a.__dict__)

# from nanochat.gpt import GPTConfig
# config = GPTConfig()
# print(config.__class__, config.__dict__, sep='\n')

# class sophia:
#     name = "shuang"
#     def __init__(self):
#         self.__setattr__("project", "nanochat")
# a = sophia()
# print(a.__class__)
# print(a.__dict__)

# class sophia:
#     name = "shuang"
#     def __init__(self):
#         pass
# a = sophia()
# print(a.__dict__)
# print(id(a.name), id(sophia.name))
# print(a.__dict__)
# a.name = "wu"
# print(sophia.name, a.name, a.__dict__, sep='\n')

# from torch.nn import Module
# a = Module()
# print(a.call_super_init)

# class sophia:
#     name = "shuang"
#     def __init__(self):
#         pass
#         # self.name = "wu"
#         # super().__setattr__("name", "wu")
# a = sophia()
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
# print(a.name)
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
# print(tokenizer.encode("I'm Sophia."))
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