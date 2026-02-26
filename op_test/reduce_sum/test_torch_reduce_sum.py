import torch

# 使用 torch.compile 触发动态编译
@torch.compile
def my_sum(x):
    # 这就是你关注的 ATen 原生调用
    return torch.ops.aten.sum.default(x)

def main():
    # 构造与 JAX 对标的数据
    x = torch.randn(1024, dtype=torch.float16, device='cuda')
    
    # 预热并触发编译
    res = my_sum(x)
    print("Done.")

if __name__ == "__main__":
    main()