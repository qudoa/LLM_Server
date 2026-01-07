"""用于 NCU profiling 的 append_paged_kv_cache 测试脚本"""
import torch
import flashinfer
import torch.cuda.profiler as profiler

# --- 配置参数 ---
num_kv_heads = 4        # KV 头数（GQA: 每个 KV 头可以服务多个 Q 头）
head_dim = 128          # 每个头的维度
page_size = 16          # 每页存储的 token 数（类似 OS 的页大小）
device = "cuda:0"
dtype = torch.float16

# --- 场景设置: 已有 4096 个 token，追加第 4097 个 ---
current_seq_len = 4096  # 当前已缓存的序列长度
nnz_kv = 256              # 要追加的 token 数量（通常是 1）【可改: 1, 16, 32, 64, 128, 256】

# 计算需要的总页数
# 4096 个 token 需要 256 页（每页 16 个），追加 1 个后需要 257 页
total_seq_len = current_seq_len + nnz_kv
num_pages_needed = (total_seq_len + page_size - 1) // page_size  # 向上取整

# --- 创建 Page Pool（全局页池，类似物理内存）---
max_num_pages = 60000   # 页池总容量（所有请求共享）
paged_kv_cache = torch.empty(
    max_num_pages,      # 页池大小
    2,                  # K 和 V（维度 0: K, 维度 1: V）
    page_size,          # 每页的 token 容量
    num_kv_heads,       # KV 头数
    head_dim,           # 每个头的维度
    device=device, dtype=dtype
)

# 随机分配页（模拟内存碎片化，实际推理中页分布不连续）
all_indices = torch.randperm(max_num_pages, device=device, dtype=torch.int32)
kv_page_indices = all_indices[:num_pages_needed]  # 为这个请求分配 257 个不连续的页

# --- 构造页表元数据（类似进程的页表）---
# kv_page_indptr: 标记每个请求使用的页索引范围
# [0, 257] 表示请求 0 使用 kv_page_indices[0:257]
kv_page_indptr = torch.tensor([0, num_pages_needed], dtype=torch.int32, device=device)

# kv_last_page_len: 最后一页当前的填充长度
# 0 表示最后一页（Page 255）已满，新 token 会在新页（Page 256）的位置 0
kv_last_page_len = torch.tensor([0], dtype=torch.int32, device=device)

# --- 构造要追加的 K/V 数据 ---
k_append = torch.randn(nnz_kv, num_kv_heads, head_dim, device=device, dtype=dtype)
v_append = torch.randn(nnz_kv, num_kv_heads, head_dim, device=device, dtype=dtype)

# --- 生成写入位置信息 ---
# kv_append_indptr: 标记每个请求要追加多少个 token（这里只有 1 个请求）
kv_append_indptr = torch.tensor([0, nnz_kv], dtype=torch.int32, device=device)

# seq_lens: 计算当前序列的实际长度（基于页表和 last_page_len）
seq_lens = flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size)

# batch_indices: 每个 token 属于哪个请求（这里都是 0）
# positions: 每个 token 在序列中的位置（这里是 4095，因为 0-indexed）
batch_indices, positions = flashinfer.get_batch_indices_positions(
    kv_append_indptr, seq_lens, nnz_kv
)

# --- Warmup（避免 JIT 编译影响 profiling 结果）---
# 停止 profiling（跳过 warmup）
profiler.stop()
for _ in range(3):
    flashinfer.page.append_paged_kv_cache(
        k_append, v_append, batch_indices, positions,
        paged_kv_cache, kv_page_indices, kv_page_indptr, kv_last_page_len
    )
torch.cuda.synchronize()  # 确保 GPU 操作完成

print("🔥 开始 profiling...")

# 启动 profiling（只 profile 后面 3 次）
profiler.start()

# --- 正式 Profiling: 跑 3 次取平均 ---
for i in range(3):
    # NVTX 标记：在 ncu-ui 中可以看到每次调用
    torch.cuda.nvtx.range_push(f"append_paged_kv_cache_{i}")
    
    # 核心操作：把新 token 的 K/V 写入到对应的页中
    # 内部会根据 positions 计算页号和页内偏移，执行写入
    flashinfer.page.append_paged_kv_cache(
        k_append,           # [1, 4, 128] - 要追加的 K
        v_append,           # [1, 4, 128] - 要追加的 V
        batch_indices,      # [0] - 属于请求 0
        positions,          # [4095] - 写入位置（0-indexed）
        paged_kv_cache,     # [60000, 2, 16, 4, 128] - 全局页池
        kv_page_indices,    # [257] - 这个请求的页号列表
        kv_page_indptr,     # [0, 257] - 页索引范围
        kv_last_page_len    # [0] - 最后一页长度
    )
    
    torch.cuda.nvtx.range_pop()

torch.cuda.synchronize()  # 等待所有 GPU 操作完成

# 停止 profiling
profiler.stop()
print("✅ Profiling 完成!")

# ============================================================
# NCU Profiling 命令使用指南
# ============================================================
#
# 使用 cudaProfilerStart/Stop 控制 profiling 范围（只 profile warmup 后的 3 次调用）:
#
# 1. 基础 profiling（推荐）:
#    ncu --set full --kernel-name "AppendPagedKVCacheKernel" \
#        -o append_profile python append_paged_kvcache.py
#
# 2. 查看内存吞吐量（快速指标）:
#    ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
#        --kernel-name "AppendPagedKVCacheKernel" \
#        python append_paged_kvcache.py
#
# 3. 导出 CSV 格式:
#    ncu --csv --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
#        --kernel-name "AppendPagedKVCacheKernel" \
#        python append_paged_kvcache.py > append_metrics.csv
#
# 4. 查看结果（需要在 Windows 或有 GUI 的环境）:
#    ncu-ui append_profile.ncu-rep
#
# 5. 命令行查看统计（无需 GUI）:
#    ncu --print-summary per-kernel append_profile.ncu-rep
#
# ============================================================


# use this :
# ncu -f  --set full --kernel-name  "AppendPagedKVCacheKernel" -o append_profile python append_paged_kvcache.py