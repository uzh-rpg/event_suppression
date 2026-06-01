import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def measure_scaled_attention(
    seq_len=1024,
    batch_size=32,
    n_heads=8,
    d_head=64,
    masked=False,
    warmup_iters=5
):
    """
    Measure the runtime of a single scaled_dot_product_attention call
    with and without a custom attention mask.
    """
    # Create random q, k, v: shape (B, S, H, D)
    q = torch.randn(batch_size, seq_len, n_heads, d_head, device='cuda', dtype=torch.half)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Permute to (B, H, S, D)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    # Create a boolean mask of shape (B, 1, S, S) if requested
    if masked:
        # Example: causal + random masking
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device='cuda', dtype=torch.bool))
        rand_mask = torch.rand(batch_size, 1, seq_len, seq_len, device='cuda') > 0.1
        mask = causal_mask.view(1, 1, seq_len, seq_len) & rand_mask
        # Expand to (B, H, S, S)
        attn_mask = mask.expand(batch_size, n_heads, seq_len, seq_len)
    else:
        attn_mask = None

    # Warm-up to avoid startup overhead
    for _ in range(warmup_iters):
        _ = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=0.0)
    torch.cuda.synchronize()

    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    _ = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=0.0)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)

if __name__ == "__main__":
    time_ratio = []
    for sequence_len in range(8, 2049, 8):
        t_no_mask = measure_scaled_attention(masked=False, seq_len=sequence_len)
        t_masked = measure_scaled_attention(masked=True, seq_len=sequence_len)
        time_ratio.append(t_masked / t_no_mask)
        print(f"Seq len {sequence_len}: No mask: {t_no_mask:.2f} ms, Masked: {t_masked:.2f} ms")

    plt.plot(time_ratio)
    plt.savefig("time_ratio.png")
    t_no_mask = measure_scaled_attention(masked=False, seq_len=2048)
    t_masked = measure_scaled_attention(masked=True, seq_len=2048)
    print(f"No mask:   {t_no_mask:.2f} ms")
    print(f"Masked:    {t_masked:.2f} ms")
    print(f"Masked is {t_masked / t_no_mask:.2f}× slower")

