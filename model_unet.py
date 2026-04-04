import torch
import torch.nn as nn
import math


# --- 1. 时间嵌入 (让模型知道现在是第几步) ---
# 扩散模型必须知道“当前噪声有多大”，这个模块负责把数字(t)变成向量
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        inv_freq = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -inv_freq)
        embeddings = time[:, None] * freqs[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# --- 2. Cond Encoder (用于将 LR 图像或文本 embedding 映射到一个向量) ---
class CondEncoder(nn.Module):
    """
    一个轻量级条件编码器：把 LR 图（或预计算的embedding）压缩成一个固定维度的向量。
    这个向量可以进一步被用于 FiLM/AdaIN 层来调制 UNet。
    """
    def __init__(self, in_ch=3, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# --- 3. 基础卷积块 (ResNet Block) ---
# 这是搭建大楼的砖头：卷积 + 归一化 + 激活函数
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, use_film=False, film_dim=128):
        super().__init__()
        self.use_film = use_film
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)  # 翻倍输入是因为有 Skip Connection
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)  # 上采样(变大)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  # 下采样(变小)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        # 如果启用 FiLM，则准备生成 gamma/beta 的小 MLP（从 cond 向量生成）
        if self.use_film:
            # film_dim 是 cond 向量的长度
            self.film_gen = nn.Sequential(
                nn.Linear(film_dim, out_ch * 2),  # 生成 gamma 和 beta
                nn.ReLU()
            )

    def forward(self, x, t, cond_vec=None):
        # 第一次卷积
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 注入时间信息 (关键！让每一层都知道现在是第几步)
        time_emb = self.relu(self.time_mlp(t))
        # 把时间加到特征图上 (广播机制)
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb

        # 如果启用了 FiLM 且提供了 cond_vec，那么生成 gamma/beta 并应用在 h 上
        if self.use_film and cond_vec is not None:
            # cond_vec: [B, film_dim]
            params = self.film_gen(cond_vec)  # [B, out_ch*2]
            gamma, beta = params.chunk(2, dim=1)
            gamma = gamma[(...,) + (None,) * 2]
            beta = beta[(...,) + (None,) * 2]
            # 归一化后再调制
            h = self.bnorm2(h)
            h = gamma * h + beta
            h = self.relu(h)
        else:
            # 第二次卷积
            h = self.bnorm2(self.relu(self.conv2(h)))

        # 变大或变小
        return self.transform(h)


# --- 新增：自注意力模块 SelfAttention2d ---
class SelfAttention2d(nn.Module):
    """
    // 新增：自注意力模块（空间注意力）
    输入: x [B, C, H, W]
    输出: out [B, C, H, W]

    实现说明：
      - 使用 1x1 卷积生成 q,k,v
      - 将空间维 (H,W) 展平为 HW，计算注意力权重
      - 支持多头 attention（heads 参数），默认单头
      - 最后投影回原始通道并加残差

    注意显存：当 H*W 很大时（高分辨率），注意力矩阵 HWxHW 会非常大，
    建议只在 bottleneck（中间低分辨率）或中等分辨率层使用。
    """
    def __init__(self, in_channels, heads=1, head_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        if head_dim is None:
            assert in_channels % heads == 0, "in_channels must be divisible by heads"
            self.head_dim = in_channels // heads
        else:
            self.head_dim = head_dim
        self.inner_dim = self.heads * self.head_dim

        # 使用 1x1 conv 生成 q,k,v
        self.qkv = nn.Conv2d(in_channels, self.inner_dim * 3, kernel_size=1, bias=False)
        # 投影回原始通道
        self.proj = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        # qkv -> [B, 3*inner_dim, H, W]
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # 每个 [B, inner_dim, H, W]

        # reshape 到多头: [B, heads, head_dim, HW]
        q = q.view(B, self.heads, self.head_dim, H * W)
        k = k.view(B, self.heads, self.head_dim, H * W)
        v = v.view(B, self.heads, self.head_dim, H * W)

        # q_t: [B, heads, HW, head_dim]; k_t: [B, heads, head_dim, HW]
        q_t = q.permute(0, 1, 3, 2)
        k_t = k

        # 注意力 logits: [B, heads, HW, HW]
        attn_logits = torch.matmul(q_t, k_t) * self.scale
        attn = torch.softmax(attn_logits, dim=-1)

        # v -> [B, heads, HW, head_dim]
        v_t = v.permute(0, 1, 3, 2)
        out = torch.matmul(attn, v_t)  # [B, heads, HW, head_dim]

        # 恢复形状并投影回原始通道
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.inner_dim, H, W)
        out = self.proj(out)
        # 残差连接
        out = out + x
        return out


# --- 新增：文本条件模块（简化版 MoM，FiLM） ---
class TextConditionModule(nn.Module):
    """
    // 新增：TextConditionModule (简化版多模态混合模块，FiLM)
    功能：将 text_emb ([B, D]) 映射为 gamma 和 beta（每通道一个缩放和平移参数），对特征图 x 做 FiLM:
          y = gamma * x + beta
    输入:
      - text_emb: [B, D]
      - x: [B, C, H, W]
    输出:
      - y: [B, C, H, W]

    说明: 这是一个简化的 MoM，用 MLP 从文本向量直接生成通道级别仿射参数。
    """
    def __init__(self, text_dim, channels, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(128, text_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels * 2)
        )
        # 小幅初始化使 gamma~1, beta~0
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.02)

    def forward(self, text_emb, x):
        B, C, H, W = x.shape
        params = self.mlp(text_emb)  # [B, 2*C]
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        # FiLM 调制：通过文本特征对图像特征做仿射变换
        y = gamma * x + beta
        return y


# --- 3. 核心 UNet ---
class SimpleUNet(nn.Module):
    def __init__(self, cond_mode='concat', cond_dim=128, text_dim=None, use_decoder_attn=True):
        """
        cond_mode: 'concat' (简单通道拼接) 或 'film' (先编码 cond -> 向量，再用 FiLM 调制)
        cond_dim: 当使用 film 时，cond encoder 输出的维度
        text_dim: 如果提供，则启用文本条件注入模块（TextConditionModule），用于简化的多模态混合（MoM）
        """
        super().__init__()
        # 原先的 image_channels 为 6 (noisy + lr)，但我们现在支持多种模式
        image_channels = 6 if cond_mode == 'concat' else 3
        down_channels = (64, 128, 256, 512, 1024)  # 每一层的通道数
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        self.cond_mode = cond_mode
        self.cond_dim = cond_dim
        self.text_dim = text_dim
        self.use_decoder_attn = use_decoder_attn

        # 时间编码器
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # 可选的条件编码器（用于 film 模式）
        if cond_mode == 'film':
            self.cond_encoder = CondEncoder(in_ch=3, out_dim=cond_dim)
        else:
            self.cond_encoder = None

        # 初始卷积
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # --- 左边：下采样路径 (Down) ---
        # 若使用 film，则在每个 Block 中打开 FiLM 支持
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i + 1], time_emb_dim, use_film=(cond_mode == 'film'), film_dim=cond_dim) \
            for i in range(len(down_channels) - 1)
        ])

        # --- 右边：上采样路径 (Up) ---
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True, use_film=(cond_mode == 'film'), film_dim=cond_dim) \
            for i in range(len(up_channels) - 1)
        ])

        # 输出层
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

        # // 新增：分割 head（segmentation head）
        # 分割 head 使用 decoder 的最后特征作为输入，输出单通道概率图 (sigmoid)
        self.seg_head = nn.Sequential(
            nn.Conv2d(up_channels[-1], up_channels[-1] // 2, 3, padding=1),
            nn.BatchNorm2d(up_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(up_channels[-1] // 2, 1, 1),
            nn.Sigmoid()
        )

        # // 新增：自注意力和文本条件模块的初始化
        # 在 bottleneck（downs 完成后，x 的通道数为 down_channels[-1]）加入自注意力
        self.attn_bottleneck = SelfAttention2d(down_channels[-1], heads=8)  # // 新增：自注意力模块 (bottleneck)

        # 在 decoder 的前两层加入自注意力以增强中程信息（可根据显存调整/关闭）
        attn_up_channels = (up_channels[1], up_channels[2])  # 512, 256
        if self.use_decoder_attn:
            self.attn_ups = nn.ModuleList([SelfAttention2d(ch, heads=4) for ch in attn_up_channels])  # // 新增：自注意力模块 (decoder)
        else:
            self.attn_ups = nn.ModuleList()

        # 文本条件模块（简化版 MoM, FiLM）
        if text_dim is not None:
            # 在 bottleneck 以及对应 decoder 层加入文本调制
            self.text_cond_bottleneck = TextConditionModule(text_dim, down_channels[-1])  # // 新增：TextConditionModule (bottleneck)
            self.text_cond_ups = nn.ModuleList([TextConditionModule(text_dim, ch) for ch in attn_up_channels])  # // 新增：TextConditionModule (decoder)
        else:
            self.text_cond_bottleneck = None
            self.text_cond_ups = None

    def forward(self, x, timestep, cond=None, text_emb=None):
        """
        x: noisy image tensor, shape [B, 3, H, W] (如果 cond_mode=='concat' 则为 [B,6,H,W])
        timestep: [B] 或 [B,] 的整型张量
        cond: 当 cond_mode=='concat' 时，传入已经上采样并 concat 到 x 的 LR tensor
              当 cond_mode=='film' 时，传入 LR 图 [B,3,h,w]（较小），会被编码成向量
        text_emb: 可选的文本 embedding，shape [B, D]，用于 FiLM 条件注入（简化版 MoM）
        返回: (recon_or_noise, mask_pred)  # mask_pred shape [B,1,H,W]
        """
        # 1. 嵌入时间
        t = self.time_mlp(timestep)

        # 2. 如果 film 模式，需要把 cond 编码成向量
        cond_vec = None
        if self.cond_mode == 'film' and cond is not None:
            # cond 期待是 [B,3,h,w]，直接送入编码器
            cond_vec = self.cond_encoder(cond)

        # 3. 初始处理
        x = self.conv0(x)

        # 4. 记录跳跃连接 (Skip Connections)
        residuals = []
        for down in self.downs:
            # Block now accepts optional cond_vec for FiLM
            x = down(x, t, cond_vec)
            residuals.append(x)  # 存起来！

        # --- 新增：在 bottleneck 处应用文本调制（FiLM）和自注意力 ---
        if text_emb is not None and self.text_cond_bottleneck is not None:
            # // 新增：在 bottleneck 应用文本 FiLM 条件（简化版 MoM）
            x = self.text_cond_bottleneck(text_emb, x)

        # // 新增：在 bottleneck 应用自注意力（增强长程依赖）
        x = self.attn_bottleneck(x)

        # 5. 上采样并融合
        # 我们希望在部分 decoder 层也能应用文本调制和注意力，以增强多模态信息传播
        for i, up in enumerate(self.ups):
            residual = residuals.pop()  # 取出刚才存的特征
            # 把当前的 x 和之前的 residual 拼在一起 (Concat)
            x = torch.cat((x, residual), dim=1)
            x = up(x, t, cond_vec)

            # 在前两层 decoder 上应用文本调制和自注意力（如果启用）
            if self.use_decoder_attn and i < len(self.attn_ups):
                if text_emb is not None and self.text_cond_ups is not None:
                    # // 新增：在 decoder 层应用文本 FiLM 条件（简化版 MoM）
                    x = self.text_cond_ups[i](text_emb, x)
                # // 新增：在 decoder 层应用自注意力（注意显存）
                x = self.attn_ups[i](x)

        # // 新增：分割 head 的前向（使用 decoder 的最后特征 x）
        mask_pred = self.seg_head(x)  # // 新增：分割 head 输出，shape [B,1,H,W]

        # 输出重建 / 噪声预测
        recon = self.output(x)

        return recon, mask_pred


# --- 测试代码 ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在测试 UNet 模型，设备: {device}")

    # 为了兼容不同 cond_mode，我们根据模型的 cond_mode 生成合适的输入通道
    model = SimpleUNet(cond_mode='concat')
    in_ch = 6 if model.cond_mode == 'concat' else 3

    # 模拟一个 Batch 的数据
    x = torch.randn(4, in_ch, 64, 64).to(device)

    # 模拟时间步：告诉模型这4张图分别处于第几步 (比如第5步、第100步...)
    t = torch.randint(0, 1000, (4,)).to(device)

    model = model.to(device)

    # 跑一次前向传播
    try:
        out, mask = model(x, t)
        print("\n✅ 模型架构测试通过！")
        print(f"输入形状: {x.shape}")
        print(f"输出 recon 形状: {out.shape}")
        print(f"输出 mask 形状: {mask.shape}")

        if out.shape[0] == x.shape[0] and out.shape[2] == x.shape[2] and out.shape[3] == x.shape[3]:
            print("🎉 输入输出尺寸一致（batch & spatial），这是一个合格的 '画图' 模型。")
        else:
            print("❌ 警告：输入输出尺寸不一致，需要检查 Padding 或 Stride。")

    except Exception as e:
        print(f"\n❌ 模型崩溃了: {e}")