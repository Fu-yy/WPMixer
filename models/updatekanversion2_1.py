import torch
import torch.nn as nn
import torch.linalg as linalg
import matplotlib.pyplot as plt

class CovarianceProcessor:
    """协方差矩阵序列的各种近似与分析方法"""

    @staticmethod
    def extract_variances(cov_seq: torch.Tensor) -> torch.Tensor:
        """
        提取每个时刻每个通道的方差。
        Args:
            cov_seq: Tensor of shape (B, V, T, T)
        Returns:
            Tensor of shape (B, V, T) —— 对角线元素
        """
        # (B, V, T)
        return torch.diagonal(cov_seq, dim1=-2, dim2=-1)

    @staticmethod
    def diagonal_approximation(cov_seq: torch.Tensor) -> torch.Tensor:
        """
        仅保留对角线，其他协方差置零，得到对角近似。
        """
        B, V, T, _ = cov_seq.shape
        variances = CovarianceProcessor.extract_variances(cov_seq)  # (B,V,T)
        approx = torch.zeros_like(cov_seq)
        # 依时刻填充对角
        for t in range(T):
            approx[..., t, t] = variances[..., t]
        return approx

    @staticmethod
    def low_rank_approximation(cov_seq: torch.Tensor, rank: int) -> torch.Tensor:
        """
        对每个协方差矩阵做特征分解，保留前 `rank` 大的特征值及其特征向量。
        """
        B, V, T, _ = cov_seq.shape
        approx = torch.zeros_like(cov_seq)
        for b in range(B):
            for v in range(V):
                # 特征分解
                vals, vecs = linalg.eigh(cov_seq[b, v])  # 升序
                top_vals = vals[-rank:]                  # 后 rank 个最大值
                top_vecs = vecs[:, -rank:]               # 对应特征向量
                approx[b, v] = top_vecs @ torch.diag(top_vals) @ top_vecs.T
        return approx

    @staticmethod
    def plot_variances(variances: torch.Tensor, batch_idx: int = 0, chan_idx: int = 0):
        """
        可视化某一条样本、某一通道的方差随时间的变化。
        """
        # variances: (B, V, T)
        seq = variances[batch_idx, chan_idx].cpu().numpy()
        T = seq.shape[0]
        plt.figure()
        plt.plot(range(T), seq)
        plt.xlabel("Time step")
        plt.ylabel("Variance")
        plt.title(f"Variance over time (batch={batch_idx}, chan={chan_idx})")
        plt.show()

    @staticmethod
    def weighted_mse_loss(pred: torch.Tensor,
                          target: torch.Tensor,
                          variances: torch.Tensor) -> torch.Tensor:
        """
        带不确定度加权的 MSE：
            loss = mean( (pred - target)^2 / variance )
        """
        # 防止除零
        var = variances.clamp_min(1e-6)
        return ((pred - target).pow(2) / var).mean()


# ========== 在时间序列预测中的示例用法 ==========
if __name__ == "__main__":
    # 假设模型输出的协方差序列 cov_seq: (B, V, T, T)
    B, V, T = 4, 2, 100
    # 这里用逐渐增大的对角阵作示例
    cov_seq = torch.stack([
        torch.eye(T, device="cpu") * (0.1 + 0.005 * t)
        for t in range(T)
    ], dim=2).unsqueeze(0).repeat(B, V, 1, 1)  # -> (B, V, T, T)

    # 1) 提取方差
    variances = CovarianceProcessor.extract_variances(cov_seq)  # (B, V, T)

    # 2) 绘制第 0 号批次第 0 通道的方差轨迹
    CovarianceProcessor.plot_variances(variances, batch_idx=0, chan_idx=0)

    # 3) 对角近似
    cov_diag = CovarianceProcessor.diagonal_approximation(cov_seq)

    # 4) 低秩近似（保留前 5 个主成分）
    cov_low_rank = CovarianceProcessor.low_rank_approximation(cov_seq, rank=5)

    # 5) 在损失中使用加权 MSE
    # 假设 model_pred, true_series: (B, V, T)
    model_pred = torch.randn(B, V, T)
    true_series = torch.randn(B, V, T)
    loss = CovarianceProcessor.weighted_mse_loss(model_pred, true_series, variances)
    print("Weighted MSE loss:", loss.item())
