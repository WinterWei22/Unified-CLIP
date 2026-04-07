
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.typing import SparseTensor

from typing import Tuple
import math

F_dim = 9


Sigma_r = 1
Sigma_theta = 2
Sigma_phi = 2
Sigma_beta = 0.07
Eta = 10.8
Alpha = 1.5
a_0, a_1, a_2, a_3 = 3, 100, -200, -164
r_c = 6
# 积分精度
N_r = 10
N_theta = 10
N_phi = 10

r_cpu = torch.Tensor(range(N_r + 1)) / N_r * r_c
theta_cpu = torch.Tensor(range(N_theta + 1)) / N_theta * torch.tensor(math.pi)
phi_cpu = torch.Tensor(range(N_phi + 1)) / N_phi * torch.tensor(math.pi)


def weighted_bincount(inputs, weights: torch.Tensor, minlength):
    shape = (minlength,)
    dtype = weights.dtype
    device = weights.device
    output = torch.zeros(shape, dtype=dtype, device=device)
    index = inputs.expand_as(weights).long()
    return output.scatter_add_(0, index, weights)



def F_two_body(Z, R, i, j):
    r = r_cpu.to(Z.device)
    Z = Z / Z.max(-1).values.unsqueeze(-1) * r_c
    mu = (R.unsqueeze(-1) + Z[:, :Z.shape[-1]//2][j]).unsqueeze(-1)
    tau = Z[:, Z.shape[-1]//2:].unsqueeze(-1)
    y: torch.Tensor = torch.sum(torch.exp(-(r - mu) ** 2 / Sigma_r) * tau[j], dim=1) / (Sigma_r * 2 * math.sqrt(torch.tensor(math.pi)))

    F_0 = torch.exp(-Eta * r) - math.sqrt(2) * torch.tensor(math.pi) / (10 * (r + 1) ** 3)
    F_1 = math.sqrt(2) * torch.tensor(math.pi) / 10 * 2 * (r - R.reshape(-1, 1)) / (r + 1) ** 6
    F_2 = torch.exp(-Alpha * r) * 4 * ((r - R.reshape(-1, 1)) ** 2 - 2)
    return torch.stack([
        weighted_bincount(i, weights=torch.trapz(r.unsqueeze(0), y, dim=-1), minlength=i.max()+1),
        weighted_bincount(i, weights=torch.trapz(r.unsqueeze(0), y * F_0, dim=-1), minlength=i.max()+1),
        weighted_bincount(i, weights=torch.trapz(r.unsqueeze(0), y * F_1, dim=-1), minlength=i.max()+1),
        weighted_bincount(i, weights=torch.trapz(r.unsqueeze(0), y * F_2, dim=-1), minlength=i.max()+1),
    ], dim=-1)


def F_three_body(Z, theta_ijk, theta_jki, theta_kij, i, j, k, idx_kj, idx_ji, R, pos):
    theta = theta_cpu.to(Z.device)
    Z = Z / Z.max(-1).values.unsqueeze(-1) * torch.tensor(math.pi)
    mu = (theta_ijk.unsqueeze(-1) + Z[:, :Z.shape[-1]//2][j]).unsqueeze(-1)
    tao = Z[:, Z.shape[-1]//2:].unsqueeze(-1)
    y: torch.Tensor = torch.sum(torch.exp(-(theta - mu) ** 2 / Sigma_theta) * tao[j] * tao[k], dim=1) / (Sigma_r * 2 * math.sqrt(torch.tensor(math.pi)))

    R_ik = torch.norm(pos[i] -pos[k], dim=-1, p=2)
    atm = ((R[idx_ji]*R[idx_kj]*R_ik) ** 4).reshape(-1, 1)
    F_0 = (a_0 + a_1 * torch.cos(theta) + a_2 * torch.cos(2*theta) + a_3 * torch.cos(3*theta)) / atm
    F_1 = (1 + torch.cos(theta) * torch.cos(theta_jki).unsqueeze(-1) * torch.cos(theta_kij).unsqueeze(-1)) / atm
    return torch.stack([
        weighted_bincount(i, weights=torch.trapz(theta.unsqueeze(0), y, dim=-1), minlength=i.max()+1),
        weighted_bincount(i, weights=torch.trapz(theta.unsqueeze(0), y * F_0, dim=-1), minlength=i.max()+1),
        weighted_bincount(i, weights=torch.trapz(theta.unsqueeze(0), y * F_1, dim=-1), minlength=i.max()+1),
    ], dim=-1)

def F_four_body(Z, phi_ijkl, psi_ijkl, i, j, k, l):
    phi = phi_cpu.to(Z.device)
    Z = Z / Z.max(-1).values.unsqueeze(-1) * torch.tensor(math.pi)
    mu = (phi_ijkl.unsqueeze(-1) + Z[:, :Z.shape[-1]//4][j]).unsqueeze(-1)
    tao = Z[:, Z.shape[-1]//4:Z.shape[-1]//4*2].unsqueeze(-1)
    y: torch.Tensor = torch.sum(torch.exp(-(phi - mu) ** 2 / Sigma_theta) * tao[j] * tao[k] * tao[l], dim=1) / (Sigma_r * 2 * math.sqrt(torch.tensor(math.pi)))
    mu_psi = (psi_ijkl.unsqueeze(-1) + Z[:, Z.shape[-1]//4*2:Z.shape[-1]//4*3][j]).unsqueeze(-1)
    tao_psi = Z[:, Z.shape[-1]//4*3:Z.shape[-1]//4*4].unsqueeze(-1)
    y_psi: torch.Tensor = torch.sum(torch.exp(-(phi - mu_psi) ** 2 / Sigma_theta) * tao_psi[j] * tao_psi[k] * tao_psi[l], dim=1) / (Sigma_r * 2 * math.sqrt(torch.tensor(math.pi)))
    return torch.stack([
        weighted_bincount(i, weights=torch.trapz(phi.unsqueeze(0), y, dim=-1), minlength=i.max() + 1),
        weighted_bincount(i, weights=torch.trapz(phi.unsqueeze(0), y_psi, dim=-1), minlength=i.max() + 1),
    ], dim=-1)


def _Rho_two(r, Z, R, i, j, func_e=1):
    y = torch.sum(torch.unsqueeze(torch.exp(-(r - R.reshape(-1, 1)) ** 2 / Sigma_r), -1) * torch.unsqueeze(Z[j], 1), dim=-1) / (Sigma_r * 2 * math.sqrt(torch.tensor(math.pi)))
    return weighted_bincount(i, weights=torch.trapz(r.unsqueeze(0), y * func_e, dim=-1), minlength=i.max()+1)


def Rho_two(Z, R, i, j):
    r = r_cpu.to(Z.device)
    return _Rho_two(r, R, i, j)


def _Rho_three(theta, Z, theta_ijk, i, j, k, func_e=1):
    y = torch.sum(torch.unsqueeze(torch.exp(-(theta - theta_ijk.reshape(-1, 1)) ** 2 / Sigma_theta), -1) * torch.unsqueeze(Z[j], 1) * torch.unsqueeze(Z[k], 1), dim=-1) / (Sigma_theta * 2 * math.sqrt(torch.tensor(math.pi)))
    return weighted_bincount(i, weights=torch.trapz(theta.unsqueeze(0), y * func_e, dim=-1), minlength=i.max()+1)


def Rho_three(Z, theta_ijk, i, j, k):
    theta = theta_cpu.to(Z.device)
    return _Rho_three(theta, Z, theta_ijk, i, j, k)


def _Rho_four(phi, Z, phi_ijkl, i, j, k, l, func_e=1):
    y = torch.sum(torch.unsqueeze(torch.exp(-(phi - phi_ijkl.reshape(-1, 1)) ** 2 / Sigma_phi), -1) * torch.unsqueeze(Z[j], 1) * torch.unsqueeze(Z[k], 1) * torch.unsqueeze(Z[l], 1), dim=-1) / (Sigma_phi * 2 * math.sqrt(torch.tensor(math.pi)))
    return weighted_bincount(i, weights=torch.trapz(phi.unsqueeze(0), y * func_e, dim=-1), minlength=i.max()+1)


def Rho_four(Z, phi_ijkl, i, j, k, l):
    phi = phi_cpu.to(Z.device)
    return _Rho_four(phi, Z, phi_ijkl, i, j, k, l)


# 四体计算
def triplets(
        edge_index: torch.Tensor,
        num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return idx_i, idx_j, idx_k, idx_kj, idx_ji


def quadruplets(
        edge_index: torch.Tensor,
        num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Add code here for quadruplets.
    adj_q_row = adj_t[idx_k]
    num_quadruplets = adj_q_row.set_value(None).sum(dim=1).to(torch.long)
    idx_i = idx_i.repeat_interleave(num_quadruplets)
    idx_j = idx_j.repeat_interleave(num_quadruplets)
    idx_k = idx_k.repeat_interleave(num_quadruplets)
    idx_kj = idx_kj.repeat_interleave(num_quadruplets)
    idx_ji = idx_ji.repeat_interleave(num_quadruplets)

    idx_l = adj_q_row.storage.col()
    mask = (idx_k != idx_l) & (idx_j != idx_l) & (idx_i != idx_l)  # Remove i == l, j == l, and k == l quadruplets.
    idx_i, idx_j, idx_k, idx_l, idx_kj, idx_ji = idx_i[mask], idx_j[mask], idx_k[mask], idx_l[mask], idx_kj[mask], \
        idx_ji[mask]

    # Edge indices (l-k, k-j, j->i) for quadruplets.
    idx_lk = adj_q_row.storage.value()[mask]
    idx_kj = idx_kj  # already computed
    idx_ji = idx_ji  # already computed

    return idx_i, idx_j, idx_k, idx_l, idx_lk, idx_kj, idx_ji


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)
        return x


class CrossMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.scale = self.hidden_dim ** -0.5

        self.q = nn.Linear(in_dim, self.num_heads * hidden_dim)
        self.kv = nn.Linear(in_dim, 2 * self.num_heads * hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_v, mask=None, only_attn=False):
        q = self.q(x_q)
        k, v = self.kv(x_v).chunk(2, dim=-1)

        q = q.view(-1, q.size(-2), self.num_heads, self.hidden_dim).transpose(1, 2)
        k = k.view(-1, k.size(-2), self.num_heads, self.hidden_dim).permute(0, 2, 3, 1)
        v = v.view(-1, v.size(-2), self.num_heads, self.hidden_dim).transpose(1, 2)

        attn = (q @ k) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        if only_attn:
            return attn
        out = (attn @ v).transpose(1, 2).mean(-2).reshape(-1, q.size(-2), self.hidden_dim)
        out = self.out(out)
        out = self.residual_dropout(out)
        return out