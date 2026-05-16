
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.nn import HypergraphConv as HGNNConv
from torch_geometric.utils import scatter
# 全局随机种子（保证可复现）
SEED = 48
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# ==================== 修改后：三极性感知原型特征转换层（移除扰动相关逻辑） ====================
class PerturbationPrototypicalLayer(nn.Module):



    def __init__(self, in_dim, prototype_dim=None, learnable=True, alpha=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.prototype_dim = prototype_dim if prototype_dim is not None else in_dim
        self.learnable = learnable
        self.alpha = alpha  # 扰动强度系数，控制向原型靠近的程度

        # 1. 三类极性原型中心
        # 初始化策略：使用正交初始化或拉开距离，避免初期坍塌
        self.sensitivity_hub = nn.Parameter(
            torch.randn(self.prototype_dim), requires_grad=self.learnable
        )
        self.resistance_hub = nn.Parameter(
            torch.randn(self.prototype_dim), requires_grad=self.learnable
        )
        self.none_hub = nn.Parameter(
            torch.randn(self.prototype_dim), requires_grad=self.learnable
        )

        # 2. 特征映射
        self.feature_adapter = nn.Linear(self.in_dim, self.prototype_dim)

        # 4. 归一化层 (防止特征尺度爆炸)
        self.ln = nn.LayerNorm(self.in_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        # 原型中心：拉开初始距离，避免对称性坍塌
        # 方法：生成随机向量后归一化，并乘以不同标量或方向
        hubs = torch.randn(3, self.prototype_dim)
        # 简单正交化近似：让三个向量初始方向不同
        nn.init.orthogonal_(hubs)
        self.sensitivity_hub.data.copy_(hubs[0])
        self.resistance_hub.data.copy_(hubs[1])
        self.none_hub.data.copy_(hubs[2])

        # 线性层初始化
        nn.init.xavier_uniform_(self.feature_adapter.weight)
        nn.init.zeros_(self.feature_adapter.bias)


    def forward(self, x):
        # 1. 映射到原型空间 [N, in_dim] → [N, prototype_dim]
        x_proto = self.feature_adapter(x)

        # 2. 堆叠原型中心 [3, P]
        hubs = torch.stack([self.sensitivity_hub, self.resistance_hub, self.none_hub], dim=0)

        # 3. 计算样本到每个原型的相似度 (这里使用余弦相似度或点积)
        # x_proto: [N, P], hubs: [3, P] -> scores: [N, 3]
        # 使用点积后 Softmax 作为注意力权重
        scores = F.linear(x_proto, hubs)  # [N, 3]
        weights = F.softmax(scores, dim=-1)  # [N, 3], 表示样本属于三类极性的概率分布

        # 4. 计算“目标原型位置” (加权原型中心)
        # weighted_hubs: [N, P] = sum(weights[i] * hubs[i])
        weighted_hubs = torch.matmul(weights, hubs)

        # 5. 计算位移向量 (目标 - 当前)
        # 这步保留了 x 的依赖性，不同样本有不同的位移方向
        displacement = weighted_hubs - x_proto  # [N, P]



        # 这相当于将样本特征向它最可能的原型中心拉近
        x_proto_perturbed = x_proto + self.alpha * displacement


        return x_proto_perturbed


class SignedHypergraphConv(nn.Module):
    """符号超图卷积层：共享权重矩阵，区分正负扰动（敏感+1/耐药-1）
    核心改进：实现超图特有的 节点→超边→节点 双向聚合
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 节点特征变换矩阵（共享）
        self.node_lin = nn.Linear(in_channels, out_channels, bias=bias)
        # 超边特征变换矩阵
        self.edge_lin = nn.Linear(out_channels, out_channels, bias=bias)
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_lin.weight)
        nn.init.xavier_uniform_(self.edge_lin.weight)
        if self.node_lin.bias is not None:
            nn.init.zeros_(self.node_lin.bias)
        if self.edge_lin.bias is not None:
            nn.init.zeros_(self.edge_lin.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, hyperedge_index, sign_weights):
        N, _ = x.shape  # 节点数
        device = x.device

        # 1. 第一步：节点特征投影
        x_proj = self.node_lin(x)

        # 解析超边索引：node_indices(超边内的节点), edge_indices(对应的超边ID)
        node_indices, edge_indices = hyperedge_index
        E = sign_weights.size(0)  # 超边数量

        # 2. 第二步：节点→超边聚合（将超边内所有节点特征聚合成超边特征）
        # 符号权重映射到每个节点-超边对
        sign_mapped = sign_weights[edge_indices].unsqueeze(1).to(device)
        # 节点特征加权（乘以符号权重）
        weighted_node_feats = sign_mapped * x_proj[node_indices]
        # 聚合超边内的节点特征得到超边特征 (E, out_channels)
        edge_feats = scatter_add(
            src=weighted_node_feats,
            index=edge_indices.unsqueeze(1).expand_as(weighted_node_feats),
            dim=0,
            dim_size=E
        )
        # 超边特征变换
        edge_feats = self.edge_lin(edge_feats)

        # 3. 第三步：超边→节点聚合（将节点关联的所有超边特征聚合回节点）
        # 初始化节点聚合特征
        x_agg = torch.zeros(N, self.out_channels, device=device)
        # 聚合节点关联的超边特征
        x_agg = scatter_add(
            src=edge_feats[edge_indices],
            index=node_indices.unsqueeze(1).expand_as(edge_feats[edge_indices]),
            dim=0,
            dim_size=N
        )

        # 偏置与激活
        if self.bias is not None:
            x_agg += self.bias

        return F.relu(x_agg)


# ==================== 符号扰动超图编码器（移除扰动对比损失） ====================
class PerturbationHgnnEncoder(nn.Module):
    """符号扰动超图编码器：多层卷积（移除扰动对比损失）"""

    def __init__(self, in_channels, dim_1, dropout=0.15):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 多层符号超图卷积
        self.conv1 = SignedHypergraphConv(in_channels, dim_1 // 8)
        # self.conv2 = SignedHypergraphConv(dim_1 // 16, dim_1 // 8)
        # self.conv3 = SignedHypergraphConv(dim_1 // 8, dim_1 // 4)
        # self.conv4 = SignedHypergraphConv(dim_1 // 4, dim_1//2 )
        # self.conv5 = SignedHypergraphConv(dim_1//2, dim_1 )
        # 输出投影层（保持维度一致，适配后续拼接）
        self.out_proj = nn.Linear(dim_1//8, in_channels)
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, SignedHypergraphConv):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

    def forward(self, x, hyperedge_index, sign_weights):
        x = self.dropout(x)
        # 多层符号超图卷积提取特征
        x1 = self.conv1(x, hyperedge_index, sign_weights)
        # x2 = self.conv2(x1, hyperedge_index, sign_weights)
        # x3 = self.conv3(x2, hyperedge_index, sign_weights)
        # x4 = self.conv4(x3, hyperedge_index, sign_weights)
        # x5 = self.conv5(x4, hyperedge_index, sign_weights)
        # 投影回输入维度
        out = F.relu(self.out_proj(x1))
        return out

class CrossModalAttention(nn.Module):
    def __init__(self, modal_dim, out_dim=None, num_heads=8, dropout=0.15):
        super().__init__()
        self.modal_dim = modal_dim
        self.out_dim = out_dim if out_dim is not None else modal_dim
        self.num_heads = num_heads
        assert self.modal_dim % num_heads == 0, f"modal_dim({self.modal_dim})必须是num_heads({num_heads})的整数倍"

        # 双向交叉注意力
        self.ncrna_drug_attn = nn.MultiheadAttention(self.modal_dim, num_heads, dropout=dropout, batch_first=True)
        self.ncrna_disease_attn = nn.MultiheadAttention(self.modal_dim, num_heads, dropout=dropout, batch_first=True)
        self.drug_disease_attn = nn.MultiheadAttention(self.modal_dim, num_heads, dropout=dropout, batch_first=True)

        # ==================== 门控机制（替换原来的简单平均） ====================
        # 双向注意力门控：学习 A→B 和 B→A 各自的权重
        self.gate_nc_dg = nn.Sequential(
            nn.Linear(modal_dim * 2, modal_dim),
            nn.Sigmoid()
        )
        self.gate_nc_dis = nn.Sequential(
            nn.Linear(modal_dim * 2, modal_dim),
            nn.Sigmoid()
        )
        self.gate_dg_dis = nn.Sequential(
            nn.Linear(modal_dim * 2, modal_dim),
            nn.Sigmoid()
        )

        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(self.modal_dim * 6, self.modal_dim * 2),
            nn.BatchNorm1d(self.modal_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.modal_dim * 2, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU()
        )
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ncrna_feat, drug_feat, disease_feat, ncrna_ids, drug_ids, disease_ids):
        batch_nc = ncrna_feat[ncrna_ids].unsqueeze(1)   # [B, 1, D]
        batch_dg = drug_feat[drug_ids].unsqueeze(1)
        batch_dis = disease_feat[disease_ids].unsqueeze(1)

        # ==================== nc ↔ dg 双向注意力 ====================
        nc2dg, _ = self.ncrna_drug_attn(query=batch_nc, key=batch_dg, value=batch_dg)
        dg2nc, _ = self.ncrna_drug_attn(query=batch_dg, key=batch_nc, value=batch_nc)
        gate_dg = self.gate_nc_dg(torch.cat([nc2dg, dg2nc], dim=-1))
        nc_dg_gated = gate_dg * nc2dg + (1 - gate_dg) * dg2nc

        # ==================== nc ↔ dis 双向注意力 ====================
        nc2dis, _ = self.ncrna_disease_attn(query=batch_nc, key=batch_dis, value=batch_dis)
        dis2nc, _ = self.ncrna_disease_attn(query=batch_dis, key=batch_nc, value=batch_nc)
        gate_dis = self.gate_nc_dis(torch.cat([nc2dis, dis2nc], dim=-1))
        nc_dis_gated = gate_dis * nc2dis + (1 - gate_dis) * dis2nc

        # ==================== dg ↔ dis 双向注意力 ====================
        dg2dis, _ = self.drug_disease_attn(query=batch_dg, key=batch_dis, value=batch_dis)
        dis2dg, _ = self.drug_disease_attn(query=batch_dis, key=batch_dg, value=batch_dg)
        gate_dgdis = self.gate_dg_dis(torch.cat([dg2dis, dis2dg], dim=-1))
        dg_dis_gated = gate_dgdis * dg2dis + (1 - gate_dgdis) * dis2dg

        # 拼接：自身特征 + 门控融合后的交叉特征
        concat_feat = torch.cat([
            batch_nc.squeeze(1),
            batch_dg.squeeze(1),
            batch_dis.squeeze(1),
            nc_dg_gated.squeeze(1),
            nc_dis_gated.squeeze(1),
            dg_dis_gated.squeeze(1)
        ], dim=1)

        fused_feat = self.fusion(concat_feat)
        return fused_feat
# -------------------------- 分层分类器（无修改） --------------------------
class HierarchicalClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=3, dropout=0.15):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        l1_out = self.layer1(x)
        l2_out = self.layer2(l1_out)
        logits = self.classifier(l2_out)
        pred_probs = F.softmax(logits, dim=1)

        pred_list = []
        for idx, prob in enumerate(pred_probs):
            pred_list.append((prob.detach().cpu().numpy(), (idx, idx, idx)))

        return logits, pred_probs, pred_list


# -------------------------- 修改后：双分支主模型（移除扰动损失） --------------------------
class NcrnaDrugDiseaseDualBranchModel(nn.Module):
    def __init__(self, ncrna_dim, drug_dim, disease_dim, modal_hidden_dim, modal_out_dim, classifier_hidden_dim,
                 hgnn_dim_1, dropout=0.15):
        super().__init__()
        # 单模态原型对齐层
        self.ncrna_aligner = PerturbationPrototypicalLayer(ncrna_dim, modal_out_dim)
        self.drug_aligner = PerturbationPrototypicalLayer(drug_dim, modal_out_dim)
        self.disease_aligner = PerturbationPrototypicalLayer(disease_dim, modal_out_dim)
        self.cross_attention = CrossModalAttention(
            modal_dim=modal_out_dim,
            out_dim=modal_out_dim,
            dropout=dropout
        )

        # # 超图编码器
        self.hgnn_encoder = PerturbationHgnnEncoder(modal_out_dim, hgnn_dim_1, dropout)
        # 超图分支投影层
        self.hgnn_triple_proj = nn.Sequential(
            nn.Linear(modal_out_dim, modal_out_dim),
            nn.BatchNorm1d(modal_out_dim),
            nn.ReLU()
        )
        # 双分支拼接维度：D + D = 2D
        self.fusion_dim = modal_out_dim * 2
        #消融
        # self.fusion_dim = modal_out_dim
        self.classifier = HierarchicalClassifier(
            in_dim=self.fusion_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout
        )
    #
        self.reset_branch_para()

    def reset_branch_para(self):
        for m in [self.hgnn_triple_proj]:
            for sub_m in m:
                if isinstance(sub_m, nn.Linear):
                    nn.init.xavier_uniform_(sub_m.weight)
                    if sub_m.bias is not None:
                        nn.init.zeros_(sub_m.bias)

    def forward(self, ncrna_feat, drug_feat, disease_feat, ncrna_ids, drug_ids, disease_ids,
                hyperedge_index=None, sign_weights=None):
        # 步骤1：单模态原型对齐
        ncrna_aligned = self.ncrna_aligner(ncrna_feat)   # [N_n, D]
        drug_aligned = self.drug_aligner(drug_feat)      # [N_d, D]
        disease_aligned = self.disease_aligner(disease_feat)  # [N_dis, D]

        # 步骤2：分支1 - 交叉注意力
        branch1_feat = self.cross_attention(
            ncrna_feat=ncrna_aligned,
            drug_feat=drug_aligned,
            disease_feat=disease_aligned,
            ncrna_ids=ncrna_ids,
            drug_ids=drug_ids,
            disease_ids=disease_ids
        )  # [B, D]

        # 步骤3：分支2 - 超图卷积
        all_entity_feat = torch.cat([ncrna_aligned, drug_aligned, disease_aligned], dim=0)  # [N_all, D]

        if hyperedge_index is None or sign_weights is None:
            all_entity_hgnn = all_entity_feat
        else:
            all_entity_hgnn = self.hgnn_encoder(
                x=all_entity_feat,
                hyperedge_index=hyperedge_index,
                sign_weights=sign_weights
            )  # [N_all, D]

        ncrna_num = ncrna_feat.shape[0]
        drug_num = drug_feat.shape[0]

        batch_ncrna_hgnn = all_entity_hgnn[ncrna_ids]
        batch_drug_hgnn = all_entity_hgnn[ncrna_num + drug_ids]
        batch_disease_hgnn = all_entity_hgnn[ncrna_num + drug_num + disease_ids]

        hgnn_triple_feat = (batch_ncrna_hgnn + batch_drug_hgnn + batch_disease_hgnn) / 3
        branch2_feat = self.hgnn_triple_proj(hgnn_triple_feat)  # [B, D]

        # 步骤4：融合 + 分类
        final_fused_feat = torch.cat([branch1_feat, branch2_feat], dim=1)  # [B, 2D]

        # final_fused_feat = branch2_feat
        logits, pred_probs, pred_list = self.classifier(final_fused_feat)
        return logits, pred_probs, pred_list

# -------------------------- 参数配置（无修改） --------------------------
def parameters_set():
    class Args:
        def __init__(self):
            self.seed = SEED
            self.k_fold = 5
            self.epochs = 1000
            self.lr = 5e-4
            self.modal_hidden_dim = 256
            self.modal_out_dim = 64  # 必须是4的整数倍
            self.classifier_hidden_dim = 64
            self.hgnn_dim_1 = 256
            self.dropout = 0.15
    return Args()


# 全局变量
modal_out_dim = parameters_set().modal_out_dim