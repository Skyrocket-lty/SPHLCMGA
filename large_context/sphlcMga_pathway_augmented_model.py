from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model import (
    CrossModalAttention,
    HierarchicalClassifier,
    PerturbationHgnnEncoder,
    PerturbationPrototypicalLayer,
)
def aggregate_mean(
    src_emb: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    dst_size: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if src_index.numel() == 0:
        return src_emb.new_zeros((dst_size, src_emb.size(1)))
    gathered = src_emb[src_index]
    if weights is None:
        weights = torch.ones(src_index.size(0), device=src_emb.device, dtype=src_emb.dtype)
    else:
        weights = weights.to(device=src_emb.device, dtype=src_emb.dtype)
    out = src_emb.new_zeros((dst_size, src_emb.size(1)))
    normalizer = src_emb.new_zeros((dst_size, 1))
    out.index_add_(0, dst_index, gathered * weights.unsqueeze(1))
    normalizer.index_add_(0, dst_index, weights.unsqueeze(1))
    normalizer = torch.clamp(normalizer, min=1e-8)
    return out / normalizer


@dataclass
class PathwayCounts:
    ncrna: int
    drug: int
    disease: int
    pathway: int


class PathwayAugmentedSPHLCMGA(nn.Module):
    def __init__(
        self,
        counts: PathwayCounts,
        relations: dict[str, dict[str, torch.Tensor]],
        ncrna_dim: int,
        drug_dim: int,
        disease_dim: int,
        modal_out_dim: int,
        classifier_hidden_dim: int,
        hgnn_dim_1: int,
        disease_aux_features: torch.Tensor | None = None,
        use_pathway_branch: bool = True,
        use_context_branch: bool = False,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.counts = counts
        self.relations = relations
        self.modal_out_dim = modal_out_dim
        self.use_pathway_branch = use_pathway_branch
        self.use_context_branch = use_context_branch

        self.ncrna_aligner = PerturbationPrototypicalLayer(ncrna_dim, modal_out_dim)
        self.drug_aligner = PerturbationPrototypicalLayer(drug_dim, modal_out_dim)
        self.disease_aligner = PerturbationPrototypicalLayer(disease_dim, modal_out_dim)
        self.cross_attention = CrossModalAttention(
            modal_dim=modal_out_dim,
            out_dim=modal_out_dim,
            dropout=dropout,
        )
        self.hgnn_encoder = PerturbationHgnnEncoder(modal_out_dim, hgnn_dim_1, dropout)
        self.hgnn_triple_proj = nn.Sequential(
            nn.Linear(modal_out_dim, modal_out_dim),
            nn.BatchNorm1d(modal_out_dim),
            nn.ReLU(),
        )

        if disease_aux_features is None:
            disease_aux_features = torch.zeros(counts.disease, 0, dtype=torch.float32)
        self.register_buffer("disease_aux_features", disease_aux_features)
        self.context_aux_dim = int(disease_aux_features.size(1))

        self.pathway_embedding = nn.Embedding(counts.pathway, modal_out_dim)
        self.ncrna_from_pathway = nn.Linear(modal_out_dim, modal_out_dim)
        self.drug_from_pathway = nn.Linear(modal_out_dim, modal_out_dim)
        self.disease_from_pathway = nn.Linear(modal_out_dim, modal_out_dim)
        self.pathway_triple_proj = nn.Sequential(
            nn.Linear(modal_out_dim, modal_out_dim),
            nn.BatchNorm1d(modal_out_dim),
            nn.ReLU(),
        )
        self.pathway_gate = nn.Sequential(
            nn.Linear(modal_out_dim * 2, modal_out_dim),
            nn.Sigmoid(),
        )
        if self.use_context_branch and self.context_aux_dim <= 0:
            raise ValueError("Context branch requires non-empty disease auxiliary features.")
        if self.context_aux_dim > 0:
            self.context_aux_proj = nn.Sequential(
                nn.Linear(self.context_aux_dim, modal_out_dim),
                nn.BatchNorm1d(modal_out_dim),
                nn.ReLU(),
            )
            self.context_gate = nn.Sequential(
                nn.Linear(modal_out_dim * 2, modal_out_dim),
                nn.Sigmoid(),
            )
        else:
            self.context_aux_proj = None
            self.context_gate = None
        num_branches = 2 + int(self.use_pathway_branch) + int(self.use_context_branch)
        self.classifier = HierarchicalClassifier(
            in_dim=modal_out_dim * num_branches,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.pathway_embedding.weight)
        for module in [
            self.ncrna_from_pathway,
            self.drug_from_pathway,
            self.disease_from_pathway,
        ]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for seq in [self.hgnn_triple_proj, self.pathway_triple_proj]:
            for module in seq:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        gate_linear = self.pathway_gate[0]
        nn.init.xavier_uniform_(gate_linear.weight)
        nn.init.constant_(gate_linear.bias, -1.0)
        if self.context_aux_proj is not None:
            for module in self.context_aux_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        if self.context_gate is not None:
            context_gate_linear = self.context_gate[0]
            nn.init.xavier_uniform_(context_gate_linear.weight)
            nn.init.constant_(context_gate_linear.bias, -1.0)

    def _rel(self, name: str, key: str) -> torch.Tensor:
        return self.relations[name][key]

    def _encode_pathway_branch(
        self,
        ncrna_ids: torch.Tensor,
        drug_ids: torch.Tensor,
        disease_ids: torch.Tensor,
        branch2_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pathway_emb = self.pathway_embedding.weight
        pathway_ncrna = self.ncrna_from_pathway(
            aggregate_mean(
                pathway_emb,
                self._rel("rna_pathway", "pathway_index"),
                self._rel("rna_pathway", "ncrna_index"),
                self.counts.ncrna,
                self._rel("rna_pathway", "weight"),
            )
        )
        pathway_drug = self.drug_from_pathway(
            aggregate_mean(
                pathway_emb,
                self._rel("drug_pathway", "pathway_index"),
                self._rel("drug_pathway", "drug_index"),
                self.counts.drug,
                self._rel("drug_pathway", "weight"),
            )
        )
        pathway_disease = self.disease_from_pathway(
            aggregate_mean(
                pathway_emb,
                self._rel("disease_pathway", "pathway_index"),
                self._rel("disease_pathway", "disease_index"),
                self.counts.disease,
                self._rel("disease_pathway", "weight"),
            )
        )
        pathway_triplet = (
            pathway_ncrna[ncrna_ids]
            + pathway_drug[drug_ids]
            + pathway_disease[disease_ids]
        ) / 3.0
        pathway_branch = self.pathway_triple_proj(pathway_triplet)
        gate = self.pathway_gate(torch.cat([branch2_feat, pathway_branch], dim=1))
        gated_branch = gate * pathway_branch
        return gated_branch, {
            "pathway_ncrna": pathway_ncrna,
            "pathway_drug": pathway_drug,
            "pathway_disease": pathway_disease,
            "pathway_branch": pathway_branch,
            "pathway_gate": gate,
        }

    def _encode_context_branch(
        self,
        disease_ids: torch.Tensor,
        branch2_feat: torch.Tensor,
        disease_aux_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.context_aux_proj is None or self.context_gate is None:
            raise RuntimeError("Context branch requested without configured context layers.")
        if disease_aux_override is None:
            context_input = self.disease_aux_features[disease_ids]
        else:
            context_input = disease_aux_override
        context_branch = self.context_aux_proj(context_input)
        gate = self.context_gate(torch.cat([branch2_feat, context_branch], dim=1))
        gated_branch = gate * context_branch
        return gated_branch, {
            "context_input": context_input,
            "context_branch": context_branch,
            "context_gate": gate,
        }

    def compute_triplet_static_state(
        self,
        ncrna_feat: torch.Tensor,
        drug_feat: torch.Tensor,
        disease_feat: torch.Tensor,
        ncrna_ids: torch.Tensor,
        drug_ids: torch.Tensor,
        disease_ids: torch.Tensor,
        hyperedge_index: torch.Tensor | None = None,
        sign_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        ncrna_aligned = self.ncrna_aligner(ncrna_feat)
        drug_aligned = self.drug_aligner(drug_feat)
        disease_aligned = self.disease_aligner(disease_feat)

        branch1_feat = self.cross_attention(
            ncrna_feat=ncrna_aligned,
            drug_feat=drug_aligned,
            disease_feat=disease_aligned,
            ncrna_ids=ncrna_ids,
            drug_ids=drug_ids,
            disease_ids=disease_ids,
        )

        all_entity_feat = torch.cat([ncrna_aligned, drug_aligned, disease_aligned], dim=0)
        if hyperedge_index is None or sign_weights is None:
            all_entity_hgnn = all_entity_feat
        else:
            all_entity_hgnn = self.hgnn_encoder(
                x=all_entity_feat,
                hyperedge_index=hyperedge_index,
                sign_weights=sign_weights,
            )

        ncrna_num = ncrna_feat.shape[0]
        drug_num = drug_feat.shape[0]
        batch_ncrna_hgnn = all_entity_hgnn[ncrna_ids]
        batch_drug_hgnn = all_entity_hgnn[ncrna_num + drug_ids]
        batch_disease_hgnn = all_entity_hgnn[ncrna_num + drug_num + disease_ids]
        branch2_feat = self.hgnn_triple_proj(
            (batch_ncrna_hgnn + batch_drug_hgnn + batch_disease_hgnn) / 3.0
        )

        pathway_branch_feat = None
        if self.use_pathway_branch:
            pathway_branch_feat, _ = self._encode_pathway_branch(
                ncrna_ids=ncrna_ids,
                drug_ids=drug_ids,
                disease_ids=disease_ids,
                branch2_feat=branch2_feat,
            )

        return {
            "branch1_feat": branch1_feat,
            "branch2_feat": branch2_feat,
            "pathway_branch_feat": pathway_branch_feat,
        }

    def predict_with_context_aux_override(
        self,
        ncrna_feat: torch.Tensor,
        drug_feat: torch.Tensor,
        disease_feat: torch.Tensor,
        ncrna_ids: torch.Tensor,
        drug_ids: torch.Tensor,
        disease_ids: torch.Tensor,
        disease_aux_override: torch.Tensor,
        hyperedge_index: torch.Tensor | None = None,
        sign_weights: torch.Tensor | None = None,
        static_state: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        if not self.use_context_branch:
            raise RuntimeError("Context auxiliary override requires the context branch to be enabled.")

        if disease_aux_override.ndim == 1:
            disease_aux_override = disease_aux_override.unsqueeze(0)

        if static_state is None:
            static_state = self.compute_triplet_static_state(
                ncrna_feat=ncrna_feat,
                drug_feat=drug_feat,
                disease_feat=disease_feat,
                ncrna_ids=ncrna_ids,
                drug_ids=drug_ids,
                disease_ids=disease_ids,
                hyperedge_index=hyperedge_index,
                sign_weights=sign_weights,
            )

        batch_size = disease_aux_override.size(0)

        def expand_state(name: str) -> torch.Tensor:
            value = static_state[name]
            if value is None:
                raise RuntimeError(f"Static state '{name}' is not available.")
            if value.size(0) == batch_size:
                return value
            if value.size(0) == 1:
                return value.repeat(batch_size, 1)
            raise ValueError(f"Static state '{name}' has incompatible batch size {value.size(0)} for override batch {batch_size}.")

        branch1_feat = expand_state("branch1_feat")
        branch2_feat = expand_state("branch2_feat")
        fused_parts = [branch1_feat, branch2_feat]

        pathway_branch_feat = static_state.get("pathway_branch_feat")
        if self.use_pathway_branch:
            if pathway_branch_feat is None:
                raise RuntimeError("Pathway branch is enabled but pathway static state is missing.")
            if pathway_branch_feat.size(0) == 1 and batch_size > 1:
                pathway_branch_feat = pathway_branch_feat.repeat(batch_size, 1)
            fused_parts.append(pathway_branch_feat)

        context_branch_feat, _ = self._encode_context_branch(
            disease_ids=disease_ids,
            branch2_feat=branch2_feat,
            disease_aux_override=disease_aux_override,
        )
        fused_parts.append(context_branch_feat)
        final_fused_feat = torch.cat(fused_parts, dim=1)
        logits, _, _ = self.classifier(final_fused_feat)
        return logits

    def forward(
        self,
        ncrna_feat: torch.Tensor,
        drug_feat: torch.Tensor,
        disease_feat: torch.Tensor,
        ncrna_ids: torch.Tensor,
        drug_ids: torch.Tensor,
        disease_ids: torch.Tensor,
        hyperedge_index: torch.Tensor | None = None,
        sign_weights: torch.Tensor | None = None,
        return_explain: bool = False,
    ):
        ncrna_aligned = self.ncrna_aligner(ncrna_feat)
        drug_aligned = self.drug_aligner(drug_feat)
        disease_aligned = self.disease_aligner(disease_feat)

        cross_attention_details = None
        if return_explain:
            branch1_feat, cross_attention_details = self.cross_attention(
                ncrna_feat=ncrna_aligned,
                drug_feat=drug_aligned,
                disease_feat=disease_aligned,
                ncrna_ids=ncrna_ids,
                drug_ids=drug_ids,
                disease_ids=disease_ids,
                return_details=True,
            )
        else:
            branch1_feat = self.cross_attention(
                ncrna_feat=ncrna_aligned,
                drug_feat=drug_aligned,
                disease_feat=disease_aligned,
                ncrna_ids=ncrna_ids,
                drug_ids=drug_ids,
                disease_ids=disease_ids,
            )

        all_entity_feat = torch.cat([ncrna_aligned, drug_aligned, disease_aligned], dim=0)
        if hyperedge_index is None or sign_weights is None:
            all_entity_hgnn = all_entity_feat
        else:
            all_entity_hgnn = self.hgnn_encoder(
                x=all_entity_feat,
                hyperedge_index=hyperedge_index,
                sign_weights=sign_weights,
            )

        ncrna_num = ncrna_feat.shape[0]
        drug_num = drug_feat.shape[0]
        batch_ncrna_hgnn = all_entity_hgnn[ncrna_ids]
        batch_drug_hgnn = all_entity_hgnn[ncrna_num + drug_ids]
        batch_disease_hgnn = all_entity_hgnn[ncrna_num + drug_num + disease_ids]
        branch2_feat = self.hgnn_triple_proj(
            (batch_ncrna_hgnn + batch_drug_hgnn + batch_disease_hgnn) / 3.0
        )

        pathway_details: dict[str, torch.Tensor] = {}
        context_details: dict[str, torch.Tensor] = {}
        fused_parts = [branch1_feat, branch2_feat]

        if self.use_pathway_branch:
            branch3_feat, pathway_details = self._encode_pathway_branch(
                ncrna_ids=ncrna_ids,
                drug_ids=drug_ids,
                disease_ids=disease_ids,
                branch2_feat=branch2_feat,
            )
            fused_parts.append(branch3_feat)
        else:
            branch3_feat = None

        if self.use_context_branch:
            context_branch_feat, context_details = self._encode_context_branch(
                disease_ids=disease_ids,
                branch2_feat=branch2_feat,
            )
            fused_parts.append(context_branch_feat)
        else:
            context_branch_feat = None

        final_fused_feat = torch.cat(fused_parts, dim=1)
        logits, pred_probs, pred_list = self.classifier(final_fused_feat)
        if not return_explain:
            return logits, pred_probs, pred_list

        explain_outputs = {
            "ncrna_aligned": ncrna_aligned,
            "drug_aligned": drug_aligned,
            "disease_aligned": disease_aligned,
            "all_entity_feat": all_entity_feat,
            "all_entity_hgnn": all_entity_hgnn,
            "branch1_feat": branch1_feat,
            "branch2_feat": branch2_feat,
            "final_fused_feat": final_fused_feat,
            "cross_attention": cross_attention_details,
            **pathway_details,
            **context_details,
        }
        if branch3_feat is not None:
            explain_outputs["branch3_feat"] = branch3_feat
        if context_branch_feat is not None:
            explain_outputs["context_branch_feat"] = context_branch_feat
        return logits, pred_probs, pred_list, explain_outputs

