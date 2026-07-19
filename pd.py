from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call
from jaxtyping import Float, Int

from data import CleanDataset
from pd_viz import PDViz


class PD(PDViz, nn.Module, ABC):
    """
    Base class for PD methods (APD, SPD, ...).
    Subclasses to implement:
        - `masked_weight(name, mask)`
        - `method_loss_terms(...)`
    Plot methods (plot_components, plot_faithfulness, ...) come from the stateless
    PDViz mixin (pd_viz.py), which owns the matplotlib dependency.
    """
    init_scale: float = 0.01  # std of freshly initialized components

    def __init__(self, model: nn.Module, C: int, param_names: list[str] | None = None) -> None:
        super().__init__()
        self.model = model.requires_grad_(False) 
        self.C = C
        self.param_names = param_names or [
            name for name, p in model.named_parameters() if p.dim() >= 2
        ]
        all_params = dict(model.named_parameters())
        self.targets: dict[str, Tensor] = {name: all_params[name] for name in self.param_names}
        self.n_weight_params = sum(t.numel() for t in self.targets.values())

    @staticmethod
    def _key(name: str) -> str:
        """ParameterDict keys can't contain '.', so dotted paths are escaped."""
        return name.replace(".", "__")

    @abstractmethod
    def masked_weight(
        self, name: str, mask: Float[Tensor, "C"] | None
    ) -> Float[Tensor, "dim0 dim1"]:
        """Combine only the masked-in components into a full weight matrix;
        mask=None means all components (APD: indexed sum of full-rank
        components; SPD: (U * mask) @ V)."""
        ...

    def reconstructed_weight(self, name: str) -> Float[Tensor, "dim0 dim1"]:
        """Full reconstruction == masked_weight with every component active."""
        return self.masked_weight(name, None)

    def forward_masked(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        mask: dict[str, Float[Tensor, "C"]],
    ) -> Float[Tensor, "batch ..."]:
        """Model's own forward with each param rebuilt from only its masked-in
        components. Batch-shared masks only (one mask for the whole batch):
        APD's training masks and any method's post-hoc component ablation.
        SPD's per-example stochastic training masks cannot use this route --
        they are applied inline via decomp_op."""
        params = {name: self.masked_weight(name, mask[name]) for name in self.param_names}
        return functional_call(self.model, params, (x,))

    @abstractmethod
    def method_loss_terms(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        target_output: Float[Tensor, "batch ..."], **kwargs
    ) -> dict[str, tuple[Float[Tensor, ""], float]]:
        """Method-specific loss terms as {name: (term, weight)}. Faithfulness is
        added by the base class; subclasses supply everything else (APD:
        minimality + simplicity; SPD: recon + layerwise recon + importance)."""
        ...

    def loss(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        target_output: Float[Tensor, "batch ..."],
        alpha_faithfulness: float = 1.0, **kwargs
    ) -> tuple[Float[Tensor, ""], dict[str, float]]:
        """Template shared by all PD methods: faithfulness + method terms.
        Method-specific hyperparameters (APD's k, SPD's pnorm, all alphas)
        pass through **kwargs to method_loss_terms."""
        terms = {"faithful": (self.faithfulness_loss(), alpha_faithfulness)}
        terms |= self.method_loss_terms(x, target_output, **kwargs)
        return self._total_and_logs(terms)

    def faithfulness_loss(self) -> Float[Tensor, ""]:
        """Global MSE over all decomposed weights: sum of squared errors across
        ALL matrices / total weight-parameter count (spd-paper _calc_tensors_mse).
        One global mean, not a sum of per-matrix means -- invariant to how many
        matrices are decomposed and to their relative sizes."""
        sq_err = torch.stack([
            ((self.reconstructed_weight(name) - t) ** 2).sum() for name, t in self.targets.items()
        ]).sum()
        return sq_err / self.n_weight_params

    def forward_dense(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."]
    ) -> Float[Tensor, "batch ..."]:
        """Faithful forward: the model's own forward() with every decomposed
        parameter replaced by its full reconstruction. Works for any subclass;
        needs no pd-integration in the model."""
        params = {name: self.reconstructed_weight(name) for name in self.param_names}
        return functional_call(self.model, params, (x,))

    @staticmethod
    def _total_and_logs(
        weighted: dict[str, tuple[Float[Tensor, ""], float]]
    ) -> tuple[Float[Tensor, ""], dict[str, float]]:
        """Assemble total = sum(alpha_i * l_i) and a logs dict from
        {name: (loss_term, weight)}. Keeps log keys consistent across methods."""
        total = sum(w * l for l, w in weighted.values())
        logs = {k: l.item() for k, (l, _) in weighted.items()} | {"total": total.item()}
        return total, logs

    def decomp_op(
        self, name: str,
        x: Float[Tensor, "batch dim0"] | Int[Tensor, "batch"],
        mode: str = "masked",
    ) -> Float[Tensor, "batch dim1"]:
        """
        Inline per-operation interception point, called by the target model's
        forward() when pd is passed. Only methods with per-example masking
        (SPD) need this; APD intercepts at the parameter level via
        functional_call and never uses it. Overridden by SPD.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support decomp_op -- it substitutes whole "
            f"parameter tensors via functional_call instead. Call "
            f"{type(self).__name__.lower()}.forward_dense(x) / .forward_masked(x, mask) "
            f"directly rather than model(x, pd=...)."
        )

    # ---- component inspection: feature-space views of the learned parts ----

    def _component_weight(self, name: str, c: int) -> Float[Tensor, "dim0 dim1"]:
        """The (dim0, dim1) weight matrix contributed by a SINGLE component c to
        parameter `name`. APD: the full component P_c. SPD: the rank-1 outer
        product U[:, c] V[c, :]. Subclasses implement."""
        raise NotImplementedError

    def component_matrices(self, c: int) -> tuple[Tensor, Tensor]:
        """(w_in, w_out) weight matrices of component c, raw (pre-embedding)."""
        return self._component_weight("w_in", c), self._component_weight("w_out", c)

    def component_effective(self, c: int) -> Float[Tensor, "n_features n_features"]:
        """Feature->feature linear map of component c ALONE, folded through the
        model's embedding -- the per-component analog of model.effective_matrix().
        A clean decomposition -> each is ~rank-1 localised on one feature."""
        w_in, w_out = self.component_matrices(c)
        return (self.model.project_in(w_in) @ self.model.project_out(w_out)).detach()

    def component_norms(self) -> Float[Tensor, "C"]:
        """Frobenius norm of each component (read-in and read-out stacked), for
        ranking which components the fit actually uses (many end up ~0)."""
        out = []
        for c in range(self.C):
            w_in, w_out = self.component_matrices(c)
            out.append((w_in.pow(2).sum() + w_out.pow(2).sum()).sqrt())
        return torch.stack(out).detach()

    def component_feature_assignment(self) -> Float[Tensor, "C n_features"]:
        """(C, n_features): each component's mass on each feature = |diagonal of
        its feature->feature map|. A near-permutation means one component per
        feature; smeared rows mean superposition across components."""
        return torch.stack([self.component_effective(c).diagonal().abs()
                            for c in range(self.C)]).detach()

    def component_effective_norms(self) -> Float[Tensor, "C"]:
        """Frobenius norm of each component's FEATURE-SPACE effective map (after the
        embedding projection). Unlike component_norms (raw weight space), this is
        how much a component actually contributes to the feature->feature
        computation: under a compressing embedding (d_emb < n_features) a component
        can carry large raw weight norm yet a near-zero effective map, so this is
        the honest 'is this component doing feature-space work' signal and the right
        axis to rank/threshold components by."""
        return torch.stack([self.component_effective(c).norm()
                            for c in range(self.C)]).detach()

    def component_diagonal_shares(self, eps: float = 1e-12) -> Float[Tensor, "C"]:
        """(C,): fraction of each component's |effective| mass that sits on the
        diagonal. ~1 => a clean self-feature map (monosemantic); low => the
        component does off-diagonal / cross-feature (superposed) work, which the
        diagonal-only assignment view renders as 'empty' even though the component
        is active."""
        out = []
        for c in range(self.C):
            e = self.component_effective(c).abs()
            out.append(e.diagonal().sum() / (e.sum() + eps))
        return torch.stack(out).detach()

    def component_singular_values(self, c: int, name: str = "w_in") -> Float[Tensor, "r"]:
        """Singular values of component c's `name` matrix. The simplicity loss is
        L_p on exactly these, so this reveals whether components are low-rank
        (rank-1 = one clean mechanism) as intended."""
        return torch.linalg.svdvals(self._component_weight(name, c)).detach()

    def _component_signature(
        self, c: int
    ) -> tuple[Tensor, Tensor, float, Tensor, Tensor]:
        """Rank-1 signature of component c from a SINGLE SVD of w_in_c / w_out_c,
        returning BOTH the feature-space (u, v, coupling) and hidden-space (read_h,
        write_h) directions. Shared by component_read_write_profiles /
        component_weight_profiles / component_signatures so a full inspection SVDs each
        component's matrices once instead of two-to-three times (the SVD is the cost;
        deriving both views from it is free)."""
        w_in_c, w_out_c = self.component_matrices(c)
        Ui, Si, Vi = torch.linalg.svd(w_in_c, full_matrices=False)
        Uo, So, Vo = torch.linalg.svd(w_out_c, full_matrices=False)
        # feature-space read/write directions (projected through the embedding)
        a_in = (Ui[:, 0] * Si[0]).unsqueeze(1)      # (d_in, 1) input direction
        b_out = (Vo[0, :] * So[0]).unsqueeze(0)     # (1, d_out) output direction
        u = self.model.project_in(a_in)[:, 0]       # (n_features,)
        v = self.model.project_out(b_out)[0, :]     # (n_features,)
        u = u * (1.0 if u[u.abs().argmax()] >= 0 else -1.0)
        v = v * (1.0 if v[v.abs().argmax()] >= 0 else -1.0)
        coupling = float((Vi[0, :] @ Uo[:, 0]).item())
        # hidden-space weight vectors (d_mlp side), sign tied to the FEATURE side so
        # the sign is meaningful (for ExactPD this recovers the true weight signs).
        s_in = 1.0 if Ui[Ui[:, 0].abs().argmax(), 0] >= 0 else -1.0
        s_out = 1.0 if Vo[0, Vo[0, :].abs().argmax()] >= 0 else -1.0
        read_h = Si[0] * Vi[0, :] * s_in
        write_h = So[0] * Uo[:, 0] * s_out
        return u.detach(), v.detach(), coupling, read_h.detach(), write_h.detach()

    def component_read_write_profiles(
        self, c: int
    ) -> tuple[Float[Tensor, "n_features"], Float[Tensor, "n_features"], float]:
        """The natural rank-1 signature of a component in FEATURE space:
          u_c -- read profile: which input feature switches the component on
          v_c -- write profile: which output feature it drives
          coupling -- alignment of its read-side hidden pattern with its write-side
                      hidden pattern (sign/strength of the through-connection).
        A rank-1 component's effective map is exactly coupling * u_c v_c^T, so (u,v)
        losslessly capture what a 100x100 effective heatmap only implies. Taken from
        the dominant singular directions of w_in_c / w_out_c (exact at rank=1;
        dominant mode at rank>1), projected to feature space so it works for both
        LinearMLP (identity projection) and EmbedMLP (folds w_e). Sign is fixed so
        each profile's largest-magnitude entry is positive (SVD sign is arbitrary)."""
        u, v, coupling, _, _ = self._component_signature(c)
        return u, v, coupling

    def component_signatures(
        self,
    ) -> tuple[Float[Tensor, "C n_features"], Float[Tensor, "C n_features"],
               Float[Tensor, "C d_mlp"], Float[Tensor, "C d_mlp"], Float[Tensor, "C"]]:
        """All per-component signatures in ONE SVD pass over the C components:
        (reads_feat, writes_feat, reads_hidden, writes_hidden, couplings). read_write_
        profiles() and component_weight_profiles() are thin views over this; a caller
        that needs both (e.g. plot_components) should call this once rather than both
        wrappers, which would SVD every component twice."""
        uu, vv, rh, wh, cc = [], [], [], [], []
        for c in range(self.C):
            u, v, coup, r_h, w_h = self._component_signature(c)
            uu.append(u); vv.append(v); rh.append(r_h); wh.append(w_h); cc.append(coup)
        return (torch.stack(uu), torch.stack(vv), torch.stack(rh), torch.stack(wh),
                torch.tensor(cc, device=uu[0].device))

    def read_write_profiles(
        self,
    ) -> tuple[Float[Tensor, "C n_features"], Float[Tensor, "C n_features"]]:
        """Read and write feature profiles for ALL components, each (C, n_features).
        The behavioural bipartite view of the decomposition: row c of `reads` is the
        input feature(s) component c responds to, row c of `writes` the output
        feature(s) it drives. (View over component_signatures.)"""
        reads_feat, writes_feat, _, _, _ = self.component_signatures()
        return reads_feat, writes_feat

    def component_weight_profiles(
        self,
    ) -> tuple[Float[Tensor, "C d_mlp"], Float[Tensor, "C d_mlp"]]:
        """The actual per-component WEIGHTS over the d_mlp hidden neurons (as opposed
        to read_write_profiles, which projects to feature space). For component c:
          reads_hidden[c]  -- how c reads INTO each hidden neuron (the read-in weight
                              vector over neurons)
          writes_hidden[c] -- how c writes OUT of each hidden neuron (the write-out
                              weight vector over neurons)
        Taken from the dominant singular mode of w_in_c / w_out_c but kept in HIDDEN
        space (the singular vector on the d_mlp side, scaled by its singular value),
        the exact hidden counterparts of read_write_profiles' u/v. Sign is fixed by the
        FEATURE side, so it is meaningful and exact for LinearMLP / ExactPD:
        reads_hidden[c] == w_in[c, :] and writes_hidden[c] == w_out[:, c] (row c / column
        c of the effective weights). For a raw-EmbedMLP APD fit, w_in_c lives in
        embedding space, so the sign is tied to an embedding direction rather than a
        feature -- the vector is still the true dominant hidden pattern, only its
        overall sign is embedding-relative. (View over component_signatures.)"""
        _, _, reads_hidden, writes_hidden, _ = self.component_signatures()
        return reads_hidden, writes_hidden

    def component_metrics(self, eps: float = 1e-12, active_frac: float = 0.01, *,
                          reads=None, writes=None, eff_norm=None, diag_share=None) -> dict:
        """Scalar diagnostics of how cleanly the fit decomposed, from the READ and
        WRITE feature profiles (not the |diagonal|, which conflates the two and
        hides genuine cross-feature components):
          - monosemanticity[c]   = sqrt(read_share * write_share) if the read and
                                   write profiles peak on the SAME feature, else 0.
                                   1.0 = component reads and writes one shared feature
                                   (a clean self-feature mechanism); 0 = it reads one
                                   feature and writes another (cross-feature), or is
                                   diffuse.
          - components_used       = # components whose FEATURE-SPACE effective map is
                                   >= active_frac of the largest (weight-space; label
                                   as such -- APD's behavioural importance is
                                   attribution/usage, surfaced separately).
          - features_covered      = # distinct features owned (read==write peak) by a
                                   used component
          - mean_monosemanticity  = mean monosemanticity over used components
          - mean_diagonal_share   = mean |effective| diagonal share over used comps
        Returned per-component AND aggregated so both can be stored.

        reads/writes/eff_norm/diag_share may be passed in precomputed (e.g. from a
        caller that already ran component_signatures / component_effective_norms) to
        avoid recomputing the per-component SVD pass; each defaults to computing itself."""
        if reads is None or writes is None:
            reads, writes = self.read_write_profiles()        # (C, n_features) each
        if eff_norm is None:
            eff_norm = self.component_effective_norms()       # (C,) feature-space magnitude
        if diag_share is None:
            diag_share = self.component_diagonal_shares(eps)  # (C,)
        ru, wv = reads.abs(), writes.abs()
        read_share = ru.max(dim=1).values / (ru.sum(dim=1) + eps)
        write_share = wv.max(dim=1).values / (wv.sum(dim=1) + eps)
        owned = ru.argmax(dim=1)
        aligned = owned == wv.argmax(dim=1)
        mono = torch.where(aligned, (read_share * write_share).sqrt(),
                           torch.zeros_like(read_share))       # (C,)
        used = eff_norm > active_frac * (eff_norm.max() + eps)
        covered = len(set(owned[used & aligned].tolist())) if (used & aligned).any() else 0
        # Participation ratio (Σw)^2 / Σw^2 on the effective norms: a dominance-robust
        # "effective number of components". Unlike components_used (a >1%-of-max count,
        # which one huge component can drag down to ~2), this is ~C when the mass is
        # spread evenly and ~1 when a single component dominates.
        eff_count = float((eff_norm.sum() ** 2 / (eff_norm.pow(2).sum() + eps)).item())
        return {
            "monosemanticity": mono.tolist(),
            "components_used": int(used.sum().item()),
            "effective_count": eff_count,
            "features_covered": int(covered),
            "mean_monosemanticity": float(mono[used].mean().item()) if used.any() else 0.0,
            "mean_diagonal_share": float(diag_share[used].mean().item()) if used.any() else 0.0,
        }

    # ---- faithfulness analysis: how well the decomposition reconstructs the model ----

    def effective_reconstruction(self) -> tuple[Float[Tensor, "n_features n_features"],
                                                Float[Tensor, "n_features n_features"]]:
        """(model_eff, recon_eff): the model's effective feature->feature map and the
        decomposition's reconstruction of it (Σ components, projected the same way
        the model projects -- project_in/out fold in w_e for EmbedMLP). The shared
        basis of residual_frobenius and plot_faithfulness."""
        model_eff = self.model.effective_matrix()
        recon_eff = (self.model.project_in(self.reconstructed_weight("w_in"))
                     @ self.model.project_out(self.reconstructed_weight("w_out"))).detach()
        return model_eff, recon_eff

    def residual_frobenius(self) -> tuple[float, float]:
        """Frobenius norm of the effective-map residual: raw ‖W_model - W_recon‖_F and
        the relative ‖resid‖_F / ‖W_model‖_F (dimensionless, comparable across widths)."""
        model_eff, recon_eff = self.effective_reconstruction()
        resid = (model_eff - recon_eff).norm().item()
        return float(resid), float(resid / (model_eff.norm().item() + 1e-12))

    def behavioural_faithfulness(self, p: float, *, batch: int = 4096,
                                 seed: int = 0) -> tuple[float, float]:
        """Output-space faithfulness of the FULL decomposition: forward_dense (every
        decomposed weight replaced by the Σ-all-components reconstruction) vs the
        model's own output on a fresh CleanDataset batch at sparsity p. Returns
        (out_mse, rel_out_err). Distinct from residual_frobenius (weight space)."""
        ds = CleanDataset(n_features=self.model.n_features, p=float(p),
                          device=self.model.device, seed=seed)
        x, _ = ds.generate_batch(batch)
        with torch.no_grad():
            target = self.model(x)
            diff = self.forward_dense(x) - target
            out_mse = float((diff ** 2).mean().item())
            rel = float((diff.norm() / (target.norm() + 1e-12)).item())
        return out_mse, rel


# ------------------------
# APD (Braun et al., 2025)
# ------------------------

class APD(PD):
    """Attribution-based Parameter Decomposition (Braun et al., 2025) with
    *batch top-k*: per-example attributions select ~k active components per data
    point (top k*batch across the batch), and minimality is the reconstruction
    error of the resulting per-example sparse forward pass."""

    def __init__(self, model: nn.Module, C: int, rank: int = 1,
                 param_names: list[str] | None = None) -> None:
        super().__init__(model, C, param_names)
        assert set(self.param_names) == {"w_in", "w_out"}, (
            "batch-top-k APD assumes a single-hidden-layer MLP decomposing "
            f"exactly w_in/w_out; got {self.param_names}"
        )
        # Apollo-APD low-rank FACTORED components: P_c = A_c @ B_c with inner
        # dim `rank`, so each component has rank <= rank (rank=1 => rank-1, like
        # SPD's U@V). This is deliberate: (a) the Schatten-p simplicity is then a
        # closed-form polynomial in A,B (no svdvals -> no ill-conditioned SVD
        # backward as components go low-rank), and (b) a rank-1 component cannot
        # absorb the whole model, so faithfulness alone forces many active
        # components instead of a single-component collapse.
        self.rank = rank
        fscale = self.init_scale ** 0.5      # so a rank-1 A@B entry ~ init_scale
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for name, t in self.targets.items():
            dim0, dim1 = t.shape
            key = self._key(name)
            self.A[key] = nn.Parameter(torch.randn(C, dim0, rank, device=t.device) * fscale)
            self.B[key] = nn.Parameter(torch.randn(C, rank, dim1, device=t.device) * fscale)

    def _components(self, name: str) -> Float[Tensor, "C dim0 dim1"]:
        """Materialise the full (C, dim0, dim1) component tensor P = A @ B from its
        low-rank factors. Kept OUT of the simplicity path (that is closed-form on
        the factors); used by the forward, attributions and inspection."""
        key = self._key(name)
        return torch.einsum("cir,crj->cij", self.A[key], self.B[key])

    def masked_weight(
        self, name: str, mask: Float[Tensor, "C"] | None
    ) -> Float[Tensor, "dim0 dim1"]:
        comps = self._components(name)
        if mask is None:
            return comps.sum(dim=0)  # theta_hat = sum_c P_c
        # Index only the k active components: k/C of the work of summing
        # all C with a mostly-zero mask, with identical gradients.
        return comps[mask.bool()].sum(dim=0)

    def _component_weight(self, name: str, c: int) -> Float[Tensor, "dim0 dim1"]:
        key = self._key(name)
        return self.A[key][c] @ self.B[key][c]

    def _component_forward(
        self, x: Float[Tensor, "batch n_features"], gates: Float[Tensor, "batch C"]
    ) -> Float[Tensor, "batch n_features"]:
        """Single-hidden-layer MLP forward with each component c's contribution to
        w_in AND w_out gated per-example by gates[b, c]. gates=ones reproduces the
        dense reconstruction (== forward_dense); a per-example binary mask gives
        the batch-top-k sparse forward. Uses the model's embed_input/unembed_output
        so it works for LinearMLP and EmbedMLP alike. Keeps the C dimension explicit
        (per-example masks can't be folded into one weight matrix) -- this is the
        cost of faithful batch top-k."""
        P_in = self._components("w_in")                        # (C, d_in, d_mlp)
        P_out = self._components("w_out")                      # (C, d_mlp, d_out)
        e = self.model.embed_input(x)                         # (B, d_in)
        h_contrib = torch.einsum("bi,cij->bcj", e, P_in)      # (B, C, d_mlp)
        pre_h = torch.einsum("bc,bcj->bj", gates, h_contrib)  # (B, d_mlp)
        h = F.relu(pre_h)
        o_contrib = torch.einsum("bj,cjk->bck", h, P_out)     # (B, C, d_out)
        m = torch.einsum("bc,bck->bk", gates, o_contrib)      # (B, d_out)
        return self.model.unembed_output(m)                   # (B, n_features)

    def _attributions(
        self, x: Float[Tensor, "batch n_features"],
        target_output: Float[Tensor, "batch n_features"],
    ) -> Float[Tensor, "batch C"]:
        """Per-(example, component) attribution A[b,c] = |dL/dg_{b,c}| at g=1, with
        L the reconstruction MSE. One backward w.r.t. the per-example gates;
        loss-based, matching OracleAPD's exact-ablation |delta MSE|."""
        gates = torch.ones(x.shape[0], self.C, device=x.device, requires_grad=True)
        loss = F.mse_loss(self._component_forward(x, gates), target_output)
        grad = torch.autograd.grad(loss, gates)[0]
        return grad.abs().detach()

    def _batch_topk_mask(
        self, attr: Float[Tensor, "batch C"], k: int
    ) -> Float[Tensor, "batch C"]:
        """Batch top-k (Bussmann et al., 2024): keep the top k*batch attributions
        across the whole (batch x C) matrix -> ~k components active per example on
        average, but each example selects its own set. Returns a (B, C) binary mask."""
        B, C = attr.shape
        kbt = max(1, min(B * C, int(round(k)) * B))
        idx = torch.topk(attr.reshape(-1), kbt).indices
        mask = torch.zeros(B * C, device=attr.device)
        mask[idx] = 1.0
        return mask.reshape(B, C)

    def usage_stats(
        self, x: Float[Tensor, "batch n_features"],
        target_output: Float[Tensor, "batch n_features"], k: int,
    ) -> dict:
        """Behavioural usage of the components on a batch: the batch-top-k selection
        APD actually makes (its native importance notion, absent from the weight-
        space metrics). Returns:
          attr (B,C), mask (B,C),
          selection_freq (C,)   = fraction of examples that select each component,
          per_example_count (B,)= # components active per example (should center on k).
        Lets the plot layer show usage without reaching into training internals."""
        attr = self._attributions(x, target_output)   # runs its own autograd, returns detached
        mask = self._batch_topk_mask(attr, k)
        return {"attr": attr, "mask": mask, "k": k,
                "selection_freq": mask.mean(dim=0),
                "per_example_count": mask.sum(dim=1)}

    def sample_usage_stats(self, p: float, k: int, *, batch: int = 1024,
                           seed: int = 0) -> dict:
        """usage_stats on a FRESH CleanDataset batch at sparsity p (target = the
        frozen model's own output) -- the standard recipe every plot layer used to
        duplicate inline. APD-family only (SPD has no batch-top-k notion)."""
        ds = CleanDataset(n_features=self.model.n_features, p=float(p),
                          device=self.model.device, seed=seed)
        x, _ = ds.generate_batch(batch)
        with torch.no_grad():
            target = self.model(x)
        return self.usage_stats(x, target, k)

    def simplicity_loss(
        self, mask: Float[Tensor, "batch C"], p: float = 1.0, eps: float = 1e-16
    ) -> Float[Tensor, ""]:
        """Apollo-APD Schatten-p simplicity, computed in CLOSED FORM from the
        low-rank factors A@B (spd/run_spd.py:calc_schatten_loss) -- no svdvals, so
        the gradient stays finite as a component collapses to low rank. For each
        component c and inner direction r, S_A[c,r]=‖A[c,:,r]‖^2, S_B[c,r]=‖B[c,r,:]‖^2,
        and the surrogate singular value is sqrt(S_A·S_B); the penalty is
            (1/(P·B)) Σ_b Σ_{c∈topk(b)} Σ_r (S_A[c,r]·S_B[c,r] + eps)^(p/2),
        i.e. the per-example, frequency-weighted Schatten-p over active components,
        normalised per weight-parameter and per example (matching Apollo). For rank
        1 this is exactly Σ σ_c^p over active components. `p`<1 makes it rank/
        component-sparsity inducing; the +eps guards the p<1 derivative at 0."""
        freq = mask.mean(dim=0)                                   # (C,) activation frequency
        penalty = mask.new_zeros(())
        for name in self.param_names:
            A = self.A[self._key(name)]                          # (C, dim0, rank)
            Bf = self.B[self._key(name)]                         # (C, rank, dim1)
            S_A = A.pow(2).sum(dim=1)                            # (C, rank) col-norms^2
            S_B = Bf.pow(2).sum(dim=2)                           # (C, rank) row-norms^2
            sch = (S_A * S_B + eps).pow(0.5 * p).sum(dim=1)      # (C,) per-component Schatten
            penalty = penalty + freq @ sch
        return penalty / self.n_weight_params

    def method_loss_terms(
        self, x: Float[Tensor, "batch n_features"],
        target_output: Float[Tensor, "batch n_features"],
        k: int, alpha_minimality: float = 1.0, alpha_simplicity: float = 1.0, p: float = 1.0,
    ) -> dict[str, tuple[Float[Tensor, ""], float]]:
        mask = self._batch_topk_mask(self._attributions(x, target_output), k)  # (B, C)
        out_masked = self._component_forward(x, mask)                          # sparse fwd
        return {
            "minimal": (F.mse_loss(out_masked, target_output), alpha_minimality),
            "simple": (self.simplicity_loss(mask, p=p), alpha_simplicity),
        }


class OracleAPD(APD):
    """APD with oracle (exact leave-one-component-out) per-example attributions:
    A[b,c] = |MSE(forward with component c removed) - MSE(all active)|.

    Vectorised: instead of C sequential forwards, all C leave-one-out ablations
    are computed in one batched pass along a new axis. The all-ones-minus-one
    gate makes each ablation's pre-activation the *full sum minus that one
    component's contribution*, so the per-component contributions (h_contrib)
    are computed ONCE and every ablation is a subtraction off the shared sum;
    the ReLU is then applied per ablation. Same result as the loop (see the
    equivalence test), ~C fewer kernel launches -- the whole point of the GPU
    run. The (C, B, d_out) intermediates are the memory cost (keep batch modest;
    ORACLE_BATCH already does)."""

    @staticmethod
    def _per_example_mse(
        out: Float[Tensor, "... n_features"], target: Float[Tensor, "batch n_features"]
    ) -> Tensor:
        """Mean squared error over features, per example. Handles a plain
        (B, n_features) output -> (B,) and a stacked (C, B, n_features) output
        (one slice per ablation) -> (C, B), so base and the ablations share it."""
        if out.dim() == target.dim():
            return ((out - target) ** 2).flatten(1).mean(dim=1)            # (B,)
        return ((out - target.unsqueeze(0)) ** 2).flatten(2).mean(dim=2)   # (C, B)

    def _attributions(
        self, x: Float[Tensor, "batch n_features"],
        target_output: Float[Tensor, "batch n_features"],
    ) -> Float[Tensor, "batch C"]:
        with torch.no_grad():
            P_in = self._components("w_in")                           # (C, d_in, d_mlp)
            P_out = self._components("w_out")                         # (C, d_mlp, d_out)
            e = self.model.embed_input(x)                            # (B, d_in)

            # Shared work, computed ONCE (independent of which component is gated).
            h_contrib = torch.einsum("bi,cij->bcj", e, P_in)         # (B, C, d_mlp)
            full_pre_h = h_contrib.sum(dim=1)                        # (B, d_mlp)
            P_out_sum = P_out.sum(dim=0)                             # (d_mlp, d_out)

            # base: every component active (gates = ones).
            m_full = F.relu(full_pre_h) @ P_out_sum                  # (B, d_out)
            base = self._per_example_mse(
                self.model.unembed_output(m_full), target_output)    # (B,)

            # All C leave-one-out ablations at once (axis a == removed component).
            # pre-activation with a removed = full sum - a's own contribution.
            pre_ablate = full_pre_h.unsqueeze(0) - h_contrib.permute(1, 0, 2)  # (C, B, d_mlp)
            h_ablate = F.relu(pre_ablate)                                       # (C, B, d_mlp)
            # output with a removed from BOTH layers: h_a @ (sum P_out) - h_a @ P_out[a].
            out_full = torch.einsum("abj,jk->abk", h_ablate, P_out_sum)         # (C, B, d_out)
            out_self = torch.einsum("abj,ajk->abk", h_ablate, P_out)            # (C, B, d_out)
            feats = self.model.unembed_output(out_full - out_self)             # (C, B, n_features)

            mse_ablate = self._per_example_mse(feats, target_output)           # (C, B)
            attr = (mse_ablate - base.unsqueeze(0)).abs().T                    # (B, C)
        return attr


# ---------------------------------------------------
# ExactPD: the handcoded per-feature ground-truth
# ---------------------------------------------------

class ExactPD(APD):
    """The exact, NON-fitted per-feature decomposition, used as a ground-truth
    reference for the learned methods (APD/OracleAPD/SPD). Component i is feature
    i's own mechanism: read-in = row i of the effective read-in matrix, write-out
    = column i of the effective read-out matrix, so C = n_features and each
    component is rank-1.

    Exact for BOTH architectures because the model is behaviourally a feature-space
    LinearMLP:
        model(x) == relu(x @ readin) @ readout,
    where readin = readin() (n_features, d_mlp) and readout = readout()
    (d_mlp, n_features). (For EmbedMLP the linear embedding folds outside the ReLU:
    readin = w_e @ w_in, readout = w_out @ w_e.T; for LinearMLP they are just
    w_in/w_out.) Splitting those folded matrices by row / column sums back exactly,
    so faithfulness == 0 by construction.

    Implemented by reusing ALL of APD's machinery on a feature-space LinearMLP
    SURROGATE whose weights are readin()/readout(): a LinearMLP has identity
    projection/embedding hooks, so the inherited forward, batch-top-k attributions,
    component metrics and viz all work unchanged and uniformly. The per-feature
    rank-1 A@B factors are set analytically and frozen (requires_grad=False), so the
    method has NO trainable parameters and does zero fit steps. Because we force
    each component to read feature i and write feature i, monosemanticity == 1 by
    construction (a sanity marker); the informative outputs are the minimality /
    simplicity / behavioural terms -- what the ideal decomposition scores."""

    def __init__(self, model: nn.Module, C: int | None = None, rank: int = 1,
                 param_names: list[str] | None = None) -> None:
        from mlp import LinearMLP  # deferred: mlp imports pd (circular at module load)

        n_features, d_mlp = model.n_features, model.d_mlp
        readin = model.readin().detach()    # (n_features, d_mlp)   feature-space read-in
        readout = model.readout().detach()  # (d_mlp, n_features)   feature-space read-out

        # Feature-space linear surrogate: relu(x @ readin) @ readout == model(x).
        surrogate = LinearMLP(n_features=n_features, d_mlp=d_mlp, device=model.device)
        with torch.no_grad():
            surrogate.w_in.copy_(readin)
            surrogate.w_out.copy_(readout)

        # Reuse APD's A/B scaffolding + the w_in/w_out assertion; C = one per feature.
        super().__init__(surrogate, C=n_features, rank=1, param_names=param_names)

        # Overwrite the random factors with the exact per-feature rank-1 components,
        # frozen so they are excluded from the optimiser and never drift.
        #   w_in  (n_features, d_mlp):    P_in[i]  = e_i (row i) outer readin[i, :]
        #   w_out (d_mlp, n_features):    P_out[i] = readout[:, i] outer e_i (col i)
        eye = torch.eye(n_features, device=model.device)
        self.A[self._key("w_in")] = nn.Parameter(
            eye.unsqueeze(-1), requires_grad=False)            # (C, n_features, 1)
        self.B[self._key("w_in")] = nn.Parameter(
            readin.unsqueeze(1), requires_grad=False)          # (C, 1, d_mlp)
        self.A[self._key("w_out")] = nn.Parameter(
            readout.T.unsqueeze(-1), requires_grad=False)      # (C, d_mlp, 1)
        self.B[self._key("w_out")] = nn.Parameter(
            eye.unsqueeze(1), requires_grad=False)             # (C, 1, n_features)


# -----------------------------------
# SPD (Bushnaq, Braun, Sharkey, 2025)
# -----------------------------------

def leaky_hard_sigmoid_lower(x: Float[Tensor, "..."], slope: float = 0.01) -> Float[Tensor, "..."]:
    return torch.where(x < 0, slope * x, x.clamp(max=1.0))


def leaky_hard_sigmoid_upper(x: Float[Tensor, "..."], slope: float = 0.01) -> Float[Tensor, "..."]:
    return torch.where(x > 1, 1 + slope * (x - 1), x.clamp(min=0.0))


def _init_param_(param: Tensor, fan_val: float, nonlinearity: str = "linear") -> Tensor:
    gain = nn.init.calculate_gain(nonlinearity)
    with torch.no_grad():
        param.normal_(0.0, gain * fan_val ** -0.5)
    return param


class AffineGate(nn.Module):
    def __init__(self, C: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(_init_param_(torch.empty(C), fan_val=1))
        self.bias = nn.Parameter(torch.zeros(C))

    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        return x * self.weight + self.bias


class PerComponentGateMLP(nn.Module):
    def __init__(self, C: int, n_neurons: int) -> None:
        super().__init__()
        self.mlp_in = nn.Parameter(_init_param_(torch.empty(C, n_neurons), fan_val=1, nonlinearity="relu"))
        self.in_bias = nn.Parameter(torch.zeros(C, n_neurons))
        self.mlp_out = nn.Parameter(_init_param_(torch.empty(C, n_neurons), fan_val=n_neurons))
        self.out_bias = nn.Parameter(torch.zeros(C))

    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        hidden = F.gelu(x.unsqueeze(-1) * self.mlp_in + self.in_bias)      # ... C n
        return (hidden * self.mlp_out).sum(dim=-1) + self.out_bias          # ... C


class SPD(PD):
    """
    Stochastic Parameter Decomposition (Bushnaq, Braun, Sharkey, 2025),
    cross-checked against github.com/goodfire-ai/spd.
    """

    def __init__(self, model: nn.Module, C: int, param_specs: dict[str, str],
                 gate_type: str = "per_component_mlp", gate_hidden: int = 16) -> None:
        super().__init__(model, C, param_names=list(param_specs))
        self.param_specs = param_specs

        self.U, self.V, self.gate = nn.ParameterDict(), nn.ParameterDict(), nn.ModuleDict()
        for name, t in self.targets.items():
            key = self._key(name)
            self.U[key] = nn.Parameter(
                _init_param_(torch.empty(t.shape[0], C, device=t.device), fan_val=t.shape[1]))
            self.V[key] = nn.Parameter(
                _init_param_(torch.empty(C, t.shape[1], device=t.device), fan_val=C))
            gate = (PerComponentGateMLP(C, gate_hidden) if gate_type == "per_component_mlp"
                    else AffineGate(C) if gate_type == "affine"
                    else None)
            if gate is None:
                raise ValueError(f"Unknown gate_type '{gate_type}'")
            self.gate[key] = gate.to(t.device)
        self._logits: dict[str, Float[Tensor, "batch C"]] = {}
        self._active: set[str] | None = None

    def masked_weight(
        self, name: str, mask: Float[Tensor, "C"] | None
    ) -> Float[Tensor, "dim0 dim1"]:
        """W^l ~= sum_c m_c U_c V_c = (U * m) @ V. Not used in SPD training
        (whose masks are per-example, applied inline in decomp_op) -- this
        serves faithfulness and batch-shared post-hoc component ablation."""
        key = self._key(name)
        U = self.U[key] if mask is None else self.U[key] * mask
        return U @ self.V[key]

    def _component_weight(self, name: str, c: int) -> Float[Tensor, "dim0 dim1"]:
        key = self._key(name)
        return torch.outer(self.U[key][:, c], self.V[key][c])  # rank-1 U_c V_c

    def decomp_op(
        self, name: str,
        x: Float[Tensor, "batch dim0"] | Int[Tensor, "batch"],
        mode: str = "masked",
    ) -> Float[Tensor, "batch dim1"]:
        key = self._key(name)
        op = self.param_specs[name]

        if mode == "masked" and self._active is not None and name not in self._active:
            t = self.targets[name]
            return x @ t if op == "matmul" else t[x]

        U, V = self.U[key], self.V[key]
        if op == "matmul":
            inner = x @ U
        elif op == "embedding":
            inner = U[x]
        else:
            raise ValueError(f"Unknown op '{op}' for parameter '{name}'")

        if mode == "dense":
            return inner @ V
        elif mode == "masked":
            logit = self.gate[key](inner)
            self._logits[name] = logit
            g = leaky_hard_sigmoid_lower(logit)
            mask = g + (1 - g) * torch.rand_like(g)
            return (inner * mask) @ V
        raise ValueError(
            f"decomp_op got unexpected mode '{mode}' -- did you mean 'dense' or 'masked'? "
            f"('err' should never reach decomp_op -- call model(x) without pd instead.)"
        )

    def _masked_forward(self, x, active: set[str] | None) -> Float[Tensor, "batch ..."]:
        self._active = active
        try:
            return self.model(x, pd=self, mode="masked")
        finally:
            self._active = None

    def method_loss_terms(
        self, x: Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."],
        target_output: Float[Tensor, "batch ..."],
        alpha_recon: float = 1.0,
        alpha_recon_layerwise: float = 1.0,
        alpha_importance: float = 1e-5,
        pnorm: float = 2.0,
    ) -> dict[str, tuple[Float[Tensor, ""], float]]:
        self._logits = {}
        out = self._masked_forward(x, active=None)
        l_recon = F.mse_loss(out, target_output)
        l_importance = torch.stack([
            (leaky_hard_sigmoid_upper(logit) ** pnorm).sum(dim=-1).mean()
            for logit in self._logits.values()
        ]).sum()

        l_layerwise = torch.stack([
            F.mse_loss(self._masked_forward(x, active={name}), target_output)
            for name in self.param_names
        ]).mean()

        return {
            "recon": (l_recon, alpha_recon),
            "recon_layerwise": (l_layerwise, alpha_recon_layerwise),
            "importance": (l_importance, alpha_importance),
        }
