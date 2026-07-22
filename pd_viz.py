"""Plot methods for parameter decompositions.

`PDViz` is a STATELESS mixin inherited by `pd.PD`, so every decomposition object
exposes its figures directly (`pd_obj.plot_components()`, `pd_obj.plot_faithfulness()`,
...) in the same house style as `mlp.MLPBase`'s in-class plots -- while `pd.py` itself
stays pure decomposition math (this module owns the matplotlib dependency).

The mixin uses only the public PD API: `C`, `model`, `component_signatures`,
`component_effective_norms`, `component_matrices`, `component_feature_assignment`,
`effective_reconstruction`. It holds no state and adds nothing to `state_dict`, so
saved decompositions are unaffected.

`plot_components()` is the aggregator: it computes the per-component SVD signatures
ONCE and threads them (`_ctx`) into the per-view methods, preserving the
single-SVD-pass behaviour; each per-view method also works standalone (recomputing
its own context when called directly).
"""
import math

import numpy as np
import matplotlib.pyplot as plt


def component_title(view: str, params: dict | None, diagnostic: str = "") -> str:
    """Two-line figure title identifying the decomposition's model:
        '<view> — <Model>  (d_emb=.., p=.., n=.., <method>)'
        'C=.. | k=.. | rank=..  |  <diagnostic>'
    A human-readable headline (view + model) over the fixed decomposition params,
    so the model a decomposition belongs to is obvious from the image."""
    p = params or {}
    bits = []
    if p.get("d_emb") not in (None, "NA"):
        bits.append(f"d_emb={p['d_emb']}")
    if p.get("p") is not None:
        bits.append(f"p={p['p']}")
    if p.get("n_exp") is not None:
        bits.append(f"n={float(p['n_exp']):g}")   # inline of results_io._fmt_n
    if p.get("method"):
        bits.append(str(p["method"]))
    model_line = str(p.get("model", ""))
    if bits:
        model_line += f"  ({', '.join(bits)})"
    head = f"{view} — {model_line}" if model_line else view
    fixed = " | ".join(f"{k}={p[k]}" for k in ("C", "k", "rank") if p.get(k) is not None)
    second = f"{fixed}  |  {diagnostic}" if fixed and diagnostic else (fixed or diagnostic)
    return f"{head}\n{second}" if second else head


def _permutation_order(assign: np.ndarray) -> np.ndarray:
    """Row order that clusters components by the feature they most own, so a
    clean (permutation-like) decomposition lands on the diagonal."""
    peak = assign.max(axis=1)
    argf = assign.argmax(axis=1)
    return np.lexsort((-peak, argf))   # primary: owned feature; tiebreak: peak


class PDViz:
    """Figure suite for a fitted decomposition, built on the fact that each
    component is (at rank 1) a rank-1 object = (read profile u_c, write profile
    v_c, coupling). Mixed into `pd.PD`; see module docstring."""

    # ---- shared per-view context (one SVD pass over all components) ----

    def _view_context(self, top: int) -> dict:
        eff_norm = self.component_effective_norms().cpu().numpy()
        order = np.argsort(eff_norm)[::-1]
        denom = float(eff_norm.max()) + 1e-12
        reads_t, writes_t, reads_h_t, writes_h_t, _ = self.component_signatures()
        return {
            "eff_norm": eff_norm,
            "order": order,
            "n_used": int((eff_norm > 0.01 * denom).sum()),
            "top_idx": [int(c) for c in order[:min(top, self.C)]],
            "reads": reads_t.cpu().numpy(),
            "writes": writes_t.cpu().numpy(),
            "reads_h": reads_h_t.cpu().numpy(),
            "writes_h": writes_h_t.cpu().numpy(),
        }

    # ---- aggregator ----

    def plot_components(self, *, top: int = 16, params: dict | None = None,
                        usage: dict | None = None) -> dict:
        """The full component-view suite as {name: Figure}: profiles / assignment /
        [usage] / faithfulness / weights / weight_profiles / weight_matrices.
        `usage` (from APD.usage_stats / sample_usage_stats) adds the behavioural
        batch-top-k panel; `params` stamps a provenance title on each figure."""
        ctx = self._view_context(top)
        figs = {
            "profiles": self.plot_component_profiles(top=top, params=params, _ctx=ctx),
            "assignment": self.plot_component_assignment(params=params, _ctx=ctx),
        }
        if usage is not None:
            figs["usage"] = self.plot_usage(usage, params=params)
        figs["faithfulness"] = self.plot_faithfulness(params=params)
        figs["weights"] = self.plot_component_weights(params=params, _ctx=ctx)
        figs["weight_profiles"] = self.plot_component_weight_profiles(top=top, params=params,
                                                                      _ctx=ctx)
        wm = self.plot_component_weight_matrices(params=params, _ctx=ctx)
        if wm is not None:
            figs["weight_matrices"] = wm
        return figs

    # ---- individual views (each mirrors the former figutil.plot_components block) ----

    def plot_component_profiles(self, *, top: int = 16, params: dict | None = None,
                                _ctx: dict | None = None):
        """Top-`top` components (by effective norm): read profile |u_c| (blue, up)
        and write profile |v_c| (red, down). Aligned peaks = a clean self-map."""
        ctx = _ctx or self._view_context(top)
        top_idx, eff_norm = ctx["top_idx"], ctx["eff_norm"]
        reads, writes = ctx["reads"], ctx["writes"]
        nf = reads.shape[1]
        ncol = min(4, len(top_idx)) or 1
        nrow = math.ceil(len(top_idx) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.1 * nrow + 0.6),
                                 constrained_layout=True, squeeze=False)
        for i, c in enumerate(top_idx):
            ax = axes[i // ncol][i % ncol]
            u, v = np.abs(reads[c]), np.abs(writes[c])
            fr, fw = int(u.argmax()), int(v.argmax())
            ax.bar(np.arange(nf), u, color="steelblue", width=1.0)
            ax.bar(np.arange(nf), -v, color="indianred", width=1.0)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"c{c}: read {fr}{' = ' if fr == fw else ' → '}write {fw}   "
                         f"‖e‖={eff_norm[c]:.2g}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        for j in range(len(top_idx), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(component_title(
            "Read (blue ↑) / write (red ↓) feature profiles", params,
            f"top {len(top_idx)} by effective norm; aligned peaks = a clean self-map"),
            fontsize=11)
        return fig

    def plot_component_assignment(self, *, params: dict | None = None,
                                  _ctx: dict | None = None):
        """(C x n_features) heatmaps of reads |u| and writes |v|, rows sorted by
        effective norm. Clean = a permutation in BOTH with matching columns."""
        ctx = _ctx or self._view_context(0)
        order, n_used = ctx["order"], ctx["n_used"]
        ro, wo = np.abs(ctx["reads"][order]), np.abs(ctx["writes"][order])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True,
                                     sharey=True)
        im1 = a1.imshow(ro, aspect="auto", cmap="magma", vmin=0.0,
                        vmax=float(ro.max()) or 1.0)
        a1.set_title("reads |u|  (input feature that activates the component)", fontsize=10)
        a1.set_xlabel("feature index")
        a1.set_ylabel("component (sorted by effective norm)")
        fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.04)
        im2 = a2.imshow(wo, aspect="auto", cmap="magma", vmin=0.0,
                        vmax=float(wo.max()) or 1.0)
        a2.set_title("writes |v|  (output feature the component drives)", fontsize=10)
        a2.set_xlabel("feature index")
        fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.04)
        fig.suptitle(component_title(
            "Component read / write assignment", params,
            f"{n_used}/{self.C} active; clean = a permutation in BOTH with matching columns"),
            fontsize=11)
        return fig

    def plot_component_assignment_sorted(self, *, params: dict | None = None):
        """|diagonal| self-map mass with rows permutation-sorted by owned feature,
        so a clean decomposition lands on the diagonal."""
        assign = self.component_feature_assignment().cpu().numpy()
        order = _permutation_order(assign)
        fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 6))
        im = ax.imshow(assign[order], aspect="auto", cmap="magma",
                       vmin=0.0, vmax=float(assign.max()) or 1.0)
        ax.set_xlabel("feature index")
        ax.set_ylabel("component (permutation-sorted)")
        ax.set_title("|diagonal| self-map mass (permutation-sorted)")
        fig.suptitle(component_title("Component assignment (permutation-sorted)", params),
                     fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

    def plot_usage(self, usage: dict, *, params: dict | None = None):
        """Behavioural batch-top-k usage: per-component selection frequency + the
        components-active-per-example distribution. `usage` is the dict produced by
        APD.usage_stats / APD.sample_usage_stats (required -- only the APD family
        has a batch-top-k importance notion)."""
        freq = usage["selection_freq"].cpu().numpy()
        counts = usage["per_example_count"].cpu().numpy().astype(int)
        k = usage.get("k")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        a1.bar(np.arange(len(freq)), np.sort(freq)[::-1], color="steelblue")
        a1.set_xlabel("component (sorted by selection frequency)")
        a1.set_ylabel("selection frequency")
        a1.set_title(f"batch-top-k selection frequency "
                     f"({int((freq > 0).sum())}/{len(freq)} ever selected)", fontsize=10)
        a1.grid(True, alpha=0.3)
        a2.hist(counts, bins=np.arange(counts.min(), counts.max() + 2) - 0.5,
                color="slategray")
        if k is not None:
            a2.axvline(k, color="crimson", ls="--", lw=1.5, label=f"target k={k}")
            a2.legend()
        a2.set_xlabel("components active per example")
        a2.set_ylabel("# examples")
        a2.set_title(f"components per example (mean {counts.mean():.1f})", fontsize=10)
        a2.grid(True, alpha=0.3)
        fig.suptitle(component_title(
            "Component usage (attribution / batch-top-k)", params,
            "APD's behavioural importance, distinct from weight-space effective norm"),
            fontsize=11)
        return fig

    def plot_faithfulness(self, *, params: dict | None = None):
        """Model effective map vs reconstructed (Σ components) vs residual: WHERE
        the decomposition's weight reconstruction fails, in feature space."""
        model_eff_t, recon_eff_t = self.effective_reconstruction()
        model_eff = model_eff_t.cpu().numpy()
        recon_eff = recon_eff_t.cpu().numpy()
        resid = model_eff - recon_eff
        vmax = max(float(np.abs(model_eff).max()), float(np.abs(recon_eff).max())) or 1.0
        rmax = float(np.abs(resid).max()) or 1.0
        rel = float(np.linalg.norm(resid) / (np.linalg.norm(model_eff) + 1e-12))
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
        for ax, M, ttl, vm in [(axs[0], model_eff, "model effective map", vmax),
                               (axs[1], recon_eff, "reconstructed (Σ components)", vmax),
                               (axs[2], resid, f"residual (max {rmax:.1e})", rmax)]:
            im = ax.imshow(M, cmap="RdBu", vmin=-vm, vmax=vm)
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel("feature")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axs[0].set_ylabel("feature")
        fig.suptitle(component_title(
            "Faithfulness: model vs reconstructed effective map", params,
            f"relative residual {rel:.1e}"), fontsize=11)
        return fig

    def plot_component_weights(self, *, params: dict | None = None,
                               _ctx: dict | None = None):
        """The actual per-component WEIGHT vectors over the d_mlp hidden neurons,
        stacked into (C x d_mlp) signed heatmaps (for MLP these ARE
        w_in / w_out.T with rows = components = features)."""
        ctx = _ctx or self._view_context(0)
        order = ctx["order"]
        reads_h, writes_h = ctx["reads_h"], ctx["writes_h"]
        C, d_mlp = reads_h.shape
        rh_o, wh_o = reads_h[order], writes_h[order]
        # aspect="equal" (NOT "auto"): keep the true C x d_mlp proportions so a tall
        # (C > d_mlp) matrix is not stretched into a misleading square.
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(4 + d_mlp / 12, 2 + C / 12),
                                     constrained_layout=True, sharey=True)
        vr = float(np.abs(rh_o).max()) or 1.0
        vw = float(np.abs(wh_o).max()) or 1.0
        im1 = a1.imshow(rh_o, aspect="equal", cmap="RdBu", vmin=-vr, vmax=vr)
        a1.set_title("read-in weights  (component → hidden neuron)", fontsize=10)
        a1.set_xlabel(f"hidden neuron (d_mlp={d_mlp})")
        a1.set_ylabel(f"component c = feature c (C={C}, sorted by effective norm)")
        fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.04)
        im2 = a2.imshow(wh_o, aspect="equal", cmap="RdBu", vmin=-vw, vmax=vw)
        a2.set_title("write-out weights  (hidden neuron → component)", fontsize=10)
        a2.set_xlabel(f"hidden neuron (d_mlp={d_mlp})")
        fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.04)
        fig.suptitle(component_title(
            f"Component weights over hidden neurons  [{C}×{d_mlp}]", params,
            "signed; for MLP these are exactly w_in / w_out.T"), fontsize=11)
        return fig

    def plot_component_weight_profiles(self, *, top: int = 16, params: dict | None = None,
                                       _ctx: dict | None = None):
        """Per-component DETAIL: top-`top` components' signed read-in (blue) and
        write-out (red) weights across the d_mlp neurons."""
        ctx = _ctx or self._view_context(top)
        top_idx, eff_norm = ctx["top_idx"], ctx["eff_norm"]
        reads_h, writes_h = ctx["reads_h"], ctx["writes_h"]
        d_mlp = reads_h.shape[1]
        ncol = min(4, len(top_idx)) or 1
        nrow = math.ceil(len(top_idx) / ncol)
        # sharey: all panels are top-by-effective-norm, so a common weight scale is
        # meaningful and lets read-in/write-out magnitudes be compared across components.
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.1 * nrow + 0.6),
                                 constrained_layout=True, squeeze=False, sharey=True)
        xh = np.arange(d_mlp)
        for i, c in enumerate(top_idx):
            ax = axes[i // ncol][i % ncol]
            r, w = reads_h[c], writes_h[c]
            ax.bar(xh, r, color="steelblue", width=1.0, alpha=0.75, label="read-in")
            ax.bar(xh, w, color="indianred", width=1.0, alpha=0.6, label="write-out")
            ax.axhline(0, color="k", lw=0.5)
            peak = int(np.abs(r).argmax())
            ax.set_title(f"c{c}: peak neuron {peak}   ‖e‖={eff_norm[c]:.2g}", fontsize=8)
            ax.set_xticks([])
            # y-axis: tick labels on the left column only (shared scale), keep others clean
            ax.tick_params(axis="y", labelsize=6, labelleft=(i % ncol == 0))
        for j in range(len(top_idx), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        if top_idx:
            axes[0][0].legend(fontsize=6, loc="upper right")
        fig.supylabel("weight value", fontsize=9)
        fig.suptitle(component_title(
            "Read-in (blue) / write-out (red) weights over hidden neurons", params,
            f"top {len(top_idx)} by effective norm; x = hidden neuron (0..{d_mlp-1})"),
            fontsize=11)
        return fig

    def plot_component_weight_matrices(self, *, top: int = 6, params: dict | None = None,
                                       _ctx: dict | None = None):
        """FULL per-component weight matrices P_in[c] / P_out[c] at their native
        W_in / W_out shape (ExactPD: one active row / column per component; APD/SPD:
        a dense rank-1 outer product). Returns None when there are no components."""
        ctx = _ctx or self._view_context(top)
        top_idx = ctx["top_idx"]
        n_mat = min(top, 6, len(top_idx))
        if not n_mat:
            return None
        fig, axes = plt.subplots(n_mat, 2, figsize=(9, 2.4 * n_mat + 0.6),
                                 constrained_layout=True, squeeze=False)
        for i in range(n_mat):
            c = top_idx[i]
            w_in_c, w_out_c = self.component_matrices(c)
            w_in_c = w_in_c.detach().cpu().numpy()
            w_out_c = w_out_c.detach().cpu().numpy()
            for ax, M, lbl in ((axes[i][0], w_in_c, "P_in[c]  (n_features × d_mlp)"),
                               (axes[i][1], w_out_c, "P_out[c]  (d_mlp × n_features)")):
                vm = float(np.abs(M).max()) or 1.0
                im = ax.imshow(M, aspect="auto", cmap="RdBu", vmin=-vm, vmax=vm)
                ax.set_title(f"c{c}: {lbl}  [{M.shape[0]}×{M.shape[1]}]", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(component_title(
            "Full per-component weight matrices (native W_in / W_out shape)", params,
            f"top {n_mat} by effective norm; ExactPD: one active row / column per component"),
            fontsize=11)
        return fig
