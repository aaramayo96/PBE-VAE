import os
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap
    import matplotlib.gridspec as gridspec
except ImportError:
    matplotlib = None
    plt = None
    mpatches = None
    PdfPages = None
    ListedColormap = None
    gridspec = None

def plot_diagnostic_dashboard(clustering_method: str, n_clusters: int, 
                              dim_results: List[Tuple], active_wvls: np.ndarray, 
                              vnir_mask: np.ndarray, swir_mask: np.ndarray, 
                              X_vnir: np.ndarray, X_swir: np.ndarray, 
                              y_target: np.ndarray, sil: float, db: float, 
                              output_dir: Union[str, Path]):
    """
    Renders 5-panel diagnostic dashboard for VAE latent space clustering.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_c_map = max(2, n_clusters)
    try:
        global_cmap = plt.colormaps['tab20'].resampled(n_c_map)
    except AttributeError:
        global_cmap = plt.cm.get_cmap('tab20', n_c_map)
        
    for dim_label, d_all, d_vnir, d_swir, d_mask in dim_results:
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 6, figure=fig)
        
        active_indices = np.where(d_mask)[0]
        np.random.shuffle(active_indices)
        
        ax_pca1 = fig.add_subplot(gs[0, 0:2])
        ax_pca1.scatter(d_vnir[active_indices, 0], d_vnir[active_indices, 1], c=y_target[active_indices], 
                        cmap=global_cmap, s=5, alpha=0.75, edgecolors='none', vmin=1, vmax=n_c_map)
        ax_pca1.set_title(f"{dim_label}: VNIR Only ($\leq$ 1.4 $\mu$m)")

        ax_pca2 = fig.add_subplot(gs[0, 2:4])
        ax_pca2.scatter(d_swir[active_indices, 0], d_swir[active_indices, 1], c=y_target[active_indices], 
                        cmap=global_cmap, s=5, alpha=0.75, edgecolors='none', vmin=1, vmax=n_c_map)
        ax_pca2.set_title(f"{dim_label}: SWIR Only ($>$ 1.0 $\mu$m)")

        ax_pca3 = fig.add_subplot(gs[0, 4:6])
        ax_pca3.scatter(d_all[active_indices, 0], d_all[active_indices, 1], c=y_target[active_indices], 
                        cmap=global_cmap, s=5, alpha=0.75, edgecolors='none', vmin=1, vmax=n_c_map)
        ax_pca3.set_title(f"{dim_label}: Fused (Spectral + DEM)")

        ax_spec_v = fig.add_subplot(gs[1, 0:3])
        ax_spec_s = fig.add_subplot(gs[1, 3:6])

        for i in range(1, n_clusters + 1):
            mask = (y_target == i)
            if np.any(mask):
                c_color = global_cmap(i - 1)
                
                v_wvls = active_wvls[vnir_mask]
                m_vnir = np.mean(X_vnir[mask], axis=0)
                vw_clean, vm_clean = [], []
                last_w = -1
                for w, m in zip(v_wvls, m_vnir):
                    if w > last_w:
                        vw_clean.append(w)
                        vm_clean.append(m)
                        last_w = w
                ax_spec_v.plot(vw_clean, vm_clean, label=f'Cluster {i}', color=c_color, lw=1.5)
                
                s_wvls = active_wvls[swir_mask]
                m_swir = np.mean(X_swir[mask], axis=0)
                sw_clean, sm_clean = [], []
                last_w = -1
                for w, m in zip(s_wvls, m_swir):
                    if w > last_w:
                        sw_clean.append(w)
                        sm_clean.append(m)
                        last_w = w
                ax_spec_s.plot(sw_clean, sm_clean, label=f'Cluster {i}', color=c_color, lw=1.5)

        ax_spec_v.set_title("VNIR Mean Reflectance")
        ax_spec_v.set_xlabel("Wavelength ($\mu$m)")
        ax_spec_v.set_ylabel("Reflectance")
        if n_clusters <= 20: 
            ax_spec_v.legend(ncol=3, fontsize='x-small')
        ax_spec_v.grid(True, alpha=0.2)

        ax_spec_s.set_title("SWIR Mean Reflectance")
        ax_spec_s.set_xlabel("Wavelength ($\mu$m)")
        ax_spec_s.set_ylabel("Reflectance")
        ax_spec_s.grid(True, alpha=0.2)
        
        plt.suptitle(f"{clustering_method.upper()} Diagnostic Dashboard ({dim_label}) | {n_clusters} Clusters | Silhouette: {sil:.2f} / DB_Index: {db:.2f}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = output_dir / f"{clustering_method}_{dim_label.lower()}_diagnostic.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

def build_pdf_report(pdf_path: Union[str, Path], key: str, prob_map: np.ndarray, 
                     known_rois_px: List[Tuple], new_rois: List[Dict], 
                     threshold: float, valid_mask: np.ndarray):
    """
    Generates target detection PDF report for a single flight line.
    """
    pdf_path = Path(pdf_path)
    with PdfPages(str(pdf_path)) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        fig.patch.set_facecolor("#1a1a2e")

        ax = axes[0]
        disp_prob = np.where(valid_mask, prob_map, np.nan)
        im = ax.imshow(disp_prob, cmap="inferno", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Target Probability", color="white", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

        for name, col, row in known_rois_px:
            ax.plot(col, row, marker="*", markersize=12, color="#FFD700", markeredgecolor="white", markeredgewidth=0.5, zorder=5)
            ax.annotate(name, (col, row), textcoords="offset points", xytext=(6, 4), fontsize=5.5, color="#FFD700",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#1a1a2e", alpha=0.6, ec="none"))

        ax.set_title(f"Probability Map – {key}", color="white", fontsize=12, pad=8)
        ax.set_xlabel("Column (pixels)", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("Row (pixels)", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        known_patch = mpatches.Patch(color="#FFD700", label="Known ROI")
        ax.legend(handles=[known_patch], loc="upper right", framealpha=0.5, facecolor="#1a1a2e", edgecolor="#444444", labelcolor="white", fontsize=8)

        ax2 = axes[1]
        det_map = np.where(valid_mask, (prob_map >= threshold).astype(float), np.nan)
        ax2.imshow(det_map, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", interpolation="nearest")

        for i, roi in enumerate(new_rois):
            r0, c0, r1, c1 = roi["bbox"]
            rect = mpatches.Rectangle((c0, r0), c1 - c0, r1 - r0, linewidth=1.5, edgecolor="#FF4444", facecolor="none", zorder=5)
            ax2.add_patch(rect)
            ax2.text(c0, r0 - 3, f"NEW-{i+1}\n({roi['lat']:.5f}°,{roi['lon']:.5f}°)", fontsize=4.5, color="#FF6666",
                     bbox=dict(boxstyle="round,pad=0.1", fc="#1a1a2e", alpha=0.65, ec="none"))

        for name, col, row in known_rois_px:
            ax2.plot(col, row, marker="*", markersize=12, color="#FFD700", markeredgecolor="white", markeredgewidth=0.5, zorder=6)

        ax2.set_title(f"Detection Map (threshold={threshold:.2f}) – {len(new_rois)} NEW ROI(s)", color="white", fontsize=12, pad=8)
        ax2.set_xlabel("Column (pixels)", color="#aaaaaa", fontsize=8)
        ax2.tick_params(colors="#aaaaaa")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444444")

        new_patch = mpatches.Patch(color="#FF4444", label="New ROI (detected)")
        det_patch = mpatches.Patch(color="#00CC66", label="Target detection")
        bg_patch = mpatches.Patch(color="#CC3300", label="Background / No target")
        ax2.legend(handles=[det_patch, bg_patch, new_patch, known_patch], loc="upper right", framealpha=0.5, facecolor="#1a1a2e", edgecolor="#444444", labelcolor="white", fontsize=7)

        plt.suptitle(f"Target-Detection Report — {key}", color="white", fontsize=14, y=1.01)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
