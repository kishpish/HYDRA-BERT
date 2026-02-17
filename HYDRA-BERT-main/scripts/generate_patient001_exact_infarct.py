#!/usr/bin/env python3
"""
Patient SCD0000101 - Exact Infarct Visualization from VTK TissueType
====================================================================
Uses actual cell-based tissue classification from VTK file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import os
import meshio

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'figures' / 'patient_specific'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
})


def load_mesh_with_tissue_type():
    """Load mesh and compute point-level tissue classification from cell data."""
    _scd_models_dir = Path(os.environ.get('SCD_MODELS_DIR', 'SCD_MODELS'))
    mesh = meshio.read(str(_scd_models_dir / 'laplace_complete_v2' / 'SCD0000101' / 'SCD0000101_analysis.vtk'))

    tissue_type_cells = mesh.cell_data['TissueType'][0].flatten()
    cells = mesh.cells[0].data
    points = mesh.points

    n_points = len(points)

    # Count tissue type contributions for each point
    point_infarct_count = np.zeros(n_points)
    point_border_count = np.zeros(n_points)
    point_healthy_count = np.zeros(n_points)

    for i, cell in enumerate(cells):
        for pt_idx in cell:
            if tissue_type_cells[i] == 3:
                point_infarct_count[pt_idx] += 1
            elif tissue_type_cells[i] == 2:
                point_border_count[pt_idx] += 1
            else:
                point_healthy_count[pt_idx] += 1

    # Compute tissue type as ratio
    total_count = point_infarct_count + point_border_count + point_healthy_count

    # Assign tissue type based on majority
    tissue_type = np.ones(n_points)  # Default healthy

    infarct_ratio = np.zeros(n_points)
    border_ratio = np.zeros(n_points)

    valid = total_count > 0
    infarct_ratio[valid] = point_infarct_count[valid] / total_count[valid]
    border_ratio[valid] = point_border_count[valid] / total_count[valid]

    # Assign: 3=infarct if >30% infarct, 2=border if >30% border, else 1=healthy
    tissue_type[infarct_ratio > 0.3] = 3
    tissue_type[(border_ratio > 0.3) & (infarct_ratio <= 0.3)] = 2

    print(f"Tissue distribution (point-level, >30% threshold):")
    for t, name in [(1, 'Healthy'), (2, 'Border'), (3, 'Infarct')]:
        count = np.sum(tissue_type == t)
        print(f"  {name}: {count} ({count/n_points*100:.2f}%)")

    return points, tissue_type, infarct_ratio, border_ratio


def create_hydrogel_from_infarct_ratio(points, infarct_ratio, threshold=0.5, thickness=3.0):
    """
    Create hydrogel patch on points with high infarct ratio.

    Args:
        threshold: Minimum infarct ratio to include in hydrogel
                   0.5 = majority infarct (~3.8%)
                   0.3 = significant infarct (~6%)
                   0.8 = strict core (~0.7%)
    """
    hydrogel_mask = infarct_ratio >= threshold
    hydrogel_points = points[hydrogel_mask].copy()

    if len(hydrogel_points) == 0:
        print(f"  Warning: No points with infarct ratio >= {threshold}")
        return np.array([]), np.array([])

    # Offset outward for epicardial patch
    centroid = np.mean(hydrogel_points, axis=0)
    for i in range(len(hydrogel_points)):
        direction = hydrogel_points[i] - centroid
        norm = np.linalg.norm(direction)
        if norm > 0:
            hydrogel_points[i] += (direction / norm) * thickness * 0.1  # Scale for this coordinate system

    concentration = infarct_ratio[hydrogel_mask]

    print(f"  Hydrogel points: {len(hydrogel_points)} ({len(hydrogel_points)/len(points)*100:.2f}%)")
    print(f"  Threshold: {threshold*100:.0f}% infarct ratio")

    return hydrogel_points, concentration


def render_exact_infarct_mesh(points, tissue_type, hydrogel_points, hydrogel_conc,
                               filename='patient001_exact_infarct.png'):
    """Render with exact VTK tissue classification."""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Use jet colormap like original VTK visualization
    tissue_cmap = plt.cm.jet

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=tissue_type, cmap=tissue_cmap, vmin=1, vmax=3,
                        s=2, alpha=0.6, edgecolors='none')

    # Hydrogel with distinctive color and size based on concentration
    if len(hydrogel_points) > 0:
        sizes = 30 + hydrogel_conc * 50  # Larger for higher infarct ratio
        ax.scatter(hydrogel_points[:, 0], hydrogel_points[:, 1], hydrogel_points[:, 2],
                  c='#00E5FF', s=sizes, alpha=0.9, edgecolors='#004D40',
                  linewidths=0.5, marker='o')

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, pad=0.08)
    cbar.set_label('Tissue Type', fontsize=10)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Healthy', 'Border', 'Infarct'])

    # Legend
    legend_elements = [
        Patch(facecolor='#00E5FF', edgecolor='#004D40', label='GelMA Hydrogel'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Patient SCD0000101 - Exact Infarct from VTK TissueType\n'
                 f'Hydrogel covers {len(hydrogel_points)} core infarct points',
                 fontsize=12, fontweight='bold')

    ax.view_init(elev=25, azim=45)

    # Set bounds
    max_range = np.array([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])]).max() / 2.0
    mid = np.mean(points, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def render_comparison_thresholds(points, tissue_type, infarct_ratio, filename='patient001_threshold_comparison.png'):
    """Compare different infarct thresholds for hydrogel coverage."""
    fig = plt.figure(figsize=(18, 12))

    thresholds = [
        (0.8, 'Strict Core (>80%)', '~0.7% coverage'),
        (0.5, 'Majority (>50%)', '~3.8% coverage'),
        (0.3, 'Significant (>30%)', '~6% coverage'),
        (0.1, 'Any Infarct (>10%)', '~15% coverage'),
    ]

    tissue_cmap = plt.cm.jet

    for idx, (thresh, title, desc) in enumerate(thresholds):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=tissue_type, cmap=tissue_cmap, vmin=1, vmax=3,
                  s=1, alpha=0.4, edgecolors='none')

        hydrogel_mask = infarct_ratio >= thresh
        hydrogel_pts = points[hydrogel_mask]

        if len(hydrogel_pts) > 0:
            # Offset slightly outward
            centroid = np.mean(hydrogel_pts, axis=0)
            offset_pts = hydrogel_pts.copy()
            for i in range(len(offset_pts)):
                direction = offset_pts[i] - centroid
                norm = np.linalg.norm(direction)
                if norm > 0:
                    offset_pts[i] += (direction / norm) * 0.3

            ax.scatter(offset_pts[:, 0], offset_pts[:, 1], offset_pts[:, 2],
                      c='#00E5FF', s=20, alpha=0.9, edgecolors='#004D40', linewidths=0.3)

        n_hydrogel = len(hydrogel_pts)
        pct = n_hydrogel / len(points) * 100

        ax.set_title(f'{title}\n{n_hydrogel} points ({pct:.2f}%)', fontsize=10, fontweight='bold')
        ax.view_init(elev=25, azim=45)

        max_range = np.array([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])]).max() / 2.0
        mid = np.mean(points, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    legend_elements = [
        Patch(facecolor=plt.cm.jet(0.0), label='Healthy'),
        Patch(facecolor=plt.cm.jet(0.5), label='Border Zone'),
        Patch(facecolor=plt.cm.jet(1.0), label='Infarct'),
        Patch(facecolor='#00E5FF', edgecolor='#004D40', label='GelMA Hydrogel'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    plt.suptitle('Patient SCD0000101: Hydrogel Coverage at Different Infarct Thresholds',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def render_multi_angle_exact(points, tissue_type, hydrogel_points, filename='patient001_exact_angles.png'):
    """Multi-angle view with exact infarct hydrogel."""
    fig = plt.figure(figsize=(18, 12))

    tissue_cmap = plt.cm.jet

    views = [
        (25, 45, 'Anterior-Lateral'),
        (25, 135, 'Posterior-Lateral'),
        (25, 225, 'Posterior'),
        (25, 315, 'Anterior'),
        (80, 45, 'Basal (Top)'),
        (-10, 45, 'Apical (Bottom)'),
    ]

    for idx, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=tissue_type, cmap=tissue_cmap, vmin=1, vmax=3,
                  s=1, alpha=0.5, edgecolors='none')

        if len(hydrogel_points) > 0:
            ax.scatter(hydrogel_points[:, 0], hydrogel_points[:, 1], hydrogel_points[:, 2],
                      c='#00E5FF', s=25, alpha=0.95, edgecolors='#004D40', linewidths=0.3)

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)

        max_range = np.array([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])]).max() / 2.0
        mid = np.mean(points, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    legend_elements = [
        Patch(facecolor=plt.cm.jet(0.0), label='Healthy'),
        Patch(facecolor=plt.cm.jet(0.5), label='Border Zone'),
        Patch(facecolor=plt.cm.jet(1.0), label='Infarct'),
        Patch(facecolor='#00E5FF', edgecolor='#004D40', label='GelMA Hydrogel'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    plt.suptitle(f'Patient SCD0000101: Hydrogel on Exact Infarct Core ({len(hydrogel_points)} points)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def create_rotation_gif(points, tissue_type, hydrogel_points, n_frames=36,
                         filename='patient001_exact_rotation.gif'):
    """Rotation animation with exact infarct hydrogel."""
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    tissue_cmap = plt.cm.jet
    max_range = np.array([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])]).max() / 2.0
    mid = np.mean(points, axis=0)

    def update(frame):
        ax.clear()

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=tissue_type, cmap=tissue_cmap, vmin=1, vmax=3,
                  s=1, alpha=0.5, edgecolors='none')

        if len(hydrogel_points) > 0:
            ax.scatter(hydrogel_points[:, 0], hydrogel_points[:, 1], hydrogel_points[:, 2],
                      c='#00E5FF', s=30, alpha=0.95, edgecolors='#004D40', linewidths=0.3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Patient SCD0000101: Exact Infarct with Hydrogel\n'
                     f'{len(hydrogel_points)} points',
                     fontsize=11, fontweight='bold')

        ax.view_init(elev=25, azim=frame * (360 / n_frames))
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)

    save_path = OUTPUT_DIR / filename
    print(f"  Saving animation ({n_frames} frames)...")
    anim.save(str(save_path), writer=PillowWriter(fps=10))
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("PATIENT SCD0000101 - EXACT INFARCT FROM VTK TISSUETYPE")

    print("\n1. Loading mesh with actual tissue classification...")
    points, tissue_type, infarct_ratio, border_ratio = load_mesh_with_tissue_type()

    print(f"\n2. Creating hydrogel patch on majority infarct (>50% threshold)...")
    hydrogel_points, hydrogel_conc = create_hydrogel_from_infarct_ratio(
        points, infarct_ratio, threshold=0.5, thickness=3.0
    )

    print("\n3. Generating visualizations...")

    print("\n  3a. Main mesh with exact infarct hydrogel...")
    render_exact_infarct_mesh(points, tissue_type, hydrogel_points, hydrogel_conc,
                              filename='patient001_exact_infarct.png')

    print("\n  3b. Threshold comparison...")
    render_comparison_thresholds(points, tissue_type, infarct_ratio,
                                  filename='patient001_threshold_comparison.png')

    print("\n  3c. Multi-angle views...")
    render_multi_angle_exact(points, tissue_type, hydrogel_points,
                              filename='patient001_exact_angles.png')

    print("\n  3d. Rotation animation...")
    create_rotation_gif(points, tissue_type, hydrogel_points,
                        n_frames=36, filename='patient001_exact_rotation.gif')

    print("VISUALIZATION COMPLETE")
    print(f"\nOutput: {OUTPUT_DIR}")

    for f in sorted(OUTPUT_DIR.glob("patient001_exact*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
