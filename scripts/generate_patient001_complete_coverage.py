#!/usr/bin/env python3
"""
Patient SCD0000101 - Complete Infarct Coverage Visualization
Hydrogel covers ALL points touching infarct cells for continuous coverage.
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


def load_mesh_and_classify():
    """Load mesh and get complete tissue classification."""
    _scd_models_dir = Path(os.environ.get('SCD_MODELS_DIR', 'SCD_MODELS'))
    mesh = meshio.read(str(_scd_models_dir / 'laplace_complete_v2' / 'SCD0000101' / 'SCD0000101_analysis.vtk'))

    tissue_type_cells = mesh.cell_data['TissueType'][0].flatten()
    cells = mesh.cells[0].data
    points = mesh.points

    n_points = len(points)

    # Track which tissue types each point belongs to
    point_in_infarct = np.zeros(n_points, dtype=bool)
    point_in_border = np.zeros(n_points, dtype=bool)
    point_in_healthy = np.zeros(n_points, dtype=bool)

    for i, cell in enumerate(cells):
        for pt_idx in cell:
            if tissue_type_cells[i] == 3:
                point_in_infarct[pt_idx] = True
            elif tissue_type_cells[i] == 2:
                point_in_border[pt_idx] = True
            else:
                point_in_healthy[pt_idx] = True

    # Assign display tissue type (priority: infarct > border > healthy)
    tissue_type = np.ones(n_points)  # Default healthy
    tissue_type[point_in_border] = 2
    tissue_type[point_in_infarct] = 3  # Infarct takes priority

    print(f"Tissue classification (by point membership):")
    print(f"  Healthy only: {np.sum(~point_in_infarct & ~point_in_border)}")
    print(f"  Border (any): {np.sum(point_in_border)}")
    print(f"  Infarct (any): {np.sum(point_in_infarct)}")

    return points, tissue_type, point_in_infarct, point_in_border


def create_complete_hydrogel(points, point_in_infarct, thickness=0.15):
    """
    Create hydrogel covering ALL points in infarct region.
    No gaps - complete continuous coverage.
    """
    hydrogel_indices = np.where(point_in_infarct)[0]
    hydrogel_points = points[hydrogel_indices].copy()

    # Offset outward for epicardial patch
    centroid = np.mean(hydrogel_points, axis=0)
    for i in range(len(hydrogel_points)):
        direction = hydrogel_points[i] - centroid
        norm = np.linalg.norm(direction)
        if norm > 0:
            hydrogel_points[i] += (direction / norm) * thickness

    print(f"\nHydrogel patch (complete coverage):")
    print(f"  Points: {len(hydrogel_points)} ({len(hydrogel_points)/len(points)*100:.2f}%)")
    print(f"  Thickness offset: {thickness} mm")

    return hydrogel_points, hydrogel_indices


def render_complete_coverage(points, tissue_type, hydrogel_points,
                              filename='patient001_complete_coverage.png'):
    """Render with complete hydrogel coverage over infarct."""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    tissue_cmap = plt.cm.jet

    # Plot mesh points
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=tissue_type, cmap=tissue_cmap, vmin=1, vmax=3,
                        s=1.5, alpha=0.5, edgecolors='none')

    # Hydrogel with complete coverage - slightly larger markers, higher alpha
    if len(hydrogel_points) > 0:
        ax.scatter(hydrogel_points[:, 0], hydrogel_points[:, 1], hydrogel_points[:, 2],
                  c='#00E5FF', s=8, alpha=0.85, edgecolors='none')

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, pad=0.08)
    cbar.set_label('Tissue Type', fontsize=10)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Healthy', 'Border', 'Infarct'])

    # Legend
    legend_elements = [
        Patch(facecolor='#00E5FF', label=f'GelMA Hydrogel ({len(hydrogel_points)} pts)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Patient SCD0000101 - Complete Hydrogel Coverage\n'
                 'Covers ALL infarct-touching points (no gaps)',
                 fontsize=12, fontweight='bold')

    ax.view_init(elev=25, azim=45)

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


def render_side_by_side(points, tissue_type, hydrogel_points,
                         filename='patient001_coverage_comparison.png'):
    """Compare: mesh without hydrogel vs with complete hydrogel coverage."""
    fig = plt.figure(figsize=(18, 8))

    tissue_cmap = plt.cm.jet

    # Left: Original mesh (matching VTK screenshot)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=tissue_type, cmap=tissue_cmap, vmin=1, vmax=3,
               s=2, alpha=0.7, edgecolors='none')
    ax1.set_title('Original LV Mesh\n(from VTK TissueType)', fontsize=11, fontweight='bold')
    ax1.view_init(elev=25, azim=45)

    # Right: With hydrogel
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=tissue_type, cmap=tissue_cmap, vmin=1, vmax=3,
               s=2, alpha=0.5, edgecolors='none')
    ax2.scatter(hydrogel_points[:, 0], hydrogel_points[:, 1], hydrogel_points[:, 2],
               c='#00E5FF', s=10, alpha=0.9, edgecolors='none')
    ax2.set_title(f'With GelMA Hydrogel\n({len(hydrogel_points)} points, complete coverage)',
                  fontsize=11, fontweight='bold')
    ax2.view_init(elev=25, azim=45)

    for ax in [ax1, ax2]:
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        max_range = np.array([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])]).max() / 2.0
        mid = np.mean(points, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    legend_elements = [
        Patch(facecolor=plt.cm.jet(0.0), label='Healthy'),
        Patch(facecolor=plt.cm.jet(0.5), label='Border Zone'),
        Patch(facecolor=plt.cm.jet(1.0), label='Infarct'),
        Patch(facecolor='#00E5FF', label='GelMA Hydrogel'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    plt.suptitle('Patient SCD0000101: Hydrogel Therapy Coverage',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def render_multi_angle_complete(points, tissue_type, hydrogel_points,
                                 filename='patient001_complete_angles.png'):
    """Multi-angle view with complete hydrogel coverage."""
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
                  s=1, alpha=0.4, edgecolors='none')

        ax.scatter(hydrogel_points[:, 0], hydrogel_points[:, 1], hydrogel_points[:, 2],
                  c='#00E5FF', s=6, alpha=0.9, edgecolors='none')

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
        Patch(facecolor='#00E5FF', label='GelMA Hydrogel'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    plt.suptitle(f'Patient SCD0000101: Complete Hydrogel Coverage ({len(hydrogel_points)} points)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def create_rotation_animation(points, tissue_type, hydrogel_points,
                               n_frames=36, filename='patient001_complete_rotation.gif'):
    """Smooth rotation with complete hydrogel coverage."""
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    tissue_cmap = plt.cm.jet
    max_range = np.array([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])]).max() / 2.0
    mid = np.mean(points, axis=0)

    def update(frame):
        ax.clear()

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=tissue_type, cmap=tissue_cmap, vmin=1, vmax=3,
                  s=1, alpha=0.4, edgecolors='none')

        ax.scatter(hydrogel_points[:, 0], hydrogel_points[:, 1], hydrogel_points[:, 2],
                  c='#00E5FF', s=8, alpha=0.9, edgecolors='none')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Patient SCD0000101: Complete Hydrogel Coverage',
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
    print("PATIENT SCD0000101 - COMPLETE HYDROGEL COVERAGE")

    print("\n1. Loading mesh with tissue classification...")
    points, tissue_type, point_in_infarct, point_in_border = load_mesh_and_classify()

    print("\n2. Creating complete hydrogel coverage...")
    hydrogel_points, hydrogel_indices = create_complete_hydrogel(points, point_in_infarct)

    print("\n3. Generating visualizations...")

    print("\n  3a. Complete coverage view...")
    render_complete_coverage(points, tissue_type, hydrogel_points,
                              filename='patient001_complete_coverage.png')

    print("\n  3b. Side-by-side comparison...")
    render_side_by_side(points, tissue_type, hydrogel_points,
                         filename='patient001_coverage_comparison.png')

    print("\n  3c. Multi-angle views...")
    render_multi_angle_complete(points, tissue_type, hydrogel_points,
                                 filename='patient001_complete_angles.png')

    print("\n  3d. Rotation animation...")
    create_rotation_animation(points, tissue_type, hydrogel_points,
                               n_frames=36, filename='patient001_complete_rotation.gif')

    print("COMPLETE")


if __name__ == "__main__":
    main()
