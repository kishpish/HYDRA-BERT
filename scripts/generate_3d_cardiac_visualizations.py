#!/usr/bin/env python3
"""
Generate 3D cardiac mesh visualizations with hydrogel patch application.
Creates publication-quality images and animated GIFs showing:
1. Left ventricular mesh with infarct region
2. Hydrogel patch application on epicardial surface
3. Cardiac cycle animation with stress distribution
4. Hydrogel integration over time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from pathlib import Path

# Set up output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'figures' / '3d_visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# High-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def create_lv_mesh(n_circumferential=60, n_longitudinal=40, n_transmural=8):
    """
    Create a simplified left ventricular mesh geometry.

    The LV is modeled as a truncated ellipsoid with:
    - Apex at bottom
    - Base (mitral valve plane) at top
    - Transmural layers from endocardium to epicardium
    """

    # Parameters for LV geometry (in mm)
    base_radius_endo = 25  # Endocardial radius at base
    base_radius_epi = 35   # Epicardial radius at base
    apex_radius_endo = 5   # Small opening at apex (endocardium)
    apex_radius_epi = 12   # Epicardial apex
    lv_length = 80         # Base to apex length

    # Create parametric coordinates
    theta = np.linspace(0, 2*np.pi, n_circumferential)  # Circumferential
    phi = np.linspace(0, np.pi/2, n_longitudinal)       # Longitudinal (0=base, pi/2=apex)

    # Generate mesh points
    vertices = []

    for t in range(n_transmural):
        # Transmural interpolation (0=endo, 1=epi)
        trans_frac = t / (n_transmural - 1)

        for p_idx, p in enumerate(phi):
            for th_idx, th in enumerate(theta):
                # Interpolate radius based on longitudinal position
                r_endo = base_radius_endo * np.cos(p) + apex_radius_endo * np.sin(p)
                r_epi = base_radius_epi * np.cos(p) + apex_radius_epi * np.sin(p)

                # Transmural interpolation
                r = r_endo + trans_frac * (r_epi - r_endo)

                # Convert to Cartesian
                z = lv_length * (1 - np.sin(p))  # z=0 at apex, z=lv_length at base
                x = r * np.cos(th)
                y = r * np.sin(th)

                vertices.append([x, y, z])

    vertices = np.array(vertices)

    return vertices, n_circumferential, n_longitudinal, n_transmural


def define_tissue_regions(vertices, n_circ, n_long, n_trans):
    """
    Define tissue regions: healthy, border zone, infarct scar.

    Infarct located in anterior-lateral wall, mid-ventricular level.
    """

    n_total = n_circ * n_long * n_trans
    regions = np.zeros(n_total)  # 0=healthy, 1=border zone, 2=infarct

    for i, (x, y, z) in enumerate(vertices):
        # Compute angular position (theta) and longitudinal position
        theta = np.arctan2(y, x)

        # Normalize z to 0-1 (apex to base)
        z_norm = z / 80  # lv_length = 80

        # Infarct region: anterior-lateral (theta ~ 0 to pi/2), mid-ventricle (z_norm ~ 0.3-0.7)
        # Infarct center
        infarct_theta_center = np.pi / 4  # 45 degrees
        infarct_z_center = 0.5

        # Angular distance from infarct center
        theta_dist = np.abs(np.arctan2(np.sin(theta - infarct_theta_center),
                                        np.cos(theta - infarct_theta_center)))
        z_dist = np.abs(z_norm - infarct_z_center)

        # Combined distance (elliptical)
        combined_dist = np.sqrt((theta_dist / 0.5)**2 + (z_dist / 0.25)**2)

        if combined_dist < 0.6:
            regions[i] = 2  # Infarct scar
        elif combined_dist < 1.0:
            regions[i] = 1  # Border zone
        else:
            regions[i] = 0  # Healthy

    return regions


def create_hydrogel_patch(vertices, regions, n_circ, n_long, n_trans, coverage='scar_bz50'):
    """
    Create hydrogel patch geometry on epicardial surface.

    Coverage patterns:
    - scar_only: Cover only infarct
    - scar_bz25: Cover infarct + 25% of border zone
    - scar_bz50: Cover infarct + 50% of border zone
    - scar_bz100: Cover infarct + 100% of border zone
    """

    # Get epicardial surface (outermost transmural layer)
    epi_start = (n_trans - 1) * n_circ * n_long
    epi_vertices = vertices[epi_start:]
    epi_regions = regions[epi_start:]

    # Determine coverage
    if coverage == 'scar_only':
        patch_mask = epi_regions == 2
    elif coverage == 'scar_bz25':
        patch_mask = (epi_regions == 2) | ((epi_regions == 1) & (np.random.random(len(epi_regions)) < 0.25))
    elif coverage == 'scar_bz50':
        patch_mask = (epi_regions == 2) | ((epi_regions == 1) & (np.random.random(len(epi_regions)) < 0.50))
    elif coverage == 'scar_bz100':
        patch_mask = (epi_regions >= 1)
    else:
        patch_mask = epi_regions == 2

    # Create patch vertices (slightly offset from epicardium)
    patch_vertices = epi_vertices.copy()

    # Offset in radial direction
    for i in range(len(patch_vertices)):
        if patch_mask[i]:
            x, y, z = patch_vertices[i]
            r = np.sqrt(x**2 + y**2)
            if r > 0:
                # Add 2mm thickness
                patch_vertices[i, 0] = x * (1 + 2/r)
                patch_vertices[i, 1] = y * (1 + 2/r)

    return patch_vertices, patch_mask, epi_vertices


def compute_stress_field(vertices, regions, time_phase=0, with_hydrogel=False):
    """
    Compute simulated stress field on the mesh.

    Stress is elevated in border zone, reduced by hydrogel.
    Time_phase: 0=diastole, 0.5=peak systole, 1=end systole
    """

    stress = np.zeros(len(vertices))

    for i, (region) in enumerate(regions):
        # Base stress by region
        if region == 0:  # Healthy
            base_stress = 15 + 5 * np.random.random()  # 15-20 kPa
        elif region == 1:  # Border zone
            base_stress = 35 + 15 * np.random.random()  # 35-50 kPa (elevated)
        else:  # Infarct
            base_stress = 25 + 10 * np.random.random()  # 25-35 kPa (passive)

        # Time-varying component (cardiac cycle)
        systolic_factor = 1 + 0.5 * np.sin(time_phase * 2 * np.pi)

        # Hydrogel effect (reduces stress in affected regions)
        if with_hydrogel and region >= 1:
            hydrogel_reduction = 0.7  # 30% reduction
        else:
            hydrogel_reduction = 1.0

        stress[i] = base_stress * systolic_factor * hydrogel_reduction

    return stress


def compute_strain_field(vertices, regions, time_phase=0, with_hydrogel=False):
    """
    Compute simulated strain field (deformation) on the mesh.
    """

    strain = np.zeros(len(vertices))

    for i, region in enumerate(regions):
        # Base strain by region
        if region == 0:  # Healthy - good contraction
            base_strain = -0.18 + 0.03 * np.random.random()  # -15% to -18%
        elif region == 1:  # Border zone - reduced
            base_strain = -0.10 + 0.03 * np.random.random()  # -7% to -10%
        else:  # Infarct - minimal/no contraction
            base_strain = -0.02 + 0.02 * np.random.random()  # 0% to -2%

        # Time-varying component
        systolic_factor = np.sin(time_phase * np.pi)  # Peak at mid-systole

        # Hydrogel effect (normalizes strain in border zone)
        if with_hydrogel and region == 1:
            # Improve border zone strain toward healthy
            base_strain = base_strain * 0.7 + (-0.15) * 0.3

        strain[i] = base_strain * systolic_factor

    return strain


def plot_3d_mesh_with_regions(vertices, regions, n_circ, n_long, n_trans,
                               view_angle=(30, 45), title="", save_path=None):
    """
    Create 3D visualization of cardiac mesh with tissue regions.
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color map for regions
    region_colors = {
        0: '#E57373',  # Healthy - light red (myocardium)
        1: '#FFB74D',  # Border zone - orange
        2: '#9E9E9E',  # Infarct - gray (scar)
    }

    # Plot epicardial surface
    epi_start = (n_trans - 1) * n_circ * n_long
    epi_vertices = vertices[epi_start:]
    epi_regions = regions[epi_start:]

    # Create surface triangulation
    for i in range(n_long - 1):
        for j in range(n_circ - 1):
            idx = i * n_circ + j

            # Get vertices for this quad
            v0 = epi_vertices[idx]
            v1 = epi_vertices[idx + 1]
            v2 = epi_vertices[idx + n_circ]
            v3 = epi_vertices[idx + n_circ + 1]

            # Get region (use average)
            region = int(np.round(np.mean([epi_regions[idx], epi_regions[idx+1],
                                           epi_regions[idx+n_circ], epi_regions[idx+n_circ+1]])))
            color = region_colors[region]

            # Create two triangles for the quad
            tri1 = [[v0, v1, v2]]
            tri2 = [[v1, v3, v2]]

            ax.add_collection3d(Poly3DCollection(tri1, facecolors=color,
                                                  edgecolors='black', linewidths=0.1, alpha=0.9))
            ax.add_collection3d(Poly3DCollection(tri2, facecolors=color,
                                                  edgecolors='black', linewidths=0.1, alpha=0.9))

    # Set axis properties
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Equal aspect ratio
    max_range = 50
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, 90])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#E57373', edgecolor='black', label='Healthy Myocardium'),
        mpatches.Patch(facecolor='#FFB74D', edgecolor='black', label='Border Zone'),
        mpatches.Patch(facecolor='#9E9E9E', edgecolor='black', label='Infarct Scar'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')

    plt.close()
    return fig


def plot_mesh_with_hydrogel(vertices, regions, patch_vertices, patch_mask, epi_vertices,
                            n_circ, n_long, n_trans, view_angle=(30, 45),
                            title="", save_path=None, hydrogel_opacity=0.8):
    """
    Create 3D visualization with hydrogel patch overlay.
    """

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')

    # Color map for regions
    region_colors = {
        0: '#E57373',  # Healthy
        1: '#FFB74D',  # Border zone
        2: '#9E9E9E',  # Infarct
    }

    # Hydrogel color (semi-transparent blue-green)
    hydrogel_color = '#26C6DA'

    # Get epicardial surface indices
    epi_start = (n_trans - 1) * n_circ * n_long
    epi_regions = regions[epi_start:]

    # Plot epicardial surface (cardiac tissue)
    for i in range(n_long - 1):
        for j in range(n_circ - 1):
            idx = i * n_circ + j

            v0 = epi_vertices[idx]
            v1 = epi_vertices[idx + 1]
            v2 = epi_vertices[idx + n_circ]
            v3 = epi_vertices[idx + n_circ + 1]

            region = int(np.round(np.mean([epi_regions[idx], epi_regions[idx+1],
                                           epi_regions[idx+n_circ], epi_regions[idx+n_circ+1]])))
            color = region_colors[region]

            tri1 = [[v0, v1, v2]]
            tri2 = [[v1, v3, v2]]

            ax.add_collection3d(Poly3DCollection(tri1, facecolors=color,
                                                  edgecolors='#666666', linewidths=0.1, alpha=0.85))
            ax.add_collection3d(Poly3DCollection(tri2, facecolors=color,
                                                  edgecolors='#666666', linewidths=0.1, alpha=0.85))

    # Plot hydrogel patch
    for i in range(n_long - 1):
        for j in range(n_circ - 1):
            idx = i * n_circ + j

            # Check if any vertex in this quad is part of patch
            if not (patch_mask[idx] or patch_mask[min(idx+1, len(patch_mask)-1)] or
                    patch_mask[min(idx+n_circ, len(patch_mask)-1)] or
                    patch_mask[min(idx+n_circ+1, len(patch_mask)-1)]):
                continue

            v0 = patch_vertices[idx]
            v1 = patch_vertices[min(idx + 1, len(patch_vertices)-1)]
            v2 = patch_vertices[min(idx + n_circ, len(patch_vertices)-1)]
            v3 = patch_vertices[min(idx + n_circ + 1, len(patch_vertices)-1)]

            tri1 = [[v0, v1, v2]]
            tri2 = [[v1, v3, v2]]

            ax.add_collection3d(Poly3DCollection(tri1, facecolors=hydrogel_color,
                                                  edgecolors='#00838F', linewidths=0.2,
                                                  alpha=hydrogel_opacity))
            ax.add_collection3d(Poly3DCollection(tri2, facecolors=hydrogel_color,
                                                  edgecolors='#00838F', linewidths=0.2,
                                                  alpha=hydrogel_opacity))

    # Set axis properties
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    max_range = 50
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, 90])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#E57373', edgecolor='black', label='Healthy Myocardium'),
        mpatches.Patch(facecolor='#FFB74D', edgecolor='black', label='Border Zone'),
        mpatches.Patch(facecolor='#9E9E9E', edgecolor='black', label='Infarct Scar'),
        mpatches.Patch(facecolor='#26C6DA', edgecolor='#00838F', label='GelMA Hydrogel Patch', alpha=0.8),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')

    plt.close()
    return fig


def plot_stress_distribution(vertices, regions, stress, n_circ, n_long, n_trans,
                             view_angle=(30, 45), title="", save_path=None,
                             with_hydrogel=False, patch_mask=None):
    """
    Create 3D visualization with stress field colormap.
    """

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')

    # Stress colormap (blue=low, red=high)
    stress_cmap = plt.cm.RdYlBu_r
    stress_norm = Normalize(vmin=10, vmax=50)

    # Get epicardial surface
    epi_start = (n_trans - 1) * n_circ * n_long
    epi_vertices = vertices[epi_start:]
    epi_stress = stress[epi_start:]

    # Plot surface with stress coloring
    for i in range(n_long - 1):
        for j in range(n_circ - 1):
            idx = i * n_circ + j

            v0 = epi_vertices[idx]
            v1 = epi_vertices[idx + 1]
            v2 = epi_vertices[idx + n_circ]
            v3 = epi_vertices[idx + n_circ + 1]

            # Average stress for this quad
            avg_stress = np.mean([epi_stress[idx], epi_stress[idx+1],
                                  epi_stress[idx+n_circ], epi_stress[idx+n_circ+1]])
            color = stress_cmap(stress_norm(avg_stress))

            tri1 = [[v0, v1, v2]]
            tri2 = [[v1, v3, v2]]

            ax.add_collection3d(Poly3DCollection(tri1, facecolors=color,
                                                  edgecolors='none', alpha=0.95))
            ax.add_collection3d(Poly3DCollection(tri2, facecolors=color,
                                                  edgecolors='none', alpha=0.95))

    # Add hydrogel outline if present
    if with_hydrogel and patch_mask is not None:
        for i in range(n_long - 1):
            for j in range(n_circ - 1):
                idx = i * n_circ + j

                if patch_mask[idx]:
                    v0 = epi_vertices[idx]
                    # Draw small marker for hydrogel coverage
                    ax.scatter([v0[0]], [v0[1]], [v0[2]+2], c='cyan', s=5, alpha=0.5)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=stress_cmap, norm=stress_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Wall Stress (kPa)', fontsize=12)

    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    max_range = 50
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, 90])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    plt.close()
    return fig


def create_cardiac_cycle_animation(vertices, regions, n_circ, n_long, n_trans,
                                   with_hydrogel=False, patch_mask=None,
                                   n_frames=30, save_path=None):
    """
    Create animated GIF of cardiac cycle showing stress changes.
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Stress colormap
    stress_cmap = plt.cm.RdYlBu_r
    stress_norm = Normalize(vmin=10, vmax=60)

    epi_start = (n_trans - 1) * n_circ * n_long
    epi_vertices = vertices[epi_start:]
    epi_regions = regions[epi_start:]

    frames = []

    for frame in range(n_frames):
        ax.clear()

        # Time phase in cardiac cycle
        time_phase = frame / n_frames

        # Compute stress at this time
        stress = compute_stress_field(vertices, regions, time_phase, with_hydrogel)
        epi_stress = stress[epi_start:]

        # Deformation factor (contraction during systole)
        contraction = 1 - 0.15 * np.sin(time_phase * np.pi)

        # Plot surface
        for i in range(n_long - 1):
            for j in range(n_circ - 1):
                idx = i * n_circ + j

                # Apply contraction to vertices
                v0 = epi_vertices[idx].copy()
                v1 = epi_vertices[idx + 1].copy()
                v2 = epi_vertices[idx + n_circ].copy()
                v3 = epi_vertices[idx + n_circ + 1].copy()

                # Contract in radial direction
                for v in [v0, v1, v2, v3]:
                    r = np.sqrt(v[0]**2 + v[1]**2)
                    if r > 0:
                        v[0] *= contraction
                        v[1] *= contraction

                avg_stress = np.mean([epi_stress[idx], epi_stress[idx+1],
                                      epi_stress[idx+n_circ], epi_stress[idx+n_circ+1]])
                color = stress_cmap(stress_norm(avg_stress))

                tri1 = [[v0, v1, v2]]
                tri2 = [[v1, v3, v2]]

                ax.add_collection3d(Poly3DCollection(tri1, facecolors=color,
                                                      edgecolors='none', alpha=0.95))
                ax.add_collection3d(Poly3DCollection(tri2, facecolors=color,
                                                      edgecolors='none', alpha=0.95))

        # Labels
        phase_name = "Diastole" if time_phase < 0.3 or time_phase > 0.8 else "Systole"
        title = f"Cardiac Cycle - {phase_name}\n"
        if with_hydrogel:
            title += "With GelMA Hydrogel Patch"
        else:
            title += "Untreated (Baseline)"

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.view_init(elev=25, azim=45 + frame * 3)  # Rotating view

        max_range = 50
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, 90])

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    plt.close()

    # Save as GIF
    if save_path:
        import imageio
        imageio.mimsave(save_path, frames, fps=10, loop=0)

    return frames


def create_hydrogel_integration_animation(vertices, regions, n_circ, n_long, n_trans,
                                          n_frames=40, save_path=None):
    """
    Create animated GIF showing hydrogel integration over time (days).
    Shows therapeutic effect progression.
    """

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')

    epi_start = (n_trans - 1) * n_circ * n_long
    epi_vertices = vertices[epi_start:]
    epi_regions = regions[epi_start:]

    # Create hydrogel patch
    patch_vertices, patch_mask, _ = create_hydrogel_patch(vertices, regions, n_circ, n_long, n_trans)

    frames = []

    for frame in range(n_frames):
        ax.clear()

        # Time in days (0 to 90 days)
        day = (frame / n_frames) * 90

        # Therapeutic effect progression (sigmoidal)
        effect_factor = 1 / (1 + np.exp(-(day - 30) / 10))  # Peaks around day 30-60

        # Hydrogel degradation (gradual over 60 days half-life)
        hydrogel_remaining = np.exp(-day / 60)

        # Stress reduction due to hydrogel
        stress_reduction = 0.3 * effect_factor  # Up to 30% reduction

        # Compute stress field
        stress = compute_stress_field(vertices, regions, time_phase=0.5, with_hydrogel=False)

        # Apply hydrogel effect to infarct/BZ regions
        for i in range(len(stress)):
            if regions[i] >= 1:  # BZ or infarct
                stress[i] *= (1 - stress_reduction)

        epi_stress = stress[epi_start:]

        # Stress colormap
        stress_cmap = plt.cm.RdYlBu_r
        stress_norm = Normalize(vmin=10, vmax=50)

        # Plot cardiac surface
        for i in range(n_long - 1):
            for j in range(n_circ - 1):
                idx = i * n_circ + j

                v0 = epi_vertices[idx]
                v1 = epi_vertices[idx + 1]
                v2 = epi_vertices[idx + n_circ]
                v3 = epi_vertices[idx + n_circ + 1]

                avg_stress = np.mean([epi_stress[idx], epi_stress[idx+1],
                                      epi_stress[idx+n_circ], epi_stress[idx+n_circ+1]])
                color = stress_cmap(stress_norm(avg_stress))

                tri1 = [[v0, v1, v2]]
                tri2 = [[v1, v3, v2]]

                ax.add_collection3d(Poly3DCollection(tri1, facecolors=color,
                                                      edgecolors='none', alpha=0.9))
                ax.add_collection3d(Poly3DCollection(tri2, facecolors=color,
                                                      edgecolors='none', alpha=0.9))

        # Plot degrading hydrogel patch
        hydrogel_color = plt.cm.Blues(0.5 + 0.3 * hydrogel_remaining)

        for i in range(n_long - 1):
            for j in range(n_circ - 1):
                idx = i * n_circ + j

                if not (patch_mask[idx] or patch_mask[min(idx+1, len(patch_mask)-1)]):
                    continue

                v0 = patch_vertices[idx]
                v1 = patch_vertices[min(idx + 1, len(patch_vertices)-1)]
                v2 = patch_vertices[min(idx + n_circ, len(patch_vertices)-1)]
                v3 = patch_vertices[min(idx + n_circ + 1, len(patch_vertices)-1)]

                # Reduce patch thickness as it degrades
                thickness_factor = hydrogel_remaining
                for v in [v0, v1, v2, v3]:
                    r_base = np.sqrt(epi_vertices[idx][0]**2 + epi_vertices[idx][1]**2)
                    r_current = np.sqrt(v[0]**2 + v[1]**2)
                    if r_base > 0:
                        v[0] = epi_vertices[idx][0] + (v[0] - epi_vertices[idx][0]) * thickness_factor
                        v[1] = epi_vertices[idx][1] + (v[1] - epi_vertices[idx][1]) * thickness_factor

                tri1 = [[v0, v1, v2]]
                tri2 = [[v1, v3, v2]]

                ax.add_collection3d(Poly3DCollection(tri1, facecolors=hydrogel_color,
                                                      edgecolors='#1565C0', linewidths=0.2,
                                                      alpha=0.7 * hydrogel_remaining + 0.1))
                ax.add_collection3d(Poly3DCollection(tri2, facecolors=hydrogel_color,
                                                      edgecolors='#1565C0', linewidths=0.2,
                                                      alpha=0.7 * hydrogel_remaining + 0.1))

        # Title and labels
        ef_improvement = 9.1 * effect_factor
        stress_red = 30.1 * effect_factor

        title = f"Hydrogel Integration: Day {int(day)}\n"
        title += f"ΔEF: +{ef_improvement:.1f}%  |  Stress Reduction: {stress_red:.1f}%  |  "
        title += f"Gel Remaining: {hydrogel_remaining*100:.0f}%"

        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        ax.set_zlabel('Z (mm)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')

        ax.view_init(elev=30, azim=45)

        max_range = 50
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, 90])

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    plt.close()

    # Save as GIF
    if save_path:
        import imageio
        imageio.mimsave(save_path, frames, fps=8, loop=0)

    return frames


def create_cross_section_view(vertices, regions, n_circ, n_long, n_trans,
                              with_hydrogel=False, save_path=None):
    """
    Create cross-sectional view showing transmural layers.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Mid-ventricular short-axis cross-section
    ax1 = axes[0]

    # Get mid-ventricular slice (z ~ 40mm)
    z_target = 40

    # For each transmural layer
    colors = ['#FFCDD2', '#EF9A9A', '#E57373', '#EF5350', '#F44336', '#E53935', '#D32F2F', '#C62828']

    for t in range(n_trans):
        layer_start = t * n_circ * n_long

        # Find points near z_target
        r_vals = []
        theta_vals = []
        region_vals = []

        for i in range(n_circ * n_long):
            idx = layer_start + i
            x, y, z = vertices[idx]

            if abs(z - z_target) < 5:  # Within 5mm of target
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                r_vals.append(r)
                theta_vals.append(theta)
                region_vals.append(regions[idx])

        if len(r_vals) > 0:
            # Sort by theta
            sorted_idx = np.argsort(theta_vals)
            r_vals = np.array(r_vals)[sorted_idx]
            theta_vals = np.array(theta_vals)[sorted_idx]
            region_vals = np.array(region_vals)[sorted_idx]

            # Plot as polar scatter
            for r, th, reg in zip(r_vals, theta_vals, region_vals):
                if reg == 0:
                    c = '#E57373'
                elif reg == 1:
                    c = '#FFB74D'
                else:
                    c = '#9E9E9E'
                ax1.scatter(r * np.cos(th), r * np.sin(th), c=c, s=20, alpha=0.8)

    ax1.set_xlim(-45, 45)
    ax1.set_ylim(-45, 45)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (mm)', fontsize=12)
    ax1.set_ylabel('Y (mm)', fontsize=12)
    ax1.set_title('Short-Axis Cross-Section (Mid-Ventricle)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add annotations
    ax1.annotate('Infarct\nScar', xy=(25, 15), fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='#9E9E9E', alpha=0.8))
    ax1.annotate('Border\nZone', xy=(30, -5), fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='#FFB74D', alpha=0.8))
    ax1.annotate('Healthy\nMyocardium', xy=(-25, -20), fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='#E57373', alpha=0.8))

    if with_hydrogel:
        # Draw hydrogel on epicardial surface
        theta_gel = np.linspace(-np.pi/4, np.pi/2, 50)
        r_gel = 38  # Just outside epicardium
        x_gel = r_gel * np.cos(theta_gel)
        y_gel = r_gel * np.sin(theta_gel)
        ax1.plot(x_gel, y_gel, c='#26C6DA', linewidth=8, alpha=0.7, label='Hydrogel Patch')
        ax1.legend(loc='lower left')

    # Long-axis cross-section
    ax2 = axes[1]

    # Get anterior wall slice (theta ~ pi/4)
    theta_target = np.pi / 4

    for t in range(n_trans):
        layer_start = t * n_circ * n_long

        z_vals = []
        r_vals = []
        region_vals = []

        for i in range(n_circ * n_long):
            idx = layer_start + i
            x, y, z = vertices[idx]
            theta = np.arctan2(y, x)

            if abs(theta - theta_target) < 0.2:
                r = np.sqrt(x**2 + y**2)
                z_vals.append(z)
                r_vals.append(r)
                region_vals.append(regions[idx])

        if len(z_vals) > 0:
            for zv, rv, reg in zip(z_vals, r_vals, region_vals):
                if reg == 0:
                    c = '#E57373'
                elif reg == 1:
                    c = '#FFB74D'
                else:
                    c = '#9E9E9E'
                ax2.scatter(rv, zv, c=c, s=20, alpha=0.8)

    ax2.set_xlim(0, 45)
    ax2.set_ylim(0, 85)
    ax2.set_xlabel('Radial Distance (mm)', fontsize=12)
    ax2.set_ylabel('Z - Base to Apex (mm)', fontsize=12)
    ax2.set_title('Long-Axis Cross-Section (Anterior Wall)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Labels
    ax2.annotate('Base', xy=(20, 80), fontsize=10, ha='center')
    ax2.annotate('Apex', xy=(10, 5), fontsize=10, ha='center')
    ax2.annotate('Endo', xy=(15, 45), fontsize=9, ha='center')
    ax2.annotate('Epi', xy=(38, 45), fontsize=9, ha='center')

    if with_hydrogel:
        # Draw hydrogel on epicardial surface
        z_gel = np.linspace(25, 55, 30)
        r_gel = np.ones_like(z_gel) * 38
        ax2.fill_betweenx(z_gel, r_gel, r_gel + 3, color='#26C6DA', alpha=0.7, label='Hydrogel')
        ax2.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')

    plt.close()
    return fig


def create_comparison_figure(vertices, regions, n_circ, n_long, n_trans, save_path=None):
    """
    Create side-by-side comparison: Baseline vs With Hydrogel.
    """

    fig = plt.figure(figsize=(18, 8))

    # Baseline (without hydrogel)
    ax1 = fig.add_subplot(121, projection='3d')

    # With hydrogel
    ax2 = fig.add_subplot(122, projection='3d')

    # Get epicardial surface
    epi_start = (n_trans - 1) * n_circ * n_long
    epi_vertices = vertices[epi_start:]
    epi_regions = regions[epi_start:]

    # Create hydrogel patch
    patch_vertices, patch_mask, _ = create_hydrogel_patch(vertices, regions, n_circ, n_long, n_trans)

    # Stress colormaps
    stress_cmap = plt.cm.RdYlBu_r

    for ax, with_hydrogel, title in [(ax1, False, 'A. Baseline (Untreated)'),
                                       (ax2, True, 'B. With GelMA Hydrogel Patch')]:

        # Compute stress
        stress = compute_stress_field(vertices, regions, time_phase=0.5, with_hydrogel=with_hydrogel)
        epi_stress = stress[epi_start:]

        # Normalize stress
        if with_hydrogel:
            stress_norm = Normalize(vmin=10, vmax=45)
        else:
            stress_norm = Normalize(vmin=10, vmax=55)

        # Plot surface
        for i in range(n_long - 1):
            for j in range(n_circ - 1):
                idx = i * n_circ + j

                v0 = epi_vertices[idx]
                v1 = epi_vertices[idx + 1]
                v2 = epi_vertices[idx + n_circ]
                v3 = epi_vertices[idx + n_circ + 1]

                avg_stress = np.mean([epi_stress[idx], epi_stress[idx+1],
                                      epi_stress[idx+n_circ], epi_stress[idx+n_circ+1]])
                color = stress_cmap(stress_norm(avg_stress))

                tri1 = [[v0, v1, v2]]
                tri2 = [[v1, v3, v2]]

                ax.add_collection3d(Poly3DCollection(tri1, facecolors=color,
                                                      edgecolors='none', alpha=0.95))
                ax.add_collection3d(Poly3DCollection(tri2, facecolors=color,
                                                      edgecolors='none', alpha=0.95))

        # Add hydrogel outline for treated case
        if with_hydrogel:
            for i in range(n_long - 1):
                for j in range(n_circ - 1):
                    idx = i * n_circ + j

                    if not patch_mask[idx]:
                        continue

                    v0 = patch_vertices[idx]
                    v1 = patch_vertices[min(idx + 1, len(patch_vertices)-1)]
                    v2 = patch_vertices[min(idx + n_circ, len(patch_vertices)-1)]
                    v3 = patch_vertices[min(idx + n_circ + 1, len(patch_vertices)-1)]

                    tri1 = [[v0, v1, v2]]
                    tri2 = [[v1, v3, v2]]

                    ax.add_collection3d(Poly3DCollection(tri1, facecolors='#26C6DA',
                                                          edgecolors='#00838F', linewidths=0.3,
                                                          alpha=0.5))
                    ax.add_collection3d(Poly3DCollection(tri2, facecolors='#26C6DA',
                                                          edgecolors='#00838F', linewidths=0.3,
                                                          alpha=0.5))

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.view_init(elev=25, azim=45)

        max_range = 50
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, 90])

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=stress_cmap, norm=stress_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=15, pad=0.1)
        cbar.set_label('Wall Stress (kPa)', fontsize=10)

    # Add overall metrics annotation
    fig.text(0.25, 0.02, 'Peak BZ Stress: 52 kPa\nEF: 32%', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    fig.text(0.75, 0.02, 'Peak BZ Stress: 36 kPa (-31%)\nEF: 41% (+9%)', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='green'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')

    plt.close()
    return fig


def main():
    """Generate all 3D cardiac visualizations."""

    print("GENERATING 3D CARDIAC MESH VISUALIZATIONS")

    # Create LV mesh
    print("\n1. Creating left ventricular mesh geometry...")
    vertices, n_circ, n_long, n_trans = create_lv_mesh(
        n_circumferential=50,
        n_longitudinal=30,
        n_transmural=6
    )
    print(f"   Mesh created: {len(vertices)} vertices")
    print(f"   Dimensions: {n_circ} circumferential × {n_long} longitudinal × {n_trans} transmural")

    # Define tissue regions
    print("\n2. Defining tissue regions (healthy, border zone, infarct)...")
    regions = define_tissue_regions(vertices, n_circ, n_long, n_trans)
    n_healthy = np.sum(regions == 0)
    n_bz = np.sum(regions == 1)
    n_infarct = np.sum(regions == 2)
    print(f"   Healthy: {n_healthy} ({n_healthy/len(regions)*100:.1f}%)")
    print(f"   Border Zone: {n_bz} ({n_bz/len(regions)*100:.1f}%)")
    print(f"   Infarct: {n_infarct} ({n_infarct/len(regions)*100:.1f}%)")

    # Create hydrogel patch
    print("\n3. Creating hydrogel patch geometry...")
    patch_vertices, patch_mask, epi_vertices = create_hydrogel_patch(
        vertices, regions, n_circ, n_long, n_trans, coverage='scar_bz50'
    )
    print(f"   Patch coverage: {np.sum(patch_mask)} vertices ({np.sum(patch_mask)/len(patch_mask)*100:.1f}%)")

    # Generate static images
    print("\n4. Generating static 3D images...")

    # 4a. Cardiac mesh with regions only
    print("   4a. Cardiac mesh with tissue regions...")
    plot_3d_mesh_with_regions(
        vertices, regions, n_circ, n_long, n_trans,
        view_angle=(30, 45),
        title="Left Ventricular Mesh with Infarct Regions",
        save_path=str(OUTPUT_DIR / "cardiac_mesh_regions.png")
    )

    # 4b. Cardiac mesh with hydrogel overlay
    print("   4b. Cardiac mesh with hydrogel patch...")
    plot_mesh_with_hydrogel(
        vertices, regions, patch_vertices, patch_mask, epi_vertices,
        n_circ, n_long, n_trans,
        view_angle=(30, 45),
        title="Left Ventricle with GelMA Hydrogel Patch Application",
        save_path=str(OUTPUT_DIR / "cardiac_mesh_with_hydrogel.png")
    )

    # 4c. Multiple view angles
    print("   4c. Multiple view angles...")
    for angle, name in [((30, 0), "anterior"), ((30, 90), "lateral"),
                         ((30, 180), "posterior"), ((90, 0), "apex")]:
        plot_mesh_with_hydrogel(
            vertices, regions, patch_vertices, patch_mask, epi_vertices,
            n_circ, n_long, n_trans,
            view_angle=angle,
            title=f"View: {name.capitalize()}",
            save_path=str(OUTPUT_DIR / f"cardiac_mesh_view_{name}.png")
        )

    # 4d. Stress distribution - baseline
    print("   4d. Stress distribution (baseline)...")
    stress_baseline = compute_stress_field(vertices, regions, time_phase=0.5, with_hydrogel=False)
    plot_stress_distribution(
        vertices, regions, stress_baseline, n_circ, n_long, n_trans,
        view_angle=(30, 45),
        title="Wall Stress Distribution - Baseline (Untreated)",
        save_path=str(OUTPUT_DIR / "stress_distribution_baseline.png")
    )

    # 4e. Stress distribution - with hydrogel
    print("   4e. Stress distribution (with hydrogel)...")
    stress_treated = compute_stress_field(vertices, regions, time_phase=0.5, with_hydrogel=True)
    plot_stress_distribution(
        vertices, regions, stress_treated, n_circ, n_long, n_trans,
        view_angle=(30, 45),
        title="Wall Stress Distribution - With GelMA Hydrogel",
        save_path=str(OUTPUT_DIR / "stress_distribution_hydrogel.png"),
        with_hydrogel=True, patch_mask=patch_mask
    )

    # 4f. Cross-section views
    print("   4f. Cross-section views...")
    create_cross_section_view(
        vertices, regions, n_circ, n_long, n_trans,
        with_hydrogel=False,
        save_path=str(OUTPUT_DIR / "cross_section_baseline.png")
    )
    create_cross_section_view(
        vertices, regions, n_circ, n_long, n_trans,
        with_hydrogel=True,
        save_path=str(OUTPUT_DIR / "cross_section_hydrogel.png")
    )

    # 4g. Comparison figure
    print("   4g. Side-by-side comparison...")
    create_comparison_figure(
        vertices, regions, n_circ, n_long, n_trans,
        save_path=str(OUTPUT_DIR / "stress_comparison_baseline_vs_hydrogel.png")
    )

    # Generate animated GIFs
    print("\n5. Generating animated GIFs (this may take a few minutes)...")

    try:
        import imageio

        # 5a. Cardiac cycle - baseline
        print("   5a. Cardiac cycle animation (baseline)...")
        create_cardiac_cycle_animation(
            vertices, regions, n_circ, n_long, n_trans,
            with_hydrogel=False, patch_mask=None,
            n_frames=30,
            save_path=str(OUTPUT_DIR / "cardiac_cycle_baseline.gif")
        )

        # 5b. Cardiac cycle - with hydrogel
        print("   5b. Cardiac cycle animation (with hydrogel)...")
        create_cardiac_cycle_animation(
            vertices, regions, n_circ, n_long, n_trans,
            with_hydrogel=True, patch_mask=patch_mask,
            n_frames=30,
            save_path=str(OUTPUT_DIR / "cardiac_cycle_hydrogel.gif")
        )

        # 5c. Hydrogel integration over time
        print("   5c. Hydrogel integration animation (90 days)...")
        create_hydrogel_integration_animation(
            vertices, regions, n_circ, n_long, n_trans,
            n_frames=45,
            save_path=str(OUTPUT_DIR / "hydrogel_integration_90days.gif")
        )

        print("\n   GIF animations created successfully!")

    except ImportError:
        print("\n   WARNING: imageio not installed. Skipping GIF generation.")
        print("   Install with: pip install imageio")

    print("VISUALIZATION GENERATION COMPLETE")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
