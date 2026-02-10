#!/usr/bin/env python3
"""
Generate animated GIFs of cardiac mesh with hydrogel using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'figures' / '3d_visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.dpi': 100,
    'savefig.dpi': 150,
})


def create_lv_mesh(n_circumferential=40, n_longitudinal=25, n_transmural=5):
    """Create simplified LV mesh."""

    base_radius_endo = 25
    base_radius_epi = 35
    apex_radius_endo = 5
    apex_radius_epi = 12
    lv_length = 80

    theta = np.linspace(0, 2*np.pi, n_circumferential)
    phi = np.linspace(0, np.pi/2, n_longitudinal)

    vertices = []

    for t in range(n_transmural):
        trans_frac = t / (n_transmural - 1)

        for p_idx, p in enumerate(phi):
            for th_idx, th in enumerate(theta):
                r_endo = base_radius_endo * np.cos(p) + apex_radius_endo * np.sin(p)
                r_epi = base_radius_epi * np.cos(p) + apex_radius_epi * np.sin(p)
                r = r_endo + trans_frac * (r_epi - r_endo)
                z = lv_length * (1 - np.sin(p))
                x = r * np.cos(th)
                y = r * np.sin(th)
                vertices.append([x, y, z])

    return np.array(vertices), n_circumferential, n_longitudinal, n_transmural


def define_regions(vertices):
    """Define tissue regions."""
    regions = np.zeros(len(vertices))

    for i, (x, y, z) in enumerate(vertices):
        theta = np.arctan2(y, x)
        z_norm = z / 80

        infarct_theta_center = np.pi / 4
        infarct_z_center = 0.5

        theta_dist = np.abs(np.arctan2(np.sin(theta - infarct_theta_center),
                                        np.cos(theta - infarct_theta_center)))
        z_dist = np.abs(z_norm - infarct_z_center)

        combined_dist = np.sqrt((theta_dist / 0.5)**2 + (z_dist / 0.25)**2)

        if combined_dist < 0.6:
            regions[i] = 2  # Infarct
        elif combined_dist < 1.0:
            regions[i] = 1  # Border zone
        else:
            regions[i] = 0  # Healthy

    return regions


def get_stress(regions, time_phase, with_hydrogel=False):
    """Compute stress field."""
    stress = np.zeros(len(regions))

    for i, region in enumerate(regions):
        if region == 0:
            base = 15 + 5 * np.random.random()
        elif region == 1:
            base = 35 + 15 * np.random.random()
        else:
            base = 25 + 10 * np.random.random()

        systolic = 1 + 0.5 * np.sin(time_phase * 2 * np.pi)

        if with_hydrogel and region >= 1:
            reduction = 0.7
        else:
            reduction = 1.0

        stress[i] = base * systolic * reduction

    return stress


def create_cardiac_cycle_gif(vertices, regions, n_circ, n_long, n_trans,
                              with_hydrogel=False, filename="cardiac_cycle.gif"):
    """Create cardiac cycle animation."""

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    stress_cmap = plt.cm.RdYlBu_r
    stress_norm = Normalize(vmin=10, vmax=60)

    epi_start = (n_trans - 1) * n_circ * n_long
    epi_vertices = vertices[epi_start:]
    epi_regions = regions[epi_start:]

    n_frames = 24

    def update(frame):
        ax.clear()

        time_phase = frame / n_frames
        stress = get_stress(regions, time_phase, with_hydrogel)
        epi_stress = stress[epi_start:]

        contraction = 1 - 0.12 * np.sin(time_phase * np.pi)

        for i in range(n_long - 1):
            for j in range(n_circ - 1):
                idx = i * n_circ + j

                v0 = epi_vertices[idx].copy() * np.array([contraction, contraction, 1])
                v1 = epi_vertices[idx + 1].copy() * np.array([contraction, contraction, 1])
                v2 = epi_vertices[idx + n_circ].copy() * np.array([contraction, contraction, 1])
                v3 = epi_vertices[idx + n_circ + 1].copy() * np.array([contraction, contraction, 1])

                avg_stress = np.mean([epi_stress[idx], epi_stress[idx+1],
                                      epi_stress[idx+n_circ], epi_stress[idx+n_circ+1]])
                color = stress_cmap(stress_norm(avg_stress))

                ax.add_collection3d(Poly3DCollection([[v0, v1, v2]], facecolors=color,
                                                      edgecolors='none', alpha=0.95))
                ax.add_collection3d(Poly3DCollection([[v1, v3, v2]], facecolors=color,
                                                      edgecolors='none', alpha=0.95))

        phase_name = "Diastole" if time_phase < 0.3 or time_phase > 0.8 else "Systole"
        title = f"Cardiac Cycle - {phase_name}\n"
        title += "With Hydrogel Patch" if with_hydrogel else "Baseline (Untreated)"

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.view_init(elev=25, azim=45 + frame * 5)

        ax.set_xlim([-45, 45])
        ax.set_ylim([-45, 45])
        ax.set_zlim([0, 85])

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)

    save_path = OUTPUT_DIR / filename
    print(f"   Saving {filename}...")
    anim.save(str(save_path), writer=PillowWriter(fps=8))
    plt.close()
    print(f"   Saved: {save_path}")


def create_hydrogel_integration_gif(vertices, regions, n_circ, n_long, n_trans,
                                     filename="hydrogel_integration.gif"):
    """Create hydrogel integration over time animation."""

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    stress_cmap = plt.cm.RdYlBu_r
    stress_norm = Normalize(vmin=10, vmax=50)

    epi_start = (n_trans - 1) * n_circ * n_long
    epi_vertices = vertices[epi_start:]
    epi_regions = regions[epi_start:]

    n_frames = 36

    def update(frame):
        ax.clear()

        day = (frame / n_frames) * 90
        effect_factor = 1 / (1 + np.exp(-(day - 30) / 10))
        hydrogel_remaining = np.exp(-day / 60)
        stress_reduction = 0.3 * effect_factor

        stress = get_stress(regions, 0.5, with_hydrogel=False)

        for i in range(len(stress)):
            if regions[i] >= 1:
                stress[i] *= (1 - stress_reduction)

        epi_stress = stress[epi_start:]

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

                ax.add_collection3d(Poly3DCollection([[v0, v1, v2]], facecolors=color,
                                                      edgecolors='none', alpha=0.9))
                ax.add_collection3d(Poly3DCollection([[v1, v3, v2]], facecolors=color,
                                                      edgecolors='none', alpha=0.9))

        # Draw hydrogel patch (simplified)
        theta_gel = np.linspace(0, np.pi/2, 20)
        for th_idx in range(len(theta_gel) - 1):
            for z_frac in np.linspace(0.3, 0.7, 10):
                z = z_frac * 80
                r = 36 + 2 * hydrogel_remaining
                th = theta_gel[th_idx]

                x = r * np.cos(th)
                y = r * np.sin(th)

                ax.scatter([x], [y], [z], c='#26C6DA', s=20 * hydrogel_remaining + 5,
                          alpha=0.6 * hydrogel_remaining + 0.1)

        ef_improvement = 9.1 * effect_factor
        stress_red = 30.1 * effect_factor

        title = f"Hydrogel Integration: Day {int(day)}\n"
        title += f"ΔEF: +{ef_improvement:.1f}%  |  Stress Reduction: {stress_red:.1f}%  |  "
        title += f"Gel Remaining: {hydrogel_remaining*100:.0f}%"

        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        ax.set_zlabel('Z (mm)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.view_init(elev=30, azim=45)

        ax.set_xlim([-45, 45])
        ax.set_ylim([-45, 45])
        ax.set_zlim([0, 85])

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=150, blit=False)

    save_path = OUTPUT_DIR / filename
    print(f"   Saving {filename}...")
    anim.save(str(save_path), writer=PillowWriter(fps=6))
    plt.close()
    print(f"   Saved: {save_path}")


def create_rotating_view_gif(vertices, regions, n_circ, n_long, n_trans,
                              filename="cardiac_mesh_rotating.gif"):
    """Create rotating 3D view animation."""

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    region_colors = {0: '#E57373', 1: '#FFB74D', 2: '#9E9E9E'}

    epi_start = (n_trans - 1) * n_circ * n_long
    epi_vertices = vertices[epi_start:]
    epi_regions = regions[epi_start:]

    # Precompute triangles
    triangles = []
    colors = []

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

            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])
            colors.extend([color, color])

    n_frames = 36

    def update(frame):
        ax.clear()

        for tri, color in zip(triangles, colors):
            ax.add_collection3d(Poly3DCollection([tri], facecolors=color,
                                                  edgecolors='#666666', linewidths=0.1, alpha=0.9))

        # Add hydrogel patch visualization
        for th in np.linspace(0, np.pi/2, 15):
            for z in np.linspace(25, 55, 8):
                r = 38
                x = r * np.cos(th)
                y = r * np.sin(th)
                ax.scatter([x], [y], [z], c='#26C6DA', s=25, alpha=0.7)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Left Ventricle with GelMA Hydrogel Patch', fontsize=12, fontweight='bold')
        ax.view_init(elev=30, azim=frame * 10)

        ax.set_xlim([-45, 45])
        ax.set_ylim([-45, 45])
        ax.set_zlim([0, 85])

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)

    save_path = OUTPUT_DIR / filename
    print(f"   Saving {filename}...")
    anim.save(str(save_path), writer=PillowWriter(fps=10))
    plt.close()
    print(f"   Saved: {save_path}")


def main():
    print("GENERATING 3D CARDIAC ANIMATION GIFs")

    print("\n1. Creating mesh and regions...")
    vertices, n_circ, n_long, n_trans = create_lv_mesh(40, 25, 5)
    regions = define_regions(vertices)
    print(f"   Mesh: {len(vertices)} vertices")

    print("\n2. Generating animations...")

    print("\n   2a. Rotating view (36 frames)...")
    create_rotating_view_gif(vertices, regions, n_circ, n_long, n_trans,
                             "cardiac_mesh_rotating.gif")

    print("\n   2b. Cardiac cycle - baseline (24 frames)...")
    create_cardiac_cycle_gif(vertices, regions, n_circ, n_long, n_trans,
                             with_hydrogel=False, filename="cardiac_cycle_baseline.gif")

    print("\n   2c. Cardiac cycle - with hydrogel (24 frames)...")
    create_cardiac_cycle_gif(vertices, regions, n_circ, n_long, n_trans,
                             with_hydrogel=True, filename="cardiac_cycle_hydrogel.gif")

    print("\n   2d. Hydrogel integration over 90 days (36 frames)...")
    create_hydrogel_integration_gif(vertices, regions, n_circ, n_long, n_trans,
                                    "hydrogel_integration_90days.gif")

    print("GIF GENERATION COMPLETE")
    print(f"\nOutput directory: {OUTPUT_DIR}")

    for f in sorted(OUTPUT_DIR.glob("*.gif")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
