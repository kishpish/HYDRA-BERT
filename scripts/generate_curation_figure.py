#!/usr/bin/env python3
"""
Generate polymer curation pipeline figure for research paper.
Shows filtering from 565 initial candidates to 24 final polymers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def create_curation_figure():
    """Create polymer curation pipeline figure."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.7, 'Polymer Library Curation: From 565 Candidates to 24 Cardiac-Specific Hydrogels',
            fontsize=13, fontweight='bold', ha='center')

    # Colors
    colors = {
        'initial': '#E3F2FD',
        'filter1': '#FFECB3',
        'filter2': '#C8E6C9',
        'filter3': '#F8BBD9',
        'filter4': '#B3E5FC',
        'final': '#A5D6A7',
        'arrow': '#424242',
        'exclude': '#FFCDD2',
    }

    # Stage boxes - vertical flow
    stages = [
        ('Initial Library', '565', 'Compiled from PolyInfo, literature,\npolyBERT training set, synthetic variants', colors['initial'], 8.5),
        ('Chemical Validity', '423', 'Valid SMILES, canonical forms,\nno duplicates (-142)', colors['filter1'], 7.0),
        ('Hydrogel Formation', '187', 'Gel-forming capability,\ncrosslinking mechanism (-236)', colors['filter2'], 5.5),
        ('Biocompatibility', '89', 'Cytocompatibility >70%,\nnon-toxic degradation (-98)', colors['filter3'], 4.0),
        ('Cardiac Applicability', '41', 'Injectable, 1-50 kPa stiffness,\ncardiac literature evidence (-48)', colors['filter4'], 2.5),
        ('Final Selection', '24', 'Representative diversity,\nclinical translation potential (-17)', colors['final'], 1.0),
    ]

    box_width = 4.5
    box_height = 0.9

    for i, (title, count, desc, color, y) in enumerate(stages):
        x = 4.75

        # Main box
        rect = FancyBboxPatch((x, y), box_width, box_height,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)

        # Count circle
        circle = plt.Circle((x + 0.5, y + box_height/2), 0.35,
                            facecolor='white', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x + 0.5, y + box_height/2, count, fontsize=11, fontweight='bold',
               ha='center', va='center')

        # Title
        ax.text(x + 1.1, y + box_height/2 + 0.15, title, fontsize=10, fontweight='bold',
               ha='left', va='center')

        # Description
        ax.text(x + 1.1, y + box_height/2 - 0.2, desc, fontsize=7,
               ha='left', va='center', color='#424242')

        # Arrow to next stage
        if i < len(stages) - 1:
            ax.annotate('', xy=(7, y - 0.1), xytext=(7, y + 0.05),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # Exclusion annotations on the right
    exclusions = [
        (7.7, 'Invalid SMILES,\nduplicates', '142'),
        (6.2, 'Non-gel-forming,\nno crosslinking', '236'),
        (4.7, 'Cytotoxic,\nlacking safety data', '98'),
        (3.2, 'Too rigid,\nnon-injectable', '48'),
        (1.7, 'Redundant,\nlow translation', '17'),
    ]

    for y, reason, count in exclusions:
        # Exclusion box
        rect = FancyBboxPatch((10, y), 2.8, 0.7,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=colors['exclude'], edgecolor='#C62828',
                               linewidth=1, linestyle='--')
        ax.add_patch(rect)

        ax.text(11.4, y + 0.35, f'−{count}: {reason}', fontsize=7,
               ha='center', va='center', color='#B71C1C')

        # Arrow from main flow to exclusion
        ax.annotate('', xy=(10, y + 0.35), xytext=(9.3, y + 0.35),
                   arrowprops=dict(arrowstyle='->', color='#C62828', lw=1,
                                  linestyle='--'))

    # Source databases on the left
    ax.text(1.5, 8.8, 'Source Databases:', fontsize=10, fontweight='bold', ha='center')

    sources = [
        ('PolyInfo', '312'),
        ('Cardiac Literature', '89'),
        ('polyBERT Set', '127'),
        ('Synthetic Variants', '37'),
    ]

    for i, (source, count) in enumerate(sources):
        y = 8.3 - i * 0.4
        rect = FancyBboxPatch((0.3, y), 2.4, 0.35,
                               boxstyle="round,pad=0.02,rounding_size=0.05",
                               facecolor='#E8EAF6', edgecolor='#3F51B5', linewidth=1)
        ax.add_patch(rect)
        ax.text(1.5, y + 0.175, f'{source} ({count})', fontsize=8,
               ha='center', va='center')

    # Arrow from sources to initial
    ax.annotate('', xy=(4.75, 8.9), xytext=(2.7, 8.3),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Final polymer categories on the left bottom
    ax.text(1.5, 2.8, 'Final 24 Polymers:', fontsize=10, fontweight='bold', ha='center')

    categories = [
        ('GelMA variants', '8', '#FF7043'),
        ('PEGDA', '3', '#42A5F5'),
        ('Polysaccharide', '5', '#66BB6A'),
        ('GAG (HA)', '3', '#FFCA28'),
        ('dECM', '2', '#AB47BC'),
        ('Protein', '2', '#26A69A'),
        ('Conductive', '1', '#78909C'),
    ]

    for i, (cat, count, color) in enumerate(categories):
        y = 2.4 - i * 0.35
        rect = FancyBboxPatch((0.3, y), 2.4, 0.3,
                               boxstyle="round,pad=0.02,rounding_size=0.05",
                               facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(1.5, y + 0.15, f'{cat} ({count})', fontsize=8,
               ha='center', va='center', fontweight='bold')

    # Arrow from final to categories
    ax.annotate('', xy=(2.7, 1.4), xytext=(4.75, 1.4),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Key statistics box at bottom
    stats_box = FancyBboxPatch((4, 0.1), 6, 0.6,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor='#F5F5F5', edgecolor='gray', linewidth=1)
    ax.add_patch(stats_box)

    ax.text(7, 0.4, '24 Polymers × 60 Patients × ~310 Formulations = 447,480 Training Samples',
           fontsize=10, fontweight='bold', ha='center', va='center')

    figures_dir = PROJECT_ROOT / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'figure_polymer_curation.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(figures_dir / 'figure_polymer_curation.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Polymer curation figure saved!")


def create_polymer_diversity_figure():
    """Create figure showing the 24 polymers and their properties."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Left: Category distribution pie chart
    ax1 = axes[0]
    categories = ['GelMA\nvariants', 'Polysaccharide', 'PEGDA', 'GAG (HA)',
                  'dECM', 'Protein', 'Conductive']
    counts = [8, 5, 3, 3, 2, 2, 1]
    colors = ['#FF7043', '#66BB6A', '#42A5F5', '#FFCA28', '#AB47BC', '#26A69A', '#78909C']

    wedges, texts, autotexts = ax1.pie(counts, labels=categories, autopct='%1.0f%%',
                                        colors=colors, explode=[0.05]*7,
                                        textprops={'fontsize': 9})
    plt.setp(autotexts, fontweight='bold', fontsize=9)
    ax1.set_title('A. Polymer Category Distribution\n(n=24 curated from 565 candidates)',
                  fontsize=11, fontweight='bold')

    # Right: Property coverage
    ax2 = axes[1]

    # Create a table-like visualization
    polymers = [
        'GelMA_3pct', 'GelMA_5pct', 'GelMA_7pct', 'GelMA_10pct',
        'GelMA_BioIL', 'GelMA_MXene', 'GelMA_rGO', 'GelMA_PPy',
        'PEGDA_575', 'PEGDA_700', 'PEGDA_3400',
        'Alginate_CaCl2', 'Alginate_RGD',
        'Chitosan_thermo', 'Chitosan_EGCG', 'Chitosan_HA',
        'HA_acellular', 'HA_ECM', 'MeHA',
        'dECM_VentriGel', 'dECM_cardiac',
        'Gelatin', 'Fibrin',
        'PEDOT:PSS'
    ]

    # Properties: Injectable, Conductive, Photo-XL, Clinical
    properties = np.array([
        [1, 0, 1, 1],  # GelMA_3pct
        [1, 0, 1, 1],  # GelMA_5pct
        [1, 0, 1, 1],  # GelMA_7pct
        [1, 0, 1, 1],  # GelMA_10pct
        [1, 1, 1, 0],  # GelMA_BioIL
        [1, 1, 1, 0],  # GelMA_MXene
        [1, 1, 1, 0],  # GelMA_rGO
        [1, 1, 1, 0],  # GelMA_PPy
        [1, 0, 1, 1],  # PEGDA_575
        [1, 0, 1, 1],  # PEGDA_700
        [1, 0, 1, 1],  # PEGDA_3400
        [1, 0, 0, 1],  # Alginate_CaCl2
        [1, 0, 0, 1],  # Alginate_RGD
        [1, 0, 0, 1],  # Chitosan_thermo
        [1, 0, 0, 0],  # Chitosan_EGCG
        [1, 0, 0, 0],  # Chitosan_HA
        [1, 0, 0, 1],  # HA_acellular
        [1, 0, 0, 1],  # HA_ECM
        [1, 0, 1, 0],  # MeHA
        [1, 0, 0, 1],  # dECM_VentriGel
        [1, 0, 0, 0],  # dECM_cardiac
        [1, 0, 0, 1],  # Gelatin
        [1, 0, 0, 1],  # Fibrin
        [0, 1, 0, 1],  # PEDOT:PSS
    ])

    cmap = plt.cm.RdYlGn
    im = ax2.imshow(properties, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(['Injectable', 'Conductive', 'Photo-XL', 'Clinical\nEvidence'],
                        fontsize=9, fontweight='bold')
    ax2.set_yticks(range(len(polymers)))
    ax2.set_yticklabels(polymers, fontsize=7)

    ax2.set_title('B. Property Coverage of Curated Library', fontsize=11, fontweight='bold')

    # Add grid
    ax2.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, 24, 1), minor=True)
    ax2.grid(which='minor', color='white', linestyle='-', linewidth=1)

    # Add checkmarks for positive values
    for i in range(len(polymers)):
        for j in range(4):
            if properties[i, j] == 1:
                ax2.text(j, i, '✓', ha='center', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    figures_dir = PROJECT_ROOT / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'figure_polymer_diversity.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(figures_dir / 'figure_polymer_diversity.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Polymer diversity figure saved!")


if __name__ == '__main__':
    create_curation_figure()
    create_polymer_diversity_figure()
    print("\nAll curation figures generated!")
    print(f"Files saved to {PROJECT_ROOT / 'figures'}")
