"""Render the generated simulator terrain and a close-up of its bridge."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter


PALETTE = np.array([
    [173, 177, 119], [143, 155, 101], [114, 135, 85], [108, 136, 103],
    [78, 136, 148], [192, 170, 128], [162, 116, 76], [100, 99, 75], [144, 143, 124],
], dtype=np.float32)


def terrain_image(folder):
    with np.load(folder / 'forest_map.npz') as native, np.load(folder / 'landscape.npz') as land:
        elevation = native['terrain'].astype(np.float32)
        resolution = float(native['resolution_m'])
        origin = native['origin_xy']
        cover = land['landcover']
        canopy = land['canopy_height_m']
    x0, y0 = origin
    height, width = np.array(cover.shape) * resolution
    extent = (x0, x0 + width, y0, y0 + height)
    terrain = PALETTE[cover].copy()
    rng = np.random.default_rng(312)
    fine = rng.normal(size=cover.shape).astype(np.float32)
    meadow_texture = gaussian_filter(fine, 4.) * 40 + fine * 1.3
    gy, gx = np.gradient(gaussian_filter(elevation, 2.), resolution)
    terrain *= np.clip(1. + .85 * (gy - gx), .73, 1.22)[..., None]
    terrain += meadow_texture[..., None]
    # Canopy heights come from the scene file. Shade their crowns for the
    # overhead view while retaining the physical surface classes along roads.
    tree_surface = (canopy > 0.) & np.isin(cover, (0, 1, 2, 3, 7))
    cy, cx = np.gradient(gaussian_filter(canopy.astype(np.float32), .9), resolution)
    crown = np.clip(.85 + .10 * (cy - cx), .54, 1.27)
    foliage = np.array([91, 123, 86]) * crown[..., None] + meadow_texture[..., None]
    opacity = (np.clip(canopy / 4., 0., .92) * tree_surface)[..., None]
    rgb = np.clip(terrain * (1. - opacity) + foliage * opacity, 0, 255).astype(np.uint8)
    return rgb, extent


def render(folder):
    rgb, extent = terrain_image(folder)
    x0, x1, y0, y1 = extent
    width, height = x1 - x0, y1 - y0
    case = json.loads((folder / 'case.json').read_text())
    audit = json.loads((folder / 'audit.json').read_text())
    plt.rcParams.update({'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'], 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(10.7, 6.8), facecolor='#faf9f4')
    layout = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.], left=.045, right=.975,
                              top=.86, bottom=.12, wspace=.12)
    overview = fig.add_subplot(layout[0, 0])
    detail = fig.add_subplot(layout[0, 1])
    for ax in (overview, detail):
        ax.imshow(rgb, origin='lower', extent=extent, interpolation='hanning')
        ax.set(xticks=[], yticks=[], aspect='equal')
        for spine in ax.spines.values():
            spine.set_color('#606755')
    overview.set(xlim=extent[:2], ylim=extent[2:])
    start, goal = np.array(case['start']), np.array(case['goal'])
    bridge = np.array(audit['bridge_center_xy'])
    inset = 102.
    detail.set(xlim=(bridge[0] - inset, bridge[0] + inset), ylim=(bridge[1] - inset, bridge[1] + inset))
    detail.set_title('木桥和两岸的林道', fontsize=13, loc='left', color='#26382d', pad=8)
    overview.add_patch(Rectangle(bridge - inset, 2 * inset, 2 * inset, fill=False,
                                 ec='#fcf6dc', lw=1., linestyle=(0, (4, 2))))

    for xy, name, marker, color in ((start, '起点', 'o', '#26394a'), (goal, '终点', '*', '#763e32')):
        overview.scatter(*xy, s=88, marker=marker, c=color, edgecolors='white', linewidths=1.)
        label = overview.annotate(name, xy, xytext=(0, 10), textcoords='offset points',
                                  ha='center', fontsize=10, color='white')
        label.set_path_effects([patheffects.withStroke(linewidth=2.4, foreground='#334238')])
    for ax, left, bottom, metres in ((overview, x0+45., y0+52., 200.),
                                     (detail, bridge[0]-inset+10, bridge[1]-inset+10, 40.)):
        ax.plot([left, left+metres], [bottom, bottom], color='#29372c', lw=4, solid_capstyle='butt')
        ax.plot([left, left+metres], [bottom, bottom], color='white', lw=2, solid_capstyle='butt')
        scale = ax.annotate(f'{metres:.0f} 米', (left+metres/2, bottom), xytext=(0, 4),
                            textcoords='offset points', ha='center', color='white', fontsize=8)
        scale.set_path_effects([patheffects.withStroke(linewidth=2., foreground='#334238')])
    fig.text(.046, .94, '溪谷林区', fontsize=23, color='#23372c', weight='bold')
    fig.text(.047, .898, f'合成野外导航场景 · {width:,.0f} × {height:,.0f} 米 · 上方为北',
             fontsize=11, color='#485a50')
    legend = [('草甸', PALETTE[0]), ('林地', [91, 123, 86]), ('水面', PALETTE[4]),
              ('林道', PALETTE[5]), ('木桥', PALETTE[6]), ('裸岩', PALETTE[8])]
    handles = [Patch(fc=np.array(color)/255., label=name) for name, color in legend]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.05), ncol=6,
               frameon=False, fontsize=10, handlelength=1.25, columnspacing=1.8)
    fig.savefig(folder / 'field_preview.png', dpi=170, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_family(folder, output):
    tasks = [('detour_left', '左侧过河'), ('direct', '直行过河'), ('detour_right', '右侧过河')]
    if json.loads((folder / 'direct/case.json').read_text())['kind'] == 'procedural_wooded_lake':
        return render_lake_family(folder, output)
    plt.rcParams.update({'font.family': ['Noto Sans CJK JP', 'DejaVu Sans']})
    fig, panels = plt.subplots(1, 3, figsize=(12.6, 5.5), facecolor='#faf9f4')
    fig.subplots_adjust(left=.025, right=.975, bottom=.16, top=.80, wspace=.055)
    route_color = '#ffdb77'
    for panel, (task, title) in zip(panels, tasks):
        variant = folder / task
        rgb, extent = terrain_image(variant)
        audit = json.loads((variant / 'audit.json').read_text())
        endpoints = np.asarray([audit['start'], audit['goal']])
        center = audit['bridge_center_xy']
        path = np.asarray(audit['reference_route_xy'])
        panel.imshow(rgb, origin='lower', extent=extent, interpolation='hanning')
        panel.plot(*endpoints.T, color='white', linestyle=(0, (2, 3)), lw=1.2)
        line, = panel.plot(*path.T, c=route_color, lw=1.6)
        line.set_path_effects([patheffects.withStroke(linewidth=2.8, foreground='#526045')])
        panel.scatter(*endpoints[0], s=48, marker='o', facecolors='#26394a', edgecolors='white', zorder=4)
        panel.scatter(*endpoints[1], s=112, marker='*', facecolors='#944e43', edgecolors='white', zorder=4)
        panel.scatter(*center, s=120, marker='s', facecolors='none', edgecolors='#fff4d5', linewidths=1.6)
        panel.set_title(title, size=13, color='#26392f', loc='left', pad=7)
        panel.set_xlabel(f"桥距直达线 {audit['detour_offset_m']:.0f} 米   ·   参考路线 {audit['reference_distance_m']:,.0f} 米",
                         fontsize=9, color='#38483b')
        panel.set(xticks=[], yticks=[])
        for spine in panel.spines.values():
            spine.set_color('#73846e')
    fig.text(.025, .925, '同一片溪谷，三种过河方向', size=20, color='#253c30', weight='bold')
    fig.text(.025, .865, '从起点面向终点区分左右。三张图起终点和林道相同，每张只有方框处的桥可通车。', size=11, color='#465a4d')
    markers = [Line2D([], [], marker=m, color='none', markerfacecolor=c, markeredgecolor=e, label=label)
               for m, c, e, label in [('o', '#26394a', '#26394a', '起点'), ('*', '#944e43', '#944e43', '终点'),
                                      ('s', 'none', '#626b52', '桥面')]]
    handles = markers + [Line2D([], [], color='#ae842d', label='离线几何参考路线')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.025), ncol=4, frameon=False, fontsize=10)
    fig.savefig(output, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_lake_family(folder, output):
    plt.rcParams.update({'font.family': ['Noto Sans CJK JP', 'DejaVu Sans']})
    fig, panels = plt.subplots(1, 3, figsize=(13, 5.8), facecolor='#faf9f4')
    fig.subplots_adjust(left=.025, right=.975, bottom=.16, top=.80, wspace=.07)
    colors = {'left': '#ffd36d', 'right': '#f3e9fa'}
    for ax, (task, title) in zip(panels, [('detour_left', '左岸较短'), ('detour_right', '右岸较短'),
                                          ('direct', '湖在路线旁')]):
        variant = folder / task
        rgb, extent = terrain_image(variant)
        a = json.loads((variant / 'audit.json').read_text())
        ax.imshow(rgb, origin='lower', extent=extent, interpolation='hanning')
        if task == 'direct':
            path = np.asarray(a['reference_route_xy'])
            ax.plot(*path.T, c='#fff9d9', lw=2)
            caption = f"直线路程 {a['direct_distance_m']:,.0f} 米"
        else:
            for side in ('left', 'right'):
                route = np.asarray(a[f'{side}_reference_route_xy'])
                line, = ax.plot(*route.T, c=colors[side], lw=1.8)
                line.set_path_effects([patheffects.withStroke(linewidth=3.2, foreground='#4c5848')])
            caption = f"左 {a['left_route_length_m']:,.0f} 米   ·   右 {a['right_route_length_m']:,.0f} 米"
        ax.scatter(*a['start'], s=49, c='#293e4f', edgecolors='white', linewidths=1, zorder=4)
        ax.scatter(*a['goal'], s=105, c='#93493d', marker='*', edgecolors='white', linewidths=.8, zorder=4)
        ax.set(xticks=[], yticks=[], xlabel=caption)
        ax.set_title(title, loc='left', fontsize=13, color='#26392f')
        for spine in ax.spines.values():
            spine.set_color('#73846e')
    fig.text(.025, .92, '公里级野外导航 · 两侧均可绕湖', size=20, color='#253c30', weight='bold')
    fig.text(.025, .862, '从起点面向终点区分左右。线条是离线核验的路线。', size=11, color='#465a4d')
    handles = [Line2D([], [], marker=m, color='none', markerfacecolor=c, markeredgecolor=c, label=t)
               for m, c, t in [('o', '#293e4f', '起点'), ('*', '#93493d', '终点')]]
    handles += [Line2D([], [], c=c, lw=2.2, label=t) for c, t in [('#ba8d30', '左岸路线'), ('#99889b', '右岸路线')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.03), ncol=4, frameon=False, fontsize=10)
    fig.savefig(output, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Standalone scene, or terrain family with --family-output')
    parser.add_argument('--family-output', type=Path, help='Render the three matched variants to this PNG')
    args = parser.parse_args()
    if args.family_output:
        render_family(args.folder, args.family_output)
    else:
        render(args.folder)
