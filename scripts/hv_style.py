"""Helivan Blues house style (per HH style guide, 2026-09-09).

Ported verbatim from hhelm10/dkps (agentic-traces: scripts/hv_style.py);
only this docstring differs. One blue, many lightnesses; line style encodes
the paper's declared binary attribute.

COLOR always encodes the method, with one fixed mapping per paper, in every
figure. LINESTYLE follows the paper's declared binary attribute everywhere.
A second swept factor gets a separate channel with its own legend: marker
fill (filled / open) or, where markers are unavailable, a figure-wide
solid / dashed pair declared in that figure's legend. Orthogonal factors ->
separate legends. Each project declares its role assignment (method -> role,
the binary attribute, and the reserved violet concept) in
projects/<name>/scripts/<name>_common.py and its README.
"""
import matplotlib as mpl

INK_TITLE = '#0a2245'
INK = '#213c66'
INK_MUTE = '#486884'
SPINE = '#afbec6'
EDGE = '#d7dde6'
GRID = '#e5e7eb'
WASH = '#eef2f7'
REFLINE = '#94a3b8'

ROLES = {
    'baseline_gray': dict(color='#6b7280', ls='--', lw=2.6),
    'baseline_pale': dict(color='#afbec6', ls='--', lw=2.6),
    'comparator':    dict(color='#93aacc', ls='-', lw=2.6),
    'slate':         dict(color='#486884', ls='--', lw=2.6),
    'generic':       dict(color='#486884', ls='-', lw=2.6),
    'focus':         dict(color='#3596ff', ls='-', lw=2.6),
    'anchor':        dict(color='#114471', ls='-', lw=3.6),
    'reserve':       dict(color='#6d5bd0'),
}

SIZES = dict(suptitle=21.2, panel_letter=19.6, label=18.9, subtitle=16.7,
             legend=16.7, tick=15.8, annot=14.0)

# sequential single-hue ramp in the brand blue (light -> navy)
CMAP_BLUE = mpl.colors.LinearSegmentedColormap.from_list(
    'hv_blues', ['#e8f0fb', '#93aacc', '#3596ff', '#114471'])
# the reserve concept's ramp (tint -> violet); this paper reserves violet
# for the benchmark score y
CMAP_RESERVE = mpl.colors.LinearSegmentedColormap.from_list(
    'hv_reserve', ['#e6e0f8', '#a795e3', '#6d5bd0', '#43307e'])

# Helivan diverging map (per HH): blue = negative / below center,
# helivan gold = positive / above center; center on zero (TwoSlopeNorm)
CMAP_DIV = mpl.colors.LinearSegmentedColormap.from_list('helivan_div', [
    '#19395e', '#065199', '#0c70cf', '#3894fc', '#7cb6fd', '#b5d6fe',
    '#f3f1ef',
    '#fecf99', '#fca72a', '#d08713', '#a3690d', '#774b06', '#4f3005',
])
try:
    mpl.colormaps.register(CMAP_DIV)
    mpl.colormaps.register(CMAP_DIV.reversed())  # 'helivan_div_r'
except ValueError:
    pass  # already registered in this process


def apply():
    mpl.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': SPINE, 'axes.linewidth': 0.9,
        'axes.grid': True, 'axes.axisbelow': True,
        'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': INK_MUTE, 'ytick.color': INK_MUTE,
        'axes.labelcolor': INK, 'axes.titlecolor': INK_TITLE,
        'text.color': INK, 'font.size': 11,
        'legend.frameon': False,
    })
