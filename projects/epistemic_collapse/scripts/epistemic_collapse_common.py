"""Shared paths, artifact saving, and the channel grammar for the
epistemic_collapse project scripts.

Every epistemic_collapse_*.py script imports this first; it pins the working
directory to the repo root (so repo-relative data paths resolve) and exposes
save_artifact / save_text_artifact, which write the canonical artifact files
into projects/epistemic_collapse/artifacts/ (git history is the version
control).

Channel grammar (fixed across every figure in this paper):
  COLOR     = mitigation strategy (method), via hv_style.ROLES:
                no mitigation        -> baseline_gray
                sharing secondhand   -> comparator    (mimesis)
                books                -> focus
                mimesis + books      -> anchor        (reserved for the combination)
  LINESTYLE = information source (declared in the Fig. 1 caption):
                solid  = firsthand only (agents answer only from what they observed)
                dashed = uses secondhand information (peers' answers or books)
              (overrides the hv_style role linestyles, so the opening figures,
              which are all firsthand, use solid lines)
  FAILURE   = project exception to the canon (per HH, 2026-10-24): failure
              components use matplotlib Set1 so that they "add" visually --
              collapse red (#e41a1c) + misalignment blue (#377eb8) = all
              failures purple (#984ea3).
  A second swept factor (e.g. forgetting speed) gets marker fill or a
  figure-wide solid/dashed pair declared in that figure's legend.
"""
import os
import sys

# compute etiquette: cap BLAS threads (set before numpy is imported anywhere)
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

REPO =os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = 'projects/epistemic_collapse/data'
ART = 'projects/epistemic_collapse/artifacts'

import hv_style  # noqa: E402

# font sizes: the whole hv_style.SIZES dict scaled for this paper (never one label)
FONT_SCALE = 1.25
SIZES = {k: v * FONT_SCALE for k, v in hv_style.SIZES.items()}

# method -> hv_style role (color + linestyle + width)
METHODS = {
    'no mitigation': 'baseline_gray',
    'sharing secondhand': 'comparator',
    'books': 'focus',
    'mimesis + books': 'anchor',
}

# epistemic failure and its components (Set1; project exception, see docstring)
FAILURE = {
    'all failures': '#984ea3',   # Set1 purple
    'collapse': '#e41a1c',       # Set1 red
    'misalignment': '#377eb8',   # Set1 blue
}


# linestyle = information source: solid = firsthand only, dashed = secondhand
LINESTYLE = {
    'no mitigation': '-',
    'sharing secondhand': '--',
    'books': '--',
    'mimesis + books': '--',
}


def style(method):
    """matplotlib kwargs (color, ls, lw) for a method."""
    return {**hv_style.ROLES[METHODS[method]], 'ls': LINESTYLE[method]}


def save_artifact(fig, name, dpi=200, pad=0.03, facecolor=None):
    """PNG (raster preview) + vector PDF (for the paper)."""
    kw = dict(dpi=dpi, bbox_inches='tight', pad_inches=pad)
    if facecolor is not None:
        kw['facecolor'] = facecolor
    for ext in ('png', 'pdf'):
        fig.savefig(f'{ART}/{name}.{ext}', **kw)
    print(f'wrote {ART}/{name}.png/.pdf')


def save_text_artifact(name, ext, text):
    open(f'{ART}/{name}.{ext}', 'w').write(text)
    print(f'wrote {ART}/{name}.{ext}')
