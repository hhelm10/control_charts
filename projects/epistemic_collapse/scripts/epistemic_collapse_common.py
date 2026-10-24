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
                dashed = firsthand only (agents answer only from what they observed)
                solid  = uses secondhand information (peers' answers or books)
  VIOLET    = the reserved concept: epistemic failure phi. Its components use the
              violet ramp: all failures (#6d5bd0), collapse (#43307e, dark),
              misalignment (#a795e3, light).
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

# method -> hv_style role (color + linestyle + width)
METHODS = {
    'no mitigation': 'baseline_gray',
    'sharing secondhand': 'comparator',
    'books': 'focus',
    'mimesis + books': 'anchor',
}

# the reserved concept (epistemic failure) and its components
FAILURE = {
    'all failures': '#6d5bd0',
    'collapse': '#43307e',
    'misalignment': '#a795e3',
}


def style(method):
    """matplotlib kwargs (color, ls, lw) for a method."""
    return dict(hv_style.ROLES[METHODS[method]])


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
