"""Resolve IMDB variant data directories (SeHGNN defaults vs skip preprocess)."""

from __future__ import annotations

import os

_SEHGNN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_imdb_nc_dir(
    variant: int,
    *,
    imdb_skip_data: bool,
    imdb_data_root: str | None = None,
) -> str:
    """
    Absolute path to an IMDB variant directory for NC / 0-hop scripts.

    Standard: ``<SeHGNN>/data/IMDB_var{N}/``
    Skip (default): ``<SeHGNN>/data/preprocessed_IMDB_skip_v{N}/`` from
    ``preprocess_IMDB_skip.py`` in this repo.
    Skip (custom root): ``<imdb_data_root>/IMDB_preprocessed_skip_v{N}/`` (MAGNN
    layout) when ``imdb_data_root`` is set.
    """
    if imdb_skip_data:
        if imdb_data_root is None:
            path = os.path.join(
                _SEHGNN_ROOT, 'data', f'preprocessed_IMDB_skip_v{variant}',
            )
        else:
            path = os.path.join(
                os.path.abspath(imdb_data_root),
                f'IMDB_preprocessed_skip_v{variant}',
            )
    else:
        path = os.path.join(_SEHGNN_ROOT, 'data', f'IMDB_var{variant}')
    return os.path.abspath(path)


def resolve_imdb_graphonly_dir(
    variant: int,
    *,
    imdb_skip_data: bool,
    imdb_data_root: str | None = None,
) -> str:
    """
    Absolute path for graph-only NC script.

    Standard: ``<SeHGNN>/data/IMDB_var{N}_graphonly/``
    Skip: same bundle as ``resolve_imdb_nc_dir`` (not the ``*_graphonly`` folders).
    """
    if imdb_skip_data:
        return resolve_imdb_nc_dir(
            variant, imdb_skip_data=True, imdb_data_root=imdb_data_root,
        )
    return os.path.abspath(os.path.join(_SEHGNN_ROOT, 'data', f'IMDB_var{variant}_graphonly'))


def add_imdb_skip_args(parser) -> None:
    """Register ``--imdb-skip-data`` and ``--imdb-data-root`` on an ArgumentParser."""
    parser.add_argument(
        '--imdb-skip-data',
        action='store_true',
        default=False,
        help=(
            'Use MAGNN-aligned skip bundles: fixed semantic movie channels (M, MAM, MDM, MLM). '
            'Default data dir: data/preprocessed_IMDB_skip_vN under SeHGNN (generate with '
            'python preprocess_IMDB_skip.py --variant vN or --all). '
            'With --imdb-data-root DIR, use DIR/IMDB_preprocessed_skip_vN (legacy MAGNN output).'
        ),
    )
    parser.add_argument(
        '--imdb-data-root',
        type=str,
        default=None,
        help=(
            'Parent of IMDB_preprocessed_skip_v* when set (MAGNN layout). '
            'If omitted with --imdb-skip-data, use SeHGNN data/preprocessed_IMDB_skip_vN.'
        ),
    )
