"""Read the closed paths of an SVG file as an `Outline`.

Each `<path>` becomes one loop per `M ... Z` subpath: an `L` command is a `Line`, a
`C` command a `CubicBezier` with its control points kept (nothing is flattened; the
outline samples the curve when it is meshed), and `Z` closes the loop with a `Line`
when the artwork's last point does not land on its first. Quadratic Beziers and arcs
are refused rather than dropped. Coordinates arrive in a y-up frame mirrored about the
document height, so the artwork plots the way it looks in a browser.

`Outline.from_svg` is the entry point; this module is what it runs.
"""
import re
import xml.etree.ElementTree as ET

import numpy as np
import svg.path  # pyright: ignore[reportMissingImports]

from fem.mesh.curves import CubicBezier, Line, Piece, _coincide
from fem.mesh.outline import Outline

__all__ = ['read_svg_outline']


def _document_height(root):
    '''Height of the SVG user-space box, or None if the file does not say.

    Needed to mirror the artwork: SVG's y axis points down the page, so a path
    read literally arrives upside down in any y-up frame.
    '''
    height = root.get('height')
    if height is not None:
        # Lengths may carry a unit ('737.6px'); the number is what matters here.
        number = re.match(r'\s*([0-9.eE+-]+)', height)
        if number:
            return float(number.group(1))

    view_box = root.get('viewBox')
    if view_box is not None:
        bounds = view_box.replace(',', ' ').split()
        if len(bounds) == 4:
            return float(bounds[1]) + float(bounds[3])
    return None


def _subpaths(d):
    '''The `M ... Z` subpaths of a path's `d`, each as its svg.path commands.'''
    current: list = []
    for command in svg.path.parse_path(d):
        if isinstance(command, svg.path.path.Move):
            if current:
                yield current
            current = [command]
        else:
            current.append(command)
    if current:
        yield current


def _loop_pieces(commands, mirror) -> tuple[Piece, ...]:
    '''One subpath's commands as pieces in the y-up frame, closed; or `()` if it has
    no closing `Z` or draws nothing.'''
    pieces: list[Piece] = []
    closed = False
    for command in commands:
        start, end = mirror(command.start), mirror(command.end)
        if isinstance(command, svg.path.path.Move):
            continue
        if isinstance(command, svg.path.path.Line):
            if not _coincide(start, end):
                pieces.append(Line(start, end))
        elif isinstance(command, svg.path.path.CubicBezier):
            pieces.append(CubicBezier(start, mirror(command.control1),
                                      mirror(command.control2), end))
        elif isinstance(command, svg.path.path.Close):
            # The closing edge, unless the artwork already landed back on its start.
            if pieces and not _coincide(pieces[-1].end, pieces[0].start):
                pieces.append(Line(pieces[-1].end, pieces[0].start))
            closed = True
            break
        else:
            raise NotImplementedError(
                f'SVG path command {type(command).__name__} is not supported; the outline '
                'reader takes M, L, C, and Z'
            )
    return tuple(pieces) if closed and len(pieces) >= 2 else ()


def read_svg_outline(svg_file) -> Outline:
    '''The closed paths of an SVG file as an `Outline`; see the module docstring.'''
    root = ET.parse(svg_file).getroot()
    paths = [d for path in root.findall('.//{http://www.w3.org/2000/svg}path')
             if (d := path.get('d')) is not None]

    # Fall back to the artwork's own extent when the file declares no size: the mirror
    # line only shifts the result, and shape is what callers use.
    height = _document_height(root)
    if height is None:
        ys = [command.end.imag for d in paths for sub in _subpaths(d) for command in sub]
        height = max(ys) if ys else 0.0

    def mirror(z):
        return np.array([float(z.real), height - float(z.imag)])

    loops = [pieces for d in paths for sub in _subpaths(d) if (pieces := _loop_pieces(sub, mirror))]
    if not loops:
        raise ValueError(f'{svg_file} has no closed path to read an outline from')
    return Outline(loops)
