"""
Utility functions for plotting.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple, Union

from matplotlib.axes import SubplotBase
from matplotlib.cm import get_cmap as original_get_cmap
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.colors as mc
import colorsys


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_cmap(cmap: str = 'RdBu_r',
             bad_color: str = '#212121') -> Union[Colormap,
                                                  LinearSegmentedColormap,
                                                  ListedColormap]:
    """
    Convenience wrapper around matplotlib.cm.get_cmap() which allows to
    also set the `bad_color` (i.e., the color for NaN value)

    Args:
        cmap: The name of a matplotlib color map, for example 'RdBu_r'.
        bad_color: A string specifying a color in HTML format: (e.g.,
            '#FF0000') which will be used as the 'bad color' of the
            color map; that is, the color used, for example, for NaN
            values in plt.imshow plots.

    Returns:
        A matplotlib colormap with the desired `bad_color`.
    """

    # Get desired color map and set the desired bad_color
    cmap = original_get_cmap(cmap)
    cmap.set_bad(color=bad_color)

    return cmap


def add_colorbar_to_ax(img: AxesImage,
                       fig: Figure,
                       ax: SubplotBase,
                       where: str = 'right') -> None:
    """
    Add a "nice" colorbar to an imshow plot.

    Source: https://stackoverflow.com/a/18195921/4100721

    Args:
        img: The return of the respective imshow() command.
        fig: The figure that the plot is part of (e.g., `plt.gcf()`).
        ax: The that of the plot is contained in (e.g., `plt.gca()`).
        where: Where to place the colorbar (left, right, top or bottom).
    """

    if where in ('left', 'right'):
        orientation = 'vertical'
    elif where in ('top', 'bottom'):
        orientation = 'horizontal'
    else:
        raise ValueError(f'Illegal value for `where`: "{where}". Must be one '
                         'of ["left", "right", "top", "bottom"].')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(where, size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation=orientation)


def adjust_luminosity(color: Union[str, Tuple[float, float, float]],
                      amount: float = 1.4) -> Tuple[float, float, float]:
    """
    Adjusts the luminosity of the input `color` by the given `amount`.

    Original source:
        https://stackoverflow.com/a/49601444/4100721

    Args:
        color: The input color. Can either be a hex string (e.g.,
            "#FF0000"), matplotlib color string (e.g., "C1" or "green"),
            or an RGB tuple in float format (e.g., (1.0, 0.0, 0.0)).
        amount: The amount by how much the input color should be
            lightened. For amount > 1, the color gets brighter; for
            amount < 1, the color is darkened. By default, colors are
            lightened by 40%.

    Returns:
        An RGB tuple describing the luminosity-adjusted input color.
    """

    # In case `color` is a proper color name, we can try to resolve it into
    # an RGB tuple using the lookup table (of HEX strings) in mc.cnames.
    if isinstance(color, str) and (color in mc.cnames.keys()):
        rgb = mc.to_rgb(mc.cnames[color])

    # Otherwise we try to convert the color to RGB; this will raise a value
    # error for invalid color formats.
    else:
        rgb = mc.to_rgb(color)

    # Convert color from RBG to HLS representation
    hue, luminosity, saturation = colorsys.rgb_to_hls(*rgb)

    # Multiply `1 - luminosity` by given `amount` and convert back to RGB
    luminosity = max(0, min(1, amount * luminosity))
    rgb = colorsys.hls_to_rgb(hue, luminosity, saturation)

    return rgb
