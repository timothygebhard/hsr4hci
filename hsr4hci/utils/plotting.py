"""
Utility functions for plotting.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, List, Optional, Tuple, Union

import colorsys

from matplotlib.axes import SubplotBase
from matplotlib.cm import get_cmap as original_get_cmap
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils import CircularAperture

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# TYPE DEFINITIONS
# -----------------------------------------------------------------------------

# A valid matplotlib color can be one of the following three options:
# - A string specifying either the name of the color, or defining the color
#   as a HEX string. Examples: "red", "C3", "#FF0000".
# - An RGB tuple specifying the color. Example: (1, 0, 0) for red.
# - An RGBA tuple, specifying the color and the alpha channel (i.e., the
#   transparency). Example: (1, 0, 0, 0.5) for a semitransparent red.
MatplotlibColor = Union[str,
                        Tuple[float, float, float],
                        Tuple[float, float, float, float]]


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_cmap(cmap_name: str = 'RdBu_r',
             bad_color: str = '#212121') -> Union[Colormap,
                                                  LinearSegmentedColormap,
                                                  ListedColormap]:
    """
    Convenience wrapper around matplotlib.cm.get_cmap() which allows to
    also set the `bad_color` (i.e., the color for NaN value)

    Args:
        cmap_name: The name of a matplotlib color map, for example 'RdBu_r'.
        bad_color: A string specifying a color in HTML format: (e.g.,
            '#FF0000') which will be used as the 'bad color' of the
            color map; that is, the color used, for example, for NaN
            values in plt.imshow plots.

    Returns:
        A matplotlib colormap with the desired `bad_color`.
    """

    # Get desired color map and set the desired bad_color
    cmap = original_get_cmap(cmap_name)
    cmap.set_bad(color=bad_color)

    return cmap


def get_transparent_cmap(color: MatplotlibColor = 'red') -> ListedColormap:
    """
    Return a colormap that goes from transparent to the target color.

    Color maps of this type can be useful, for example, when plotting
    or overlaying masks, where only the selected pixels should receive
    a color, while everything else stays

    Args:
        color: A valid matplotlib color.

    Returns:
        A ListedColormap, which gradually goes from transparent to the
        given target color.´
    """

    return ListedColormap([(0, 0, 0, 0), color])


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


def adjust_luminosity(color: MatplotlibColor,
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
        rgb: Tuple[float, float, float] = mc.to_rgb(mc.cnames[color])

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


def disable_ticks(
    ax: Any,
) -> None:
    """
    Disable the ticks and labels on the given matplotlib `ax`. This is
    similar to calling `ax.axis('off')`, except that the frame around
    the plot is preserved.

    Args:
        ax: A matplotlib axis.
    """

    ax.tick_params(axis='both', which='both', top=False, bottom=False,
                   left=False, right=False, labelbottom=False, labelleft=False)


def plot_frame(
    frame: np.ndarray,
    file_path: Optional[str] = None,
    figsize: Tuple[float, float] = (4.0, 4.0),
    aperture_radius: Optional[float] = None,
    positions: Optional[List[Tuple[float, float]]] = None,
    snrs: Optional[List[float]] = None,
    draw_color: Optional[MatplotlibColor] = 'darkgreen',
    limit: Optional[float] = None,
) -> Figure:
    """
    Plot a single frame (e.g., a signal estimate). If desired, also add
    apertures and labels for the SNR at given positions.

    Args:
        frame: A 2D numpy array of shape `(width, height)` containing
            the frame to be plotted.
        file_path: A string containing the path at which to save the
            resulting plot. If None is given, the plot is not saved.
        figsize: A two-tuple `(x_size, y_size)` containing the size of
            the figure in inches.
        aperture_radius: If apertures are to be draw (see below), this
            is the radius (in units of pixels) that is used for them.
        positions: Optionally, a list of positions. At each position,
            an aperture is drawn with the given aperture radius.
        snrs: Optionally, each aperture can also be decorated with a
            label containing the SNR. This list contains the values for
            these SNR labels.
        draw_color: The color to be used for drawing the aperture and
            also the label.
        limit: Range limit to be used for the plot (vmin, vmax).

    Returns:
        A matplotlib figure containing the plot of the frame.
    """

    # -------------------------------------------------------------------------
    # General set up, draw frame to canvas
    # -------------------------------------------------------------------------

    # Set up a new figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)

    # If apertures are to be drawn, we can define them here and use the
    # photometry values from them to define the value limits of the plot
    if (aperture_radius is not None) and (positions is not None):

        # Define aperture to get plot limits, and also to for plotting later
        aperture = CircularAperture(positions=positions, r=aperture_radius)
        photometry, _ = aperture.do_photometry(data=frame)

        # Determine the limits for the color map: Use 120% of the average pixel
        # value in the aperture with the highest total flux
        if limit is None:
            limit = 1.2 * np.nanmax(np.abs(photometry)) / aperture.area

    # Otherwise, just compute the limit based on the entire frame
    else:
        if limit is None:
            limit = 1.2 * np.nanmax(np.abs(frame))
        aperture = None

    # Create the actual plot and use the limit we just computed
    plt.imshow(
        X=frame,
        origin='lower',
        cmap=get_cmap(),
        vmin=(-1 * limit),
        vmax=limit,
        interpolation='none',
    )

    # -------------------------------------------------------------------------
    # Plot apertures and labels
    # -------------------------------------------------------------------------

    # Plot the desired apertures
    if aperture is not None:
        aperture.plot(axes=ax, **dict(color=draw_color, lw=1, ls='-'))

    # Add labels for their respective SNR
    if (snrs is not None) and (positions is not None):

        # If SNRs and positions do not have the same length, the zip will fail,
        # so we do not have to add another check for this
        for snr, position in zip(snrs, positions):

            # Compute position of the label containing the SNR
            angle = np.arctan2(
                position[1] - frame.shape[1] / 2,
                position[0] - frame.shape[0] / 2
            )
            x = position[0] + max((8, 0.2 * position[0])) * np.cos(angle)
            y = position[1] + max((8, 0.2 * position[1])) * np.sin(angle)

            # Actually add the label with the SNR at this position
            ax.text(
                x=x,
                y=y,
                s=f'{snr:.1f}',
                ha='center',
                va='center',
                color='white',
                fontsize=12,
                bbox=dict(
                    facecolor=draw_color,
                    edgecolor='none',
                    boxstyle='round,pad=0.15',
                ),
            )

            # Draw connection between label and aperture
            ax.plot(
                [position[0] + aperture_radius * np.cos(angle), x],
                [position[1] + aperture_radius * np.sin(angle), y],
                color=draw_color,
                lw=1,
            )

    # -------------------------------------------------------------------------
    # Set plot options and save result
    # -------------------------------------------------------------------------

    # Remove ax ticks
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelleft=False,
        labelbottom=False,
    )

    # Save the results
    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight', dpi=300)

    return fig
