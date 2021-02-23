"""
Utility functions for plotting.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import colorsys

from astropy.units import Quantity
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap as original_get_cmap
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import bottleneck as bn
import matplotlib.colors as mc
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.utils.photometry import CustomCircularAperture


# -----------------------------------------------------------------------------
# TYPE DEFINITIONS
# -----------------------------------------------------------------------------

# A valid matplotlib color can be one of the following three options:
# - A string specifying either the name of the color, or defining the color
#   as a HEX string. Examples: "red", "C3", "#FF0000".
# - An RGB tuple specifying the color. Example: (1, 0, 0) for red.
# - An RGBA tuple, specifying the color and the alpha channel (i.e., the
#   transparency). Example: (1, 0, 0, 0.5) for a semitransparent red.
MatplotlibColor = Union[
    str, Tuple[float, float, float], Tuple[float, float, float, float]
]


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_cmap(
    cmap_name: str = 'RdBu_r', bad_color: str = '#212121'
) -> Union[Colormap, LinearSegmentedColormap, ListedColormap]:
    """
    Convenience wrapper around matplotlib.cm.get_cmap() which allows to
    also set the `bad_color` (i.e., the color for NaN value)

    Args:
        cmap_name: The name of a matplotlib color map (e.g., 'RdBu_r').
        bad_color: A string specifying a color in HTML format: (e.g.,
            '#FF0000') which will be used as the 'bad color' of the
            color map; that is, the color used, for example, for NaN
            values in plt.imshow plots.

    Returns:
        A matplotlib colormap with the desired `bad_color`.
    """

    # Get desired color map and set the desired bad_color
    cmap = copy(original_get_cmap(cmap_name))
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


def add_colorbar_to_ax(
    img: AxesImage, fig: Figure, ax: Axes, where: str = 'right'
) -> Colorbar:
    """
    Add a "nice" colorbar to an imshow plot.

    Source: https://stackoverflow.com/a/18195921/4100721

    Args:
        img: The return of the respective imshow() command.
        fig: The figure that the plot is part of (e.g., `plt.gcf()`).
        ax: The ax which contains the plot (e.g., `plt.gca()`).
        where: Where to place the colorbar (left, right, top or bottom).

    Returns:
        The colorbar that was added to the axis.
    """

    if where in ('left', 'right'):
        orientation = 'vertical'
    elif where in ('top', 'bottom'):
        orientation = 'horizontal'
    else:
        raise ValueError(
            f'Illegal value for `where`: "{where}". Must be one '
            'of ["left", "right", "top", "bottom"].'
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(where, size='5%', pad=0.05)
    cbar = fig.colorbar(img, cax=cax, orientation=orientation)

    return cbar


def adjust_luminosity(
    color: MatplotlibColor, amount: float = 1.4
) -> Tuple[float, float, float]:
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

    ax.tick_params(
        axis='both',
        which='both',
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )


def plot_frame(
    frame: np.ndarray,
    file_path: Optional[Union[Path, str]] = None,
    figsize: Tuple[float, float] = (4.0, 4.0),
    aperture_radius: Optional[float] = None,
    expand_radius: float = 2.5,
    positions: Optional[List[Tuple[float, float]]] = None,
    snrs: Optional[List[float]] = None,
    draw_color: Optional[MatplotlibColor] = 'darkgreen',
    limit: Optional[float] = None,
    label_options: Optional[Dict[str, Any]] = None,
    use_ticks: bool = True,
    use_colorbar: bool = False,
    use_logscale: bool = True,
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
        expand_radius: Factor by which to multiply the `aperture_radius`
            for drawing purposes.
        positions: Optionally, a list of positions. At each position,
            an aperture is drawn with the given aperture radius.
        snrs: Optionally, each aperture can also be decorated with a
            label containing the SNR. This list contains the values for
            these SNR labels.
        draw_color: The color to be used for drawing the aperture and
            also the label.
        limit: Range limit to be used for the plot (vmin, vmax).
        label_options: Additional keyword arguments that are passed to
            the `plt.text()` command that is used for the SNR labels.
        use_ticks: Whether or not to place ticks around the borders of
            the frame (to better understand the scale of the frame).
        use_colorbar: Whether or not to place a (tiny) colorbar on the
            frame (in the bottom right-hand corner).
        use_logscale: Whether or not to use a (symmetric) log scale for
            the color bar.

    Returns:
        A matplotlib figure containing the plot of the frame.
    """

    # -------------------------------------------------------------------------
    # General set up, draw frame to canvas
    # -------------------------------------------------------------------------

    # Set up a new figure
    fig, ax = plt.subplots(figsize=figsize)

    # Initialize aperture
    aperture: Optional[CustomCircularAperture] = None

    # Define the radius for drawing
    if aperture_radius is not None:
        draw_radius: Optional[float] = aperture_radius * expand_radius
    else:
        draw_radius = None

    # If apertures are to be drawn, we can define them here and use the
    # photometry values from them to define the value limits of the plot
    if (aperture_radius is not None) and (positions is not None):

        # Define aperture, because we need it for plotting later
        aperture = CustomCircularAperture(positions=positions, r=draw_radius)

        # If no explicit plot limits are given, we fit the aperture(s) to
        # determine the limit from the data
        if limit is None:

            # Fit each aperture with a 2D Gaussian; the results should have
            # the form `(amplitude, sigma)`.
            fit_results = aperture.fit_2d_gaussian(data=np.nan_to_num(frame))

            # If we have multiple apertures, we need to still take the maximum
            # over them; otherwise, we can directly use the value from the fit
            if isinstance(fit_results, list):
                limit = float(
                    np.around(1.1 * max(_[0] for _ in fit_results), 1)
                )
            else:
                limit = float(np.around(1.1 * fit_results[0], 1))

    # If the limit is still None at this point (i.e., if no apertures were
    # given, and there are also no explicit plot limits), just compute the
    # limit based on the entire frame
    if limit is None:
        limit = np.around(1.1 * bn.nanmax(np.abs(frame)), 1)

    # Prepare norm for the
    if use_logscale:
        norm = mc.SymLogNorm(
            linthresh=0.1 * limit,
            vmin=-1 * limit,
            vmax=limit,
            base=10,
        )
    else:
        norm = mc.PowerNorm(
            gamma=1,
            vmin=-1 * limit,
            vmax=limit,
        )

    # Prepare grid for the pcolormesh()
    x_range = np.arange(frame.shape[0])
    y_range = np.arange(frame.shape[1])
    x, y = np.meshgrid(x_range, y_range)

    # Create the actual plot and use the limit we just computed.
    # Using pcolormesh() instead of imshow() avoids interpolation artifacts in
    # most PDF viewers (otherwise, the PDF version will often look blurry).
    img = ax.pcolormesh(
        x,
        y,
        frame,
        shading='nearest',
        cmap=get_cmap(),
        snap=True,
        rasterized=True,
        norm=norm,
    )

    # -------------------------------------------------------------------------
    # Plot apertures and labels
    # -------------------------------------------------------------------------

    # Plot the desired apertures
    if aperture is not None:
        aperture.plot(axes=ax, **dict(color=draw_color, lw=2, ls='-'))

    # Define default options for the SNR label
    label_kwargs = dict(
        ha='left',
        va='center',
        color='white',
        fontsize=18,
        bbox=dict(
            facecolor=draw_color,
            edgecolor='none',
            boxstyle='square,pad=0.075',
        ),
    )

    # Add or overwrite options that were passed using the `label_kwargs`
    if label_options is not None:
        for key, value in label_kwargs.items():
            label_options[key] = value

    # Add labels for their respective SNR
    if (
        (snrs is not None)
        and (positions is not None)
        and (draw_radius is not None)
    ):
        for snr, position in zip(snrs, positions):

            ax.annotate(
                text=f'{snr:.1f}',
                xy=(position[0] + draw_radius, position[1]),
                xytext=(8, 0),
                textcoords='offset pixels',
                arrowprops=dict(
                    arrowstyle='-',
                    shrinkA=0,
                    shrinkB=0,
                    lw=2,
                    color=draw_color,
                ),
                **label_kwargs,
            )

    # -------------------------------------------------------------------------
    # Add stellar position and scale bar
    # -------------------------------------------------------------------------

    # Place a "+"-marker at the center for the frame
    center = (frame.shape[0] / 2, frame.shape[1] / 2)
    ax.plot(center[0], center[1], '+', ms=10, color='black')

    # Compute size of the scale bar, and define its label accordingly
    scalebar_size = Quantity(1.0, 'arcsec').to('pixel').value
    scalebar_label_value = 1.0
    while scalebar_size > 0.3 * frame.shape[0]:
        scalebar_size /= 2
        scalebar_label_value /= 2
    scalebar_label = f'{scalebar_label_value}"'

    # Create the scale bar and add it to the frame (loc=1 means "upper right")
    scalebar = AnchoredSizeBar(
        transform=ax.transData,
        size=scalebar_size,
        label=scalebar_label,
        loc=1,
        pad=1,
        color='white',
        frameon=False,
        size_vertical=0,
        fontproperties=fm.FontProperties(size=12),
    )
    ax.add_artist(scalebar)

    # -------------------------------------------------------------------------
    # Add color bar
    # -------------------------------------------------------------------------

    if use_colorbar:

        # Create new ax object for colorbar
        cax = inset_axes(
            parent_axes=ax,
            width="18%",
            height="2%",
            loc='lower right',
            borderpad=2,
        )

        # Set up the rest of the colorbar options
        cbar = fig.colorbar(img, cax=cax, orientation='horizontal')
        cbar.set_ticks([-limit, 0, limit])
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')
        cbar.outline.set_edgecolor('white')
        cbar.ax.tick_params(labelsize=8, pad=1, length=3, color='white')
        cbar.ax.set_xticklabels(["{:.1f}".format(i) for i in cbar.get_ticks()])

    # -------------------------------------------------------------------------
    # Set plot options and save result
    # -------------------------------------------------------------------------

    # Define tick positions
    delta = scalebar_size / 2
    xticks, yticks = [], []
    for i in range(5):
        xticks += [center[0] - i * delta, center[0] + i * delta]
        yticks += [center[1] - i * delta, center[1] + i * delta]
    xticks = list(filter(lambda _: 0 < _ < frame.shape[0], xticks))
    yticks = list(filter(lambda _: 0 < _ < frame.shape[0], yticks))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Define which ticks to show
    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        color='white',
        top=use_ticks,
        bottom=use_ticks,
        left=use_ticks,
        right=use_ticks,
        labelleft=False,
        labelbottom=False,
    )

    # Save the results
    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=600)

    return fig


def zerocenter_imshow(ax: Axes) -> None:
    """
    Make sure that he `(vmin, vmax)` range of the `imshow()` plot in
    the given `ax` object is symmetric around zero.

    Args:
        ax: The ax which contains the plot (e.g., `plt.gca()`).
    """

    # Get plot and current limits
    img = ax.get_images()[0]
    vmin, vmax = img.get_clim()

    # Compute and set new limits
    limit = max(np.abs(vmin), np.abs(vmax))
    img.set_clim((-limit, limit))


def zerocenter_plot(ax: Axes, which: str) -> None:
    """
    Make sure that the `xlim` or `ylim` range of the plot object in the
    given `ax` object is symmetric around zero.

    Args:
        ax: The ax which contains the plot (e.g., `plt.gca()`).
        which: Which axis to center around zero, that is, "x" or "y".
    """

    if which == 'x':
        vmin, vmax = ax.get_xlim()
        limit = max(np.abs(vmin), np.abs(vmax))
        ax.set_ylim(ymin=-limit, ymax=limit)
    elif which == 'y':
        vmin, vmax = ax.get_ylim()
        limit = max(np.abs(vmin), np.abs(vmax))
        ax.set_ylim(ymin=-limit, ymax=limit)
    else:
        raise ValueError('Parameter which must be "x" or "y"!')
