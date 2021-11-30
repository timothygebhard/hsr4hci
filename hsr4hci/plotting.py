"""
Utility functions for plotting.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import copy
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import colorsys

from astropy.modeling import models, fitting
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap as original_get_cmap
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from photutils import CircularAperture

import matplotlib.colors as mc
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.coordinates import get_center
from hsr4hci.masking import mask_frame_around_position


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
    cbar = fig.colorbar(
        img, cax=cax, orientation=orientation, ticklocation=where
    )

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


def disable_ticks(ax: Axes) -> None:
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
        ax.set_xlim(xmin=-limit, xmax=limit)
    elif which == 'y':
        vmin, vmax = ax.get_ylim()
        limit = max(np.abs(vmin), np.abs(vmax))
        ax.set_ylim(ymin=-limit, ymax=limit)
    else:
        raise ValueError('Parameter which must be "x" or "y"!')


def set_fontsize(ax: Axes, fontsize: int) -> None:
    """
    Set the `fontsize` for all labels (title, x- and y-label, and tick
    labels) of a target axis at once.

    Args:
        ax: The ax which contains the plot (e.g., `plt.gca()`).
        fontsize: The target font size for the labels.
    """

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)


# -----------------------------------------------------------------------------
# AUXILIARY FUNCTION DEFINITIONS AND PLOT_FRAME()
# -----------------------------------------------------------------------------

def _determine_limit(
    frame: np.ndarray,
    positions: Optional[Sequence[Tuple[float, float]]],
) -> float:
    """
    Auxiliary function to determine the plot limits for plot_frame().
    """

    # If no positions are given, simply use the 99.9th percentile of the
    # entire frame as the "global" limit
    if (positions is None) or (not positions):
        return float(np.nanpercentile(np.abs(frame), 99.9))

    # Otherwise, loop over the positions, fit the frame at each position with
    # a 2D Gaussian, and set the limit to the maximum amplitude we find.

    # Define a grid for the fit
    x, y = np.meshgrid(np.arange(frame.shape[0]), np.arange(frame.shape[1]))

    # Keep track of the maximum amplitude (= the limit we will return). This
    # limit should always be positive!
    limit = float(np.nanmin(np.abs(frame)))

    # Loop over all given positions
    for position in positions:

        # Set up the model (and keep the mean = position fixed)
        model = models.Gaussian2D(x_mean=position[0], y_mean=position[1])
        model.x_mean.fixed = True
        model.y_mean.fixed = True

        # Mask the frame (set everything to zero that is too far from position)
        masked_frame = mask_frame_around_position(
            frame=np.nan_to_num(frame), position=position, radius=5
        )

        # Fit the frame and update the limit
        fit_p = fitting.LevMarLSQFitter()
        model = fit_p(model=model, x=x, y=y, z=masked_frame)
        limit = max(limit, model.amplitude.value)

    return limit


def _add_apertures_and_labels(
    ax: Axes,
    positions: Sequence[Tuple[float, float]],
    labels: Sequence[Union[str, float]],
    aperture_radius: float,
    draw_color: MatplotlibColor,
) -> None:
    """
    Auxiliary function for `plot_frame()` to add apertures and labels
    to mark planet positions and indicate the SNR / FPF / ...
    """

    # Define default options for the label
    label_kwargs = dict(
        ha='left',
        va='center',
        color='white',
        fontsize=6,
        bbox=dict(
            facecolor=draw_color,
            edgecolor='none',
            boxstyle='square,pad=0.2',
        ),
    )

    # Draw apertures at positions (if positions are given)
    if positions:
        aperture = CircularAperture(positions=positions, r=aperture_radius)
        aperture.plot(axes=ax, **dict(lw=1, color=draw_color))

    # Add labels for positions (if labels are given)
    if labels and positions:
        for position, label in zip(positions, labels):
            ax.annotate(
                text=label,
                xy=(position[0] + aperture_radius, position[1]),
                xytext=(8, 0),
                textcoords='offset pixels',
                arrowprops=dict(
                    arrowstyle='-',
                    shrinkA=0,
                    shrinkB=0,
                    lw=1,
                    color=draw_color,
                ),
                **label_kwargs,
            )


def _add_scalebar(
    ax: Axes,
    frame_size: Tuple[int, int],
    pixscale: float,
    color: MatplotlibColor = 'white',
    loc: int = 1,
) -> float:
    """
    Auxiliary function for `plot_frame()` to add a scale bar.
    """

    # Compute size of the scale bar, and define its label accordingly
    scalebar_size = 1 / pixscale
    scalebar_label_value = 1.0
    while scalebar_size > 0.3 * frame_size[0]:
        scalebar_size /= 2
        scalebar_label_value /= 2

    # Create the scale bar and add it to the frame
    scalebar = AnchoredSizeBar(
        transform=ax.transData,
        size=scalebar_size,
        label=f'{scalebar_label_value}"',
        loc=loc,
        pad=1,
        color=color,
        frameon=False,
        size_vertical=0,
        fontproperties=fm.FontProperties(size=6),
    )
    ax.add_artist(scalebar)

    return scalebar_size


def _add_ticks(
    ax: Axes,
    frame_size: Tuple[int, int],
    scalebar_size: float,
    color: MatplotlibColor = 'white',
) -> None:
    """
    Auxiliary function for `plot_frame()` to add ticks to the frame.
    """

    # Define shortcut for the center
    center = get_center(frame_size)

    # Define tick positions
    delta = scalebar_size / 2
    xticks, yticks = [], []
    for i in range(10):
        xticks += [center[0] - i * delta, center[0] + i * delta]
        yticks += [center[1] - i * delta, center[1] + i * delta]
    xticks = list(filter(lambda _: 0 < _ < frame_size[0], xticks))
    yticks = list(filter(lambda _: 0 < _ < frame_size[1], yticks))

    # Add ticks to the axis
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Define which ticks to show
    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        color=color,
        length=1.25,
        top=True,
        bottom=True,
        left=True,
        right=True,
        labelleft=False,
        labelbottom=False,
    )


def _add_colorbar(
    img: AxesImage,
    limits: Tuple[float, float],
    fig: Figure,
    ax: Axes,
    use_logscale: bool,
) -> Colorbar:
    """
    Auxiliary function for `plot_frame()` to add a colorbar.
    """

    # Create a color bar at the bottom of the axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.025)
    cbar = fig.colorbar(img, cax=cax, orientation='horizontal')

    # Unpack the limits
    vmin, vmax = limits

    # Set up the rest of the colorbar options
    if use_logscale:
        cbar.set_ticks([vmin / 2, vmin / 10, 0, vmax / 10, vmax / 2])
    else:
        cbar.set_ticks([2 * vmin / 3, vmin / 3, 0, vmax / 3, 2 * vmax / 3])
    cbar.ax.set_xticklabels(["{:.1f}".format(i) for i in cbar.get_ticks()])
    cbar.ax.tick_params(labelsize=5)

    return cbar


def _add_cardinal_directions(
    ax: Axes,
    color: MatplotlibColor = 'white',
) -> None:
    """
    Auxiliary function for `plot_frame()` to add cardinal directions.
    """

    # Define position (i.e., where do the arrows start) and length of arrows
    position = (0.95, 0.05)
    arrow_length = 0.075

    # Define common parameters for annotate()
    params = dict(
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(
            arrowstyle='<-',
            lw=0.75,
            color=color,
            shrinkA=2.5,
            shrinkB=0,
            patchA=None,
            patchB=None,
        ),
        color=color,
        fontsize=6,
        bbox=dict(fc='none', ec='none', pad=0),
    )

    # Plot an arrow for "North" and "East"
    ax.annotate(
        'N',
        xy=position,
        xytext=(position[0], position[1] + arrow_length),
        ha='center',
        va='bottom',
        **params,
    )
    ax.annotate(
        'E',
        xy=position,
        xytext=(position[0] - arrow_length, position[1]),
        ha='right',
        va='center',
        **params,
    )


def plot_frame(
    frame: np.ndarray,
    positions: Sequence[Tuple[float, float]],
    labels: Sequence[Union[str, float]],
    pixscale: float,
    figsize: Tuple[float, float] = (4.3 / 2.54, 5.0 / 2.54),
    subplots_adjust: Optional[Dict[str, float]] = None,
    aperture_radius: float = 0,
    draw_color: MatplotlibColor = 'darkgreen',
    scalebar_color: MatplotlibColor = 'white',
    cmap: str = 'RdBu_r',
    limits: Optional[Tuple[float, float]] = None,
    use_logscale: bool = False,
    add_colorbar: bool = True,
    add_scalebar: bool = True,
    add_cardinal_directions: bool = True,
    scalebar_loc: int = 1,
    file_path: Optional[Union[Path, str]] = None,
) -> Tuple[Figure, Axes, Optional[Colorbar]]:
    """
    Plot a single frame (e.g., a signal estimate) with various options.

    Args:
        frame: A 2D numpy array of shape `(x_size, y_size)` containing
            the frame to be plotted (e.g., a signal estimate).
        positions: A list of positions (which may be empty). At each
            position, an aperture is drawn with the given radius.
        labels: A list of labels (which may be empty) that are placed
            next to the apertures drawn at the `positions`. Can be
            used, for example, to add the SNR or FPF to the plot.
        pixscale: The pixel scale, in units of arcsecond / pixel. Only
            needed if `add_scalebar` is True.
        figsize: A two-tuple `(x_size, y_size)` containing the size of
            the figure in inches.
        subplots_adjust: Dictionary with parameters that will be passed
            to fig.subplots_adjust().
        aperture_radius: The radius of the apertures to be drawn at the
            given `positions`. If `positions` is empty, this value is
            never used.
        draw_color: The color that is used for drawing the apertures and
            also labels.
        scalebar_color: The color that is used for the scale bar and the
            ticks.
        cmap: Name of the color map to be used for plotting.
        limits: A tuple `(vmin, vmax)` that is used for the plot limits.
            If None, the limits are estimated from the data.
        use_logscale: Whether or not to use a (symmetric) log scale for
            the plot / color bar.
        add_colorbar: Whether or not to add a colorbar at the bottom of
            the frame.
        add_scalebar: Whether or not to add a scale bar and a grid of
            ticks around the borders of the frame (to better understand
            the scale of the frame).
        add_cardinal_directions: Whether or not to add labeled arrows
            to indicate the cardinal directions (North and East).
        scalebar_loc: Location parameter for the scalebar. Example:
            loc=1 means "upper right".
        file_path: The path at which to save the resulting plot. The
            path should include the file name plus file ending. If None
            is given, the plot is not saved.

    Returns:
        A 3-tuple containing:
        (1) the current matplotlib figure,
        (2) the current axis containing the plot of the frame, and
        (3) the colorbar object (or None, if add_colobar == False).
    """

    # Define shortcuts
    frame_size = (frame.shape[0], frame.shape[1])
    center = get_center(frame_size)

    # In case no explicit plot limit is specified, determine it from the data
    if limits is None:
        vmax = _determine_limit(frame=frame, positions=positions)
        vmin = -vmax
    else:
        vmin, vmax = limits

    # Set up the `norm`, which determines whether we use linear or log scale
    if use_logscale:
        norm = mc.SymLogNorm(linthresh=0.1 * vmax, vmin=vmin, vmax=vmax)
    else:
        norm = mc.PowerNorm(gamma=1, vmin=vmin, vmax=vmax)

    # Prepare grid for the pcolormesh()
    x, y = np.meshgrid(np.arange(frame.shape[0]), np.arange(frame.shape[1]))

    # Prepare parameters for the adjust_subplots() call
    if subplots_adjust is None:
        subplots_adjust = dict(left=0, top=1, right=1, bottom=0.075)

    # Set up a new figure and adjust margins
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(**subplots_adjust)

    # Create the actual plot
    # Using pcolormesh() instead of imshow() avoids interpolation artifacts in
    # most PDF viewers (otherwise, the PDF version will often look blurry).
    img = ax.pcolormesh(
        x,
        y,
        frame,
        shading='nearest',
        cmap=get_cmap(cmap),
        snap=True,
        rasterized=True,
        norm=norm,
    )
    ax.set_aspect('equal')

    # Place a "+"-marker at the center for the frame
    ax.plot(center[0], center[1], '+', ms=6, color='black')

    # Plot apertures and add labels
    if positions:
        _add_apertures_and_labels(
            ax, positions, labels, aperture_radius, draw_color
        )

    # If desired, add a scale bar and a grid of ticks
    if add_scalebar:
        scalebar_size = _add_scalebar(
            ax, frame_size, pixscale, scalebar_color, scalebar_loc
        )
        _add_ticks(ax, frame_size, scalebar_size, scalebar_color)
    else:
        disable_ticks(ax)

    # If desired, add the cardinal directions
    if add_cardinal_directions:
        _add_cardinal_directions(ax, scalebar_color)

    # If desired, add a color bar
    if add_colorbar:
        cbar = _add_colorbar(img, (vmin, vmax), fig, ax, use_logscale)
    else:
        cbar = None

    # Save the results, if desired
    if file_path is not None:
        plt.savefig(file_path, pad_inches=0, dpi=600)

    return fig, ax, cbar
