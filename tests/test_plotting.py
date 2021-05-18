"""
Tests for plotting.py

Note: Most of these unit tests only ensure if a function can be called
    as expected but do not verify if it produces the "correct" output.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from matplotlib.colorbar import Colorbar

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hsr4hci.plotting import (
    add_colorbar_to_ax,
    adjust_luminosity,
    disable_ticks,
    get_cmap,
    get_transparent_cmap,
    zerocenter_plot,
    zerocenter_imshow,
)

# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test_get_cmap() -> None:

    cmap = get_cmap()
    assert cmap.name == 'RdBu_r'
    assert cmap._rgba_bad == (
        (0.12941176470588237, 0.12941176470588237, 0.12941176470588237, 1.0)
    )


def test_get_transparent_cmap() -> None:

    cmap = get_transparent_cmap('red')
    assert cmap.name == 'from_list'
    assert cmap.colors == [(0, 0, 0, 0), 'red']


def test_add_colorbar_to_ax() -> None:

    fig, ax = plt.subplots()
    img = ax.imshow(np.random.normal(0, 1, (10, 10)))

    # Case 1
    cbar = add_colorbar_to_ax(img=img, fig=fig, ax=ax, where='right')
    assert isinstance(cbar, Colorbar)

    # Case 2
    cbar = add_colorbar_to_ax(img=img, fig=fig, ax=ax, where='left')
    assert isinstance(cbar, Colorbar)

    # Case 3
    cbar = add_colorbar_to_ax(img=img, fig=fig, ax=ax, where='top')
    assert isinstance(cbar, Colorbar)

    # Case 4
    cbar = add_colorbar_to_ax(img=img, fig=fig, ax=ax, where='bottom')
    assert isinstance(cbar, Colorbar)

    # Case 5
    with pytest.raises(ValueError) as error:
        add_colorbar_to_ax(img=img, fig=fig, ax=ax, where='illegal')
    assert 'Illegal value for `where`' in str(error)

    plt.close()


def test_adjust_luminosity() -> None:

    # Case 1
    with pytest.raises(ValueError) as error:
        adjust_luminosity(color='not_a_color')
    assert 'Invalid RGBA argument' in str(error)

    # Case 2
    color = adjust_luminosity('#FF0000', amount=1.4)
    assert np.isclose(color[0], 1.0)
    assert np.isclose(color[1], 0.4)
    assert np.isclose(color[2], 0.4)

    # Case 3
    color = adjust_luminosity((0, 1, 0), amount=1.4)
    assert np.isclose(color[0], 0.4)
    assert np.isclose(color[1], 1.0)
    assert np.isclose(color[2], 0.4)

    # Case 4
    color = adjust_luminosity('blue', amount=1.4)
    assert np.isclose(color[0], 0.4)
    assert np.isclose(color[1], 0.4)
    assert np.isclose(color[2], 1.0)


def test_disable_ticks() -> None:

    fig, ax = plt.subplots()
    ax.imshow(np.random.normal(0, 1, (10, 10)))
    disable_ticks(ax)
    plt.close()


def test_zerocenter_plot() -> None:

    fig, ax = plt.subplots()
    ax.plot([-1, 2], [-3, 4])
    assert ax.get_xlim() == (-1.15, 2.15)
    assert ax.get_ylim() == (-3.35, 4.35)

    # Case 1
    zerocenter_plot(ax, which='x')
    assert ax.get_xlim() == (-2.15, 2.15)

    # Case 2
    zerocenter_plot(ax, which='y')
    assert ax.get_ylim() == (-4.35, 4.35)

    # Case 3
    with pytest.raises(ValueError) as error:
        zerocenter_plot(ax, which='illegal')
    assert 'Parameter which must be "x" or "y"!' in str(error)

    plt.close()


def test_zerocenter_imshow() -> None:

    data = np.zeros((10, 10))
    data[4, 3] = 10
    data[7, 1] = -12

    fig, ax = plt.subplots()
    ax.imshow(data)
    assert ax.get_images()[0].get_clim() == (-12, 10)

    zerocenter_imshow(ax)
    assert ax.get_images()[0].get_clim() == (-12, 12)

    plt.close()
