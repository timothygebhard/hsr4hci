"""
Tests for plotting.py

Note: Most of these unit tests only ensure if a function can be called
    as expected but do not verify if it produces the "correct" output.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from _pytest.tmpdir import TempPathFactory
from matplotlib.colorbar import Colorbar

from astropy.modeling import models

import matplotlib.pyplot as plt
import numpy as np
import pytest

# noinspection PyProtectedMember
from hsr4hci.plotting import (
    _add_apertures_and_labels,
    _add_colorbar,
    _add_scalebar,
    _add_ticks,
    _determine_limit,
    add_colorbar_to_ax,
    adjust_luminosity,
    disable_ticks,
    get_cmap,
    get_transparent_cmap,
    plot_frame,
    set_fontsize,
    zerocenter_imshow,
    zerocenter_plot,
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


def test_set_fontsize() -> None:

    fig, ax = plt.subplots()
    ax.set_title('Test')
    set_fontsize(ax=ax, fontsize=6)
    assert ax.title.get_fontsize() == 6
    plt.close()


def test__determine_limit() -> None:

    # Case 1
    frame = np.arange(10_000).reshape(100, 100)
    limit = _determine_limit(frame=frame, positions=[])
    assert limit == 9989.001

    # Case 2
    frame_size = (51, 51)
    positions = [(11.0, 8.2), (30.2, 20.5), (10.0, 30.0)]
    amplitudes = [1, 2, 3]
    frame = np.zeros(frame_size)
    x, y = np.meshgrid(np.arange(frame_size[0]), np.arange(frame_size[1]))
    for a, (x_mean, y_mean) in zip(amplitudes, positions):
        gaussian = models.Gaussian2D(x_mean=x_mean, y_mean=y_mean, amplitude=a)
        frame += gaussian(x, y)
    limit = _determine_limit(frame=frame, positions=positions)
    assert np.isclose(limit, 3)


def test__add_apertures_and_labels() -> None:

    fig, ax = plt.subplots()
    ax.imshow(np.zeros((50, 50)))

    # Case 1
    _add_apertures_and_labels(
        ax=ax,
        positions=[],
        labels=[],
        aperture_radius=0,
        draw_color='red',
    )

    # Case 2
    _add_apertures_and_labels(
        ax=ax,
        positions=[(15, 15), (30, 8)],
        labels=['Label 1', 'Label 2'],
        aperture_radius=2,
        draw_color='red',
    )

    plt.close()


def test__add_scalebar() -> None:

    fig, ax = plt.subplots()
    ax.imshow(np.zeros((50, 50)))

    # Case 1
    scalebar_size = _add_scalebar(ax=ax, frame_size=(50, 50), pixscale=0.0271)
    assert np.isclose(scalebar_size, 9.22509225)

    # Case 1
    scalebar_size = _add_scalebar(ax=ax, frame_size=(70, 70), pixscale=0.0271)
    assert np.isclose(scalebar_size, 18.4501845)

    plt.close()


def test__add_ticks() -> None:

    fig, ax = plt.subplots()
    ax.imshow(np.zeros((50, 50)))

    # Case 1
    _add_ticks(ax=ax, frame_size=(50, 50), scalebar_size=9.22509225)

    plt.close()


def test__add_colorbar() -> None:

    fig, ax = plt.subplots()
    img = ax.imshow(np.random.normal(0, 1, (50, 50)))

    # Case 1
    _add_colorbar(img=img, limits=(-3, 3), fig=fig, ax=ax, use_logscale=True)

    # Case 2
    _add_colorbar(img=img, limits=(-5, 5), fig=fig, ax=ax, use_logscale=False)

    plt.close()


def test_plot_frame(tmp_path_factory: TempPathFactory) -> None:

    test_dir = tmp_path_factory.mktemp('plotting', numbered=False)
    frame = np.random.normal(0, 1, (51, 51))

    # Case 1
    fig, ax, cbar = plot_frame(
        frame=frame,
        positions=[],
        labels=[],
        pixscale=0.0271,
        figsize=(4.0, 4.0),
        subplots_adjust=None,
        aperture_radius=2,
        draw_color='darkgreen',
        scalebar_color='#00FF00',
        cmap='RdBu_r',
        limits=(-5, 5),
        use_logscale=False,
        add_colorbar=True,
        add_scalebar=True,
        scalebar_loc=1,
        file_path=None,
    )
    plt.close(fig)
    assert cbar is not None

    # Case 1
    fig, ax, cbar = plot_frame(
        frame=frame,
        positions=[(20, 20)],
        labels=['test'],
        pixscale=0.0271,
        figsize=(4.0, 4.0),
        subplots_adjust=None,
        aperture_radius=2,
        draw_color='darkgreen',
        scalebar_color='white',
        cmap='viridis',
        limits=None,
        use_logscale=True,
        add_colorbar=False,
        add_scalebar=True,
        scalebar_loc=2,
        file_path=test_dir / 'test.pdf',
    )
    plt.close(fig)
    assert cbar is None

    # Case 2
    fig, ax, cbar = plot_frame(
        frame=frame,
        positions=[(20, 20), (10, 10)],
        labels=['test', 'test 2'],
        pixscale=0.0271,
        figsize=(4.0, 4.0),
        subplots_adjust=dict(left=0.005, top=1, right=0.995, bottom=0.105),
        aperture_radius=2,
        draw_color='darkgreen',
        scalebar_color='red',
        cmap='magma',
        limits=None,
        use_logscale=True,
        add_colorbar=False,
        add_scalebar=False,
        scalebar_loc=3,
        file_path=test_dir / 'test.png',
    )
    plt.close(fig)
    assert cbar is None
