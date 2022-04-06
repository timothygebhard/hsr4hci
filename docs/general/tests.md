# Tests

Once you have set up `hsr4hci`, you can make sure that everything works as intended by running the set of unit and integration tests that we provide in the `tests` directory.
The tests are based on [pytest](https://docs.pytest.org).
If you did not install `hsr4hci` with the `[develop]` options, you might have to install this first:

```bash
pip install pytest
```

With pytest installed, you can run the tests simply by executing the following command in the root directory of the `hsr4hci` repository that you have cloned from GitHub:

```bash
pytest tests
```

If everything is working correctly, you should get an output like the following:

```pytest
============================= test session starts ==============================
platform darwin -- Python 3.8.12, pytest-6.2.5, py-1.11.0, pluggy-1.0.0
rootdir: /path/to/hsr4hci
plugins: cov-3.0.0
collected 127 items

tests/integration/test_integration.py ..                                 [  1%]
tests/unit/test_base_models.py .                                         [  2%]
tests/unit/test_config.py ....                                           [  5%]
tests/unit/test_contrast.py ..                                           [  7%]
tests/unit/test_coordinates.py ...                                       [  9%]
tests/unit/test_data.py .........                                        [ 16%]
tests/unit/test_derotating.py ..                                         [ 18%]
tests/unit/test_fits.py ..                                               [ 19%]
tests/unit/test_forward_modeling.py ...                                  [ 22%]
tests/unit/test_general.py ...........                                   [ 30%]
tests/unit/test_hdf.py ....                                              [ 33%]
tests/unit/test_htcondor.py ..                                           [ 35%]
tests/unit/test_hypotheses.py ..                                         [ 37%]
tests/unit/test_importing.py .                                           [ 37%]
tests/unit/test_masking.py ..........                                    [ 45%]
tests/unit/test_match_fraction.py ..                                     [ 47%]
tests/unit/test_merging.py .....                                         [ 51%]
tests/unit/test_metrics.py ..                                            [ 52%]
tests/unit/test_observing_conditions.py .....                            [ 56%]
tests/unit/test_pca.py .                                                 [ 57%]
tests/unit/test_photometry.py .........                                  [ 64%]
tests/unit/test_plotting.py ...............                              [ 76%]
tests/unit/test_positions.py ...                                         [ 78%]
tests/unit/test_psf.py .                                                 [ 79%]
tests/unit/test_residuals.py ....                                        [ 82%]
tests/unit/test_splitting.py .                                           [ 83%]
tests/unit/test_time_conversion.py .....                                 [ 87%]
tests/unit/test_training.py .......                                      [ 92%]
tests/unit/test_typehinting.py ...                                       [ 95%]
tests/unit/test_units.py ...                                             [ 97%]
tests/unit/test_utils.py ...                                             [100%]

============================= 127 passed in 47.63s =============================
```