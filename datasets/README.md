# Data sets

## Using the original data sets from the paper

The final data sets that we used for the experiments in our paper are publicly available [here](https://doi.org/10.17617/3.LACYPN).
To use them with the methods available from `hsr4hci.data`, you should place them in the `output` subdirectory of the respective data set.
For example, the HDF file `beta_pictoris__lp.hdf` containing the Beta Pictoris *L'* data set should be placed at the following location:
```
<path to hsr4hci>/datasets/beta_pictoris__lp/output/beta_pictoris__lp.hdf
```

To see how we created these data sets, feel free to take a look at the `prepare_dataset.py` script.



## Advice for creating your own data sets

In case you want to run our method on your own data set, here is some advice to help you get started.
We are assuming here that your data still comes from the [ESO archives](http://archive.eso.org/cms.html); for data from other sources, some details of the process (e.g., accessing the observing conditions) will be different.

If you have pre-processed your data set using the [PynPoint](https://pynpoint.readthedocs.io/en/latest/) package, you can probably just use the `prepare_dataset.py` script out of the box.
Just create a new folder for your data set, put the PynPoint database in a subdirectory `input`, and add a file `product_ids.txt` (also to `input`).
This file should contain a list of all the product IDs (e.g., `NACO.2013-02-01T01:08:03.817`) of the original FITS files from the ESO archives that went into your data set.
The script needs those to be able to access the headers of the original FITS files, which contain information that are required for obtaining and interpolating the observing conditions.

If you are using a different pre-processing pipeline, or data from a different instrument, you will have to write your own script to bring your data into the right format.
(Or, alternatively, just use your own methods for loading the data and feeding it to the post-processing methods implemented in this repository.)
The methods in `hsr4hci.data` expect the inputs to be HDF files with the following minimal structure:

```text
/metadata                Group
  /metadata/DIT_PSF_TEMPLATE  Dataset {SCALAR}    # in seconds
  /metadata/DIT_STACK         Dataset {SCALAR}    # in seconds
  /metadata/LAMBDA_OVER_D     Dataset {SCALAR}    # in arcsec
  /metadata/ND_FILTER         Dataset {SCALAR}    # transmissibility as a number in (0, 1]
  /metadata/PIXSCALE          Dataset {SCALAR}    # in arcsec / pixel
/observing_conditions    Group
  /observing_conditions/interpolated Group
    /observing_conditions/interpolated/air_mass     Dataset {n_frames}
    /observing_conditions/interpolated/air_pressure Dataset {n_frames}
    / ...
/parang                  Dataset {n_frames}       # in degree
/planets                 Group
  /planets/b             Group
    /planets/b/contrast       Dataset {SCALAR}    # in magnitudes
    /planets/b/position_angle Dataset {SCALAR}    # in degree
    /planets/b/separation     Dataset {SCALAR}    # in arcsec
  / ...
/psf_template            Dataset {x_size, y_size}
/stack                   Dataset {n_frames, x_size, y_size}
```

Most of this should be self-explanatory.
As mentioned above, this is the *minimal* structure that `hsr4hci.data.load_dataset()` expects; for debugging and transparency reasons, the data sets that we provide for download contain even more information.

Note that even if you do not want to use the observing conditions as predictors, you should still create the `/observing_conditions/interpolated` group.
You can simply leave it empty.

---

In case you get stuck with any part of this, feel free to reach out for help! :-)
