"""
Utility functions for selecting predictors (and targets).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from hsr4hci.utils.masking import get_circle_mask

from typing import Optional


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_predictor_mask(mask_size: tuple,
                       position: tuple,
                       n_regions: int = 1,
                       region_size: Optional[int] = None):

    mask = np.zeros(mask_size)
    center = (mask_size[0] / 2, mask_size[1] / 2)
    
    complex_position = ((position[0] - center[0]) * 1j +
                        (position[1] - center[1]))
    r, phi = np.abs(complex_position), np.angle(complex_position)
    
    theta = np.deg2rad(360 / (n_regions + 1))
    
    # Loop over regions to be created
    for i in range(n_regions):
        
        new_complex_position = r * np.exp(1j * (phi + (i + 1) * theta))
        new_position = (int(np.imag(new_complex_position) + center[0]),
                        int(np.real(new_complex_position) + center[1]))
        
        if region_size is None:
            mask[new_position] = 1
        else:
            disk = get_circle_mask(mask_size=mask_size,
                                   radius=region_size,
                                   center=new_position)
            mask = np.logical_or(mask, disk)
    
    return mask
