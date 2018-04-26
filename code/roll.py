#%% EEL 4930 Final Project
# Christian Marin
# Troy Tharpe
# Dean Fortier

import numpy as np

def window(array, window=(0,), a_step=None, w_step=None):
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)
    
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.") 

    _a_step = np.ones_like(orig_shape)
    if a_step is not None:
        a_step = np.atleast_1d(a_step)
        if a_step.ndim != 1:
            raise ValueError("`a_step` must be either a scalar or one dimensional.")
        if len(a_step) > array.ndim:
            raise ValueError("`a_step` cannot be longer then the `array` dimension.")
        _a_step[-len(a_step):] = a_step
        
        if np.any(a_step < 1):
             raise ValueError("All elements of `a_step` must be larger then 1.")
    a_step = _a_step
    
    _w_step = np.ones_like(window)
    if w_step is not None:
        w_step = np.atleast_1d(w_step)
        if w_step.shape != window.shape:
            raise ValueError("`w_step` must have the same shape as `window`.")
        if np.any(w_step < 0):
             raise ValueError("All elements of `w_step` must be larger then 0.")

        _w_step[:] = w_step
        _w_step[window == 0] = 1
    w_step = _w_step

    if np.any(orig_shape[-len(window):] < window * w_step):
        raise ValueError("`window` * `w_step` larger then `array` in at least one dimension.")

    new_shape = orig_shape
    
    _window = window.copy()
    _window[_window==0] = 1
    
    new_shape[-len(window):] += w_step - _window * w_step
    new_shape = (new_shape + a_step - 1) // a_step
    
    new_shape[new_shape < 1] = 1
    shape = new_shape
    
    strides = np.asarray(array.strides)
    strides *= a_step
    new_strides = array.strides[-len(window):] * w_step
    
    new_shape = np.concatenate((shape, window))
    new_strides = np.concatenate((strides, new_strides))
    
    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]
    
    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)
