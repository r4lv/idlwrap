"""

*idlwrap* helps you port IDL code to python by providing an IDL-like interface to ``numpy`` and ``scipy``.

    You do not need IDL to use *idlwrap*!


usage
-----

An **IDL function or procedure** corresponds to a lowercased function in idlwrap:

.. code:: IDL

    FINDGEN   ->  idlwrap.findgen
    POLY_FIT  ->  idlwrap.poly_fit


All *idlwrap*-specific functions end with an underscore. They have no directly corresponding IDL
functions, they rather map special **IDL syntax**:

.. code:: IDL

    A # B
        ->  idlwrap.operator_(A, "#", B)

    A[1:4,*] = 4
        ->  idlwrap.set_subset_(A, "[1:4,*]", 4)

    FOR I=0, 32000 DO J = I
        ->  for i in idlwrap.range_(0, 32000): j = i



arrays
------

In python, array indices work differently from IDL. When you are used to IDL's array subscripts,
*idlwrap*'s ``subsetify_`` function can be interesting for you.

"""

__author__ = "Ralf Farkas"

import numpy as np
import operator
# import scipy.signal
# import scipy.special
# import re


def findgen(*args, dtype=float):
    """
    Create a (multi-dimensional) range of float values.

    Notes
    -----
    Note that the shape of the output array is *reversed* compared to the
    arguments passed (e.g. ``indgen(2,3,4)`` → shape 4,3,2). For 3D cubes, the
    *last* argument to indgen is the number of frames, but the frame can be
    accessed directly with ``result[n]`` (first subset parameter.)

    The keywords INCREMENT and START are not implemented.

    Examples
    --------
    .. code-block:: IDL

        FINDGEN(n)    ->  np.arange(n)

    """
    args = _list_r_trim(_int_list(args), 1)
    return np.arange(np.prod(args), dtype=dtype).reshape(args[::-1])


def indgen(*shape):
    """
    Create a (multi-dimensional) range of integer values.

    Notes
    -----

    **porting to python**

    If ``shape`` is of one dimension only, you can use ``np.arange(n)``.
    IDL accepts floats as dimension parameters, but applies ``int()`` before
    using them. While ``np.arange()`` also accepts floats, be careful, as the
    number of elements do not match any more!

    .. code-block:: IDL

        INDGEN(5.2)         -> [0,1,2,3,4]
        INDGEN(5)           -> [0,1,2,3,4]
        np.arange(5.2)      -> [0,1,2,3,4,5] ; !!
        np.arange(int(5.2)) -> [0,1,2,3,4]
        np.arange(5)        -> [0,1,2,3,4]

    """
    return findgen(*shape, dtype=int)

def dindgen(*shape):
    """
    Create a (multi-dimensional) range of double-precision float values.
    """
    return findgen(*shape, dtype=np.float64)




def fltarr(*shape, dtype=float):
    """
    Create a float array filled with zeros.

    Parameters
    ----------
    *shape : (multiple) ints, NOT a list
        the dimensions of the new array
    dtype : np.dtype, optional
        dtype object describing the tpe and precision of the values in the new
        array. numpy's default is ``float / np.float32``

    Notes
    -----
    
    - the flag /nozero was omitted.

    **Porting to python**
    
    The core numpy function is ``np.zeros``. Pay attention when passing the
    value ``1`` to ``FLTARR`` (and its sister functions ``INTARR`` and
    ``DBLARR``), as the resulting shape is slightly different: IDL ignores any
    final ``1``s, so for IDL calling ``FLTARR(5, 1, 1, ...)`` is the same as
    ``FLTARR(5)``.

    Examples
    --------
    .. code-block:: IDL

        FLTARR(n)          -> np.zeros(n)
        FLTARR(a, b)       -> np.zeros((b, a))
        FLTARR(a, b, c)    -> np.zeros((c, b, a))
        FLTARR(a, b, 1, 1) -> np.zeros((b, a))

        FLTARR(n)+1        ->  np.ones(n)

    """
    return np.zeros(_list_r_trim(_int_list(shape), 1)[::-1])

def intarr(*shape):
    """
    Create an integer array filled with zeros.
    """
    return fltarr(*shape, dtype=int)

def dblarr(*shape):
    """
    Create a double-precision float array filled with zeros.
    """
    return fltarr(*shape, dtype=np.float64)






def shift(arr, *args):
    """
    **WARNING**

    The ``Si`` arguments can be either a single array containing the shift
    parameters for each dimension, or a sequence of up to eight scalar shift
    values. For arrays of more than one dimension, the parameter ``Sn`` specifies
    the shift applied to the n-th dimension

    while this implementation supports lists as ``arr`` argument, to match the
    style of IDL, the IDLpy bridge does *not* support lists, and returns it
    *unchanged*!

    If ``SHIFT`` is used in combination with ``FFT``, maybe you should look at
    ``np.fft.fftshift``.

    """

    arr = np.asarray(arr) # accept list (see note above)

    if arr.ndim==1:
        if len(args)==1:
            return np.roll(arr, _int_list(args))
    elif arr.ndim==2:
        if len(args)==1:
            return np.roll(arr, _int_list(args))
        if len(args)==2:
            return np.roll(arr, _int_list(args)[::-1], axis=(0,1))
    elif arr.ndim==3:
        if len(args)==1:
            return np.roll(arr, args)
        elif len(args)==1:
            raise IDLException("Incorrect number of arguments.")
        elif len(args)==3:
            return np.roll(arr, args[::-1], axis=(0,1,2))



    raise NotImplementedError("shift does only work for 1D, 2D and 3D arrays.")



def where(array_expression):
    """
    port of IDL's ``WHERE`` function.

    Parameters
    ----------
    array_expression : ndarray / expression
        see examples.

    Returns
    -------
    res : np.ndarray
        List of 'good' indices. If no index was found, ``[-1]`` is returned.


    Examples
    --------
    .. code-block:: IDL

        array = FINDGEN(100)
        B = WHERE(array GT 20)
        values = array[B]

    .. code-block:: python

        array = idlwrap.findgen(100)
        b = idlwrap.where(idlwrap.GT(array, 20))
        # equivalent to `idlwrap.where(a > 20)`
        values = array[b]

        # or even:

        values = array[array > 20]

    Notes
    -----
    see also np.put(a, ind, v), which is roughly equivalent to ``a.flat[ind]=v``

    **porting to python**

    Most of the time, you will use WHERE for subsetting arrays. While this works
    only with indices in IDL (which are returned by WHERE), it work with both
    indices (``idlwrap.where``) and boolean masks (as returned by comparison
    operators like ``array_a < array_b``). You can usually remove
    ``idlwrap.where`` entirely.

    with 2d arrays ``a``, ``b``:

    .. code:: python

        WHERE(a LT b)
            -> idlwrap.where(idlwrap.operator_(a, "LT", b))
            -> idlwrap.where(idlwrap.LT(a, b))
            -> idlwrap.where(a < b)
        # ... in fact, it could even be replaced directly by ``a < b`` (which
        # returns a boolean array in np), if WHERE is used as array index!


    """
    res = [i for i,e in enumerate(array_expression.flatten()) if e]
    if len(res) == 0:
        res = [-1]

    return np.array(res)



def size(arr):
    """
    Size and type information for arrays.


    Parameters
    ----------
    arr : array_like
    

    Returns
    -------
    ndim : int
        Number of dimensions.
    *shape : ints
        First, second, ... dimension.
    dtype : int or np.dtype
        Type of the array, as defined in the `IDL Type Codes and Names <https://www.harrisgeospatial
        .com/docs/size.html>`_, or as ``np.dtype`` object.
    size : int
        Total number of elements.

    """

    type_codes = {
            "undefined": 0,
                "UNDEFINED": 1,
            "byte": 1,
                "BYTE": 1,
            "integer": 2,
                "INT": 2,
            "longword integer": 3,
                "LONG": 3,
            "floating point": 4,
                "FLOAT": 4,
            "double-precision floating": 5,
                "DOUBLE": 5,
            "complex floating": 6,
                "COMPLEX": 6,
            "string": 7,
                "STRING": 7,
            "structure": 8,
                "STRUCT": 8,
            "double-precision complex": 9,
                "DCOMPLEX": 9,
            "pointer": 10,
                "POINTER": 10,
            "object reference": 11,
                "OBJREF": 11,
            "unsigned integer": 12,
                "UINT": 12,
            "unsigned longword integer": 13,
                "ULONG": 13,
            "64-bit integer": 14,
                "LONG64": 14,
            "unsigned 64-bit integer": 15,
                "ULONG64": 15,


            np.dtype(np.int64): 2,   # integer
            np.dtype(np.float64): 4, # floating point
        }


    return (arr.ndim, *arr.shape[::-1], type_codes.get(arr.dtype, arr.dtype), arr.size)




def median(array, width=None, even=False):
    """


    Parameters
    ----------
    array : np.ndarray
        The array to be processed. Array can have only one or two dimensions.
        If Width is not given, Array can have any valid number of dimensions.
    width : np.ndarray
        The size of the one or two-dimensional neighborhood to be used for the
        median filter. The neighborhood has the same number of dimensions as
        array.
    even : bool, optional
        If the EVEN keyword is set when Array contains an even number of points
        (i.e. there is no middle number), MEDIAN returns the average of the two
        middle numbers. The returned value may not be an element of Array . If
        Array contains an odd number of points, MEDIAN returns the median value.
        The returned value will always be an element of Array --even if the EVEN
        keyword is set--since an odd number of points will always have a single
        middle value.



    Notes
    -----------------
    **porting to python**

    As long as ``/EVEN`` is passed to ``MEDIAN``, and no ``WIDTH`` is present,
    it can safely be replaced with ``np.median()``.

    """

    if width is not None:
        if array.ndim == 2:
            import scipy.signal
            w = int(np.floor(width/2)) # 31 -> 15

            array[w:-w, w:-w] = scipy.signal.medfilt2d(array, width)[w:-w, w:-w]
            return array # TODO does this modify the array?


            # array_flt = scipy.signal.medfilt2d(array, width)
            #     # instead of medfilt2d also medfilt can be used!
            # 
            # print("w:", w, "b")
            # array_flt[0:w, :] = array[0:w, :] # left
            # array_flt[-w:, :] = array[-w:, :] # right
            # array_flt[:, 0:w] = array[:, 0:w] # top
            # array_flt[:, -w:] = array[:, -w:] # top
            # 
            # return array_flt

        raise NotImplementedError("``width`` parameter is only implemented for 2d!")

    if array.ndim > 1:
        raise NotImplementedError("ERROR median is not tested with 2d arrays!")

    if not even:
        if len(array)%2 != 0:  # has odd number
            return np.median(array)
        else:
            return np.median(np.concatenate((array, np.array([array.max()]))))
            # this is NOT memory efficient!

    else:
        return np.median(array)



def mean(x):
    """
    
    Parameters
    ----------
    x : np.ndarray
    
    
    Notes
    -----
    The keyword parameters DIMENSION, DOUBLE and NAN are not implemented.
    
    """

    return np.mean(x)




def total(array, dimension=None, integer=False):
    """
    Parameters
    ----------
    array : ndarray
    dimension : int, optional
    integer : bool, optional


    Notes
    -----

    To force ndim >= 1:

    .. code:: python
    
        if res.ndim == 0:
            return np.array([res])
        else:
            return res
    
    **Implementation differences**
    
    not implemented: /CUMULATIVE, /DOUBLE, /NAN, /PRESERVE_TYPE

    **porting to python**

    `TOTAL` corresponds to `ndarray.sum()`. The parameters /DOUBLE and /INTEGER
    can be replicated through the `dtype=...` parameter. DIMENSION needs more
    attention, as the dimensions are reversed. If no DIMENSION is passed, just
    use `np.sum()`.

    .. code:: IDL

        TOTAL(array)  ->  np.sum(array)

    **todo**
    Does IDL support a list as DIMENSION? What happens?

    """

    # sanitize input
    array = np.asarray(array)
    if dimension is not None:
        dimension = array.ndim - dimension

    dtype = None
    if integer:
        dtype = int

    return array.sum(axis=dimension, dtype=dtype)



def finite(x, infinity=False, nan=False, sign=0):
    """
    Identifies whether or not a given argument is finite.
    
    Parameters
    ----------
    x : np.ndarray
        A floating-point, double-precision, or complex scalar or array
        expression. Strings are first converted to floating-point. ????
    infinity : bool, optional
    nan : bool, optional
    sign : int, optional
        Only 0, the default behaviour, is implemented.
    
    Returns
    -------
    is_finite : bool / bool np.ndarray
        If the result is finite.
    
    Notes
    -----
    ``SIGN`` is not implemented. One difficulty arrises from the fact that IDL
    distinguishes between ``-!VALUES.F_NAN`` and ``!VALUES.F_NAN``. In python,
    there is no possibility to distinguish a negative from a positive
    ``np.nan``:
    
    .. code:: python

        a = -np.nan
        b = np.nan

        a == np.nan # -> False
        b == np.nan # -> False

        a < 0   # -> False
        a > 0   # -> False
        b < 0   # -> False
        b > 0   # -> False

    **porting to python**

    if ``SIGN`` is not set:

    .. code:: IDL

        FINITE(..., /NAN) -> np.isnan(...)
        FINITE(..., /INF) -> np.isinf(...)
        FINITE(...)       -> np.isfinite(...)

    if ``SIGN`` is set:

    .. code:: IDL

        ??? 
    
    """



    if sign != 0:
        raise NotImplementedError("``sign`` is not implemented!")

    if infinity:
        if nan:
            raise ValueError("conflicting keywords: infinity and nan")
        return np.isinf(x)
    elif nan:
        if infinity:
            raise ValueError("conflicting keywords: infinity and nan")
        return np.isnan(x)
    else:
        return np.isfinite(x)



#                                  .d888  .d888 888
#       o            o            d88P"  d88P"  888
#      d8b          d8b           888    888    888
#     d888b        d888b          888888 888888 888888
# "Y888888888P""Y888888888P"      888    888    888
#   "Y88888P"    "Y88888P"        888    888    888
#   d88P"Y88b    d88P"Y88b        888    888    Y88b.
#  dP"     "Yb  dP"     "Yb       888    888     "Y888



def fft(array, direction=-1, inverse=False):
    """

    Parameters
    ----------
    array : 2d (?????) np.ndarray
    direction : integer, optional
        Scalar indicating the direction fo the transform, which is negative by
        convention for the forward transform, and positive for the inverse
        transform. The value of direction is ignored if the inverse keyword is
        set.
    inverse : boolean, optional
        Set this keyword to perform an inverse transform. Setting this keyword
        is equivalent to setting the ``direction`` argument to a positive value.
        Note, however, that setting ``inverse`` results in an inverse transform
        even if ``direction`` is specified as negative.

    Returns
    -------

    Notes
    -----
    A normalization factor of 1/N, where N is the number of points, is applied
    during the forward transform.

    **Implementation details**
    
    The parameters ``CENTER``, ``DIMENSION``, ``DOUBLE``, ``OVERWRITE`` and the thread pool
    keywords are not implemented.

    Examples
    --------
    if you do not care about the normalization:

    .. code:: IDL

        FFT(image2d, 1)  -> np.fft.ifft2(image2d)
        FFT(image2d, -1) -> np.fft.fft2(image2d)

    
    """
    # TODO: Does this work for 1d and for 2d transforms?

    # resolve direction/inverse stuff. The truth should be stored in ``inverse``
    # Here, the user supplied `direction=1` and wants an inverse.
    if inverse or (inverse==False and direction>0):
        # inverse transform
        if array.ndim==1:
            return np.fft.ifft(array)*array.size
        elif array.ndim==2:
            return np.fft.ifft2(array)*array.size
        else:
            raise NotImplementedError("unsupported dimension: {}".format(array.ndim))
    else:
        # forward transform, direction = -1
        if array.ndim==1:
            return np.fft.fft(array)/array.size
        elif array.ndim==2:
            return np.fft.fft2(array)/array.size
        else:
            raise NotImplementedError("unsupported dimension: {}".format(array.ndim))




#                                                   888
#       o                                           888
#      d8b                                          888
#     d888b          888d888  8888b.  88888b.   .d88888
# "Y888888888P"      888P"       "88b 888 "88b d88" 888
#   "Y88888P"        888     .d888888 888  888 888  888
#   d88P"Y88b        888     888  888 888  888 Y88b 888
#  dP"     "Yb       888     "Y888888 888  888  "Y88888


def randomn(seed=None, *shape):
    """
    Normal-distributed random numbers.
    
    Parameters
    ----------
    seed : int or 1-d array_like
        seed for random generator.
    *shape : list of int
        dimension of the returned array
    
    
    Notes
    -----
    ``RANDOMN`` uses the Box-Muller method, based off of the ``gasdev``
    algorithm (section 7.2 Numerical Recipies in C, 1992) . The uniform random
    numbers required for the Box-Miller method are generated using the Mersenne
    Twister algorithm. [from the IDL documentation]

    Note that the random numbers generated by python differ from the ones from
    IDL, as the seed is handled differently and the algorithms differ too.
    
    
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(size=shape[::-1])




#                                           888    888
#       o                                   888    888
#      d8b                                  888    888
#     d888b          88888b.d88b.   8888b.  888888 88888b.
# "Y888888888P"      888 "888 "88b     "88b 888    888 "88b
#   "Y88888P"        888  888  888 .d888888 888    888  888
#   d88P"Y88b        888  888  888 888  888 Y88b.  888  888
#  dP"     "Yb       888  888  888 "Y888888  "Y888 888  888


def abs(x):
    """
    absolute value
    """
    return np.abs(x)


def round(x):
    """
    round to the *nearest* integer (-> int type).
    
    Parameters
    ----------
    x : float or array

    Returns
    -------
    x : np.int64 or int64-ndarray
    
    
    Notes
    -----
    ``ROUND`` rounds to the *nearest* integer, unlike numpy's ``np.round`` / ``np.rint``,
    which rounds to the nearest *even* value (defined in the standard IEEE 754)

    https://stackoverflow.com/a/34219827/1943546
    
    **porting to python**

    No direct match. Use this workaround.
    
    """
    return np.trunc(x+np.copysign(0.5,x)).astype(int)

def floor(x):
    """
    
    Parameters
    ----------
    x : float or array

    Returns
    -------
    x : np.int64 or int64-ndarray
    
    
    Notes
    -----
    The keyword L64 is not implemented.

    **porting to python**

    This is basically ``np.floor``, but IDL returns integer types (e.g. used as
    array indices)
    
    """
    return np.floor(x).astype(int)


def ceil(x):
    """
    Round upwards (towards infinity).

    Returns
    -------
    x : np.int64 or int64-ndarray
    """
    return np.ceil(x).astype(int)


def fix(expression):
    """
    Round to nearest integer towards zero.

    Returns
    -------
    x : np.int64 or int64-ndarray
    """
    return np.fix(expression).astype(int)


#                                                           888
#       o                                                   888
#      d8b                                                  888
#     d888b         .d8888b  .d88b.  88888b.d88b.  88888b.  888  .d88b.  888  888
# "Y888888888P"    d88P"    d88""88b 888 "888 "88b 888 "88b 888 d8P  Y8b `Y8bd8P'
#   "Y88888P"      888      888  888 888  888  888 888  888 888 88888888   X88K
#   d88P"Y88b      Y88b.    Y88..88P 888  888  888 888 d88P 888 Y8b.     .d8""8b.
#  dP"     "Yb      "Y8888P  "Y88P"  888  888  888 88888P"  888  "Y8888  888  888
#                                                  888
#                                                  888
#                                                  888


def complex(real, imaginary=0):
    """
    creates complex number. Same as ``idlwrap.dcomplex``

    Parameters
    ----------
    real : float or array or list
    imaginary : float or array or list, optional
    
    """
    return dcomplex(*args, **kwargs)

def dcomplex(real, imaginary=0):
    """
    double-precision complex number


    Parameters
    ----------
    real : float or array or list
        Real part.
    imaginary : float or array or list, optional
        Imaginary part. Defaults to 0.

    Returns
    -------
    complex_number : ndarray

    Notes
    -----
    This always returns a numpy array. Beware of that if you call e.g.
    ``idlwrap.complex(1,2).real``, which results in an 0-dimension np.ndarray.

    The second signature type, with ``Expression, Offset, D1, D2, ...`` is not
    supported.

    """
    return np.asarray(real) + np.asarray(imaginary)*1j



def real_part(z):
    """
    Parameters
    ----------
    z : complex or ndarray
    
    Notes
    -----
    numpy ``.real`` works with complex numbers and ``ndarray``.
    """
    return z.real


def imaginary(complex_expression):
    """
    imaginary part
    """
    return complex_expression.imag


def conj(x):
    """
    complex conjugate
    """
    return x.conj()






#                    888            d8b
#       o            888            Y8P
#      d8b           888
#     d888b          888888 888d888 888  .d88b.
# "Y888888888P"      888    888P"   888 d88P"88b
#   "Y88888P"        888    888     888 888  888
#   d88P"Y88b        Y88b.  888     888 Y88b 888
#  dP"     "Yb        "Y888 888     888  "Y88888
#                                            888
#                                       Y8b d88P
#                                        "Y88P"





def acos(x):
    return np.arccos(x)

def asin(x):
    return np.arcsin(x)


def atan(x):
    return np.arctan(x)




def alog(x):
    return np.log(x)

def alog2(x):
    return np.log2(x)

def alog10(x):
    return np.log10(x)



#                                      d8b
#       o                              Y8P
#      d8b
#     d888b          .d8888b   .d8888b 888
# "Y888888888P"      88K      d88P"    888
#   "Y88888P"        "Y8888b. 888      888
#   d88P"Y88b             X88 Y88b.    888
#  dP"     "Yb        88888P'  "Y8888P 888


def beta(z, w):
    import scipy.special
    return scipy.special.beta(z, w)


def ibeta(a, b, z):
    import scipy.special
    return scipy.special.betainc(a, b, z)



def beselj(x, n):
    """
    Returns the J Bessel function of order N for the argument X.
    
    Parameters
    ----------
    x
        argument.
        A scalar or array specifying the values for which the Bessel function is
        required.
        IDL: Values for X must be in the range -108 to 108. If X is negative
        then N must be an integer (either positive or negative).
    n
        order.
        A scalar or array specifying the order of the Bessel function to
        calculate. Values for N can be integers or real numbers. If N is
        negative then it must be an integer.

    
    Returns
    -------
    my_return_parameter
    
    Notes
    -----
    The output keyword ``ITER``, which returns the number of iterations, was
    omitted. For J Bessel functions, scipy's ``jn`` is just an alias for ``jv``
    (which is not the case for the other Bessel functions, e.g. yn and yv)
    
    **porting to python**

    Replace ``BESELJ(x, n)`` with ``scipy.special.jv(n, x)``. Pay attention to
    the inversed order of the arguments.
    
    """
    import scipy.special
    return scipy.special.jv(n, x)




#                  888    d8b          888
#       o          888    Y8P          888
#      d8b         888                 888
#     d888b        888888 888  .d8888b 888888  .d88b.   .d8888b
# "Y888888888P"    888    888 d88P"    888    d88""88b d88P"
#   "Y88888P"      888    888 888      888    888  888 888
#   d88P"Y88b      Y88b.  888 Y88b.    Y88b.  Y88..88P Y88b.
#  dP"     "Yb      "Y888 888  "Y8888P  "Y888  "Y88P"   "Y8888P



def tic(name=None):
    """

    Returns
    -------
    clock_name : str or None
        The parameter which was passed as ``name``. Pass it, or the ``name``
        directly, to ``toc()`` to get the timing for that particular call to
        ``tic``.

    Notes
    -----

    - The ``/PROFILER`` keyword is not implemented.
    - http://www.harrisgeospatial.com/docs/TIC.html

    """
    import timeit
    if not hasattr(tic, "start"):
        tic.start = {}
    tic.start[name] = timeit.default_timer()

    return name




def toc(name=None):
    import timeit
    stop = timeit.default_timer()

    if not hasattr(tic, "start") or name not in tic.start:
        name_param = repr(name) if name is not None else ""
        raise RuntimeError('no tic({}) was called!'.format(name_param))
    seconds = stop - tic.start[name]
    del tic.start[name]
    pretty_name = "" if name is None else " "+name
    print("Time elapsed {}: {:.6g} seconds.".format(name, seconds))




#                                  888      888
#       o                          888      888
#      d8b                         888      888
#     d888b           8888b.   .d88888  .d88888
# "Y888888888P"          "88b d88" 888 d88" 888
#   "Y88888P"        .d888888 888  888 888  888
#   d88P"Y88b        888  888 Y88b 888 Y88b 888
#  dP"     "Yb       "Y888888  "Y88888  "Y88888


def keyword_set(kw):
    """
    only true if ``kw`` is defined AND different from zero.

    here, ``None`` is used for non-defined keyword.
    """
    return kw is not None and kw!=0


#                    d8b      888 888       888               888
#       o            Y8P      888 888       888               888
#      d8b                    888 888       888               888
#     d888b          888  .d88888 888       88888b.   .d88b.  888 88888b.   .d88b.  888d888
# "Y888888888P"      888 d88" 888 888       888 "88b d8P  Y8b 888 888 "88b d8P  Y8b 888P"
#   "Y88888P"        888 888  888 888       888  888 88888888 888 888  888 88888888 888
#   d88P"Y88b        888 Y88b 888 888       888  888 Y8b.     888 888 d88P Y8b.     888
#  dP"     "Yb       888  "Y88888 888       888  888  "Y8888  888 88888P"   "Y8888  888
#                                                                 888
#                                                                 888
#                                                                 888


def range_(init, limit, increment=1):
    """
    Behaves like IDL's ``FOR i=init, limit DO statement``.

    Parameters
    ----------
    init : int, float
    limit : int, float


    Notes
    -----
    The endpoint ``stop`` is included (``<=`` comparison instead of python's
    ``<``). The ``increment`` is not implemented.

    Examples
    --------

    .. code:: IDL

        FOR I=0, 32000 DO J = I
            ->   for i in range_(0, 3200): j = i


        FOR K=100.0, 1.0, -1 DO BEGIN
            PRINT, K
        ENDFOR

            ->   for k in range_(100.0, 1.0, -1):
                     print(k)



    """
    if increment != 1:
        raise NotImplementedError("only increment=1 supported for range_!")
    
    return np.arange(init, limit+1e-12)



def range_int_(*args):
    """
    Like ``range_``, but returns integers which could then be used as list indices.
    """
    return range_(*args).astype(int)



def _transform_subset(subset, debug=False):
    import re

    def intmap(l):
        """
        this function
        - replaces the empty string "" by None, then
        - calls int() on every element
        """
        return [int(i) if i !="" else None for i in l]

    def parse_slice(s):
        """
        '1:2'    -> (1, 2, None)
        '1:2:-1' -> (1, 2, -1)
        '::-1'   -> (None, None, -1)
        '1:'     -> (1, None, None)
        '1'      -> (1, None, None) #ATTENTION!

        output should be used as `slice(*parse_slice("..."))`
        """
        return intmap(  (s+"::::").split(":", 3)[:3]  )

    def parse_idl_slice(s):
        if debug: print("> parse IDL slice: {}".format(s), end="")
        s = parse_slice(s)
        if s[1] is not None:
            s[1] += 1
        if debug: print(" -> {}".format(s))
        return slice(*s)

    subset = re.sub("[\s\[\]]+", "", subset)  # remove whitespaces and brackets
    other_chars = re.sub("[0-9,:*]", "", subset)
    if other_chars != "":
        raise ValueError("error: only numeric subsets and ':' and '*' are supported.")

    if debug: print("> cleaned subset: '{}'".format(subset))
    parts = subset.split(",")
    
    parts = [parse_idl_slice(p) if type(p)==str and ":" in p else p for p in parts]
    if debug: print("> parsed :", parts)
    
    parts = [slice(None, None, None) if p=="*" else p for p in parts]
    if debug: print("> parsed *", parts)
    
    parts = [int(p) if type(p)==str and p not in ["", ":", "*"] else p for p in parts ]
    if debug: print("> parsed int", parts)
    

    if "" in parts:
        raise ValueError("empty dimensions like [2, ] are not supported in IDL!")
    
    parts = parts[::-1]
    if debug: print("> parts:", parts)

    return tuple(parts)



def subsetify_(arr):
    """
    Transforms a numpy ndarray to an object which implements IDLs array subsetting. This is a
    convenient alternative to the ``subset_`` and ``set_subset_`` functions.

    Returns
    -------
    arr : object
        This object is like an ``ndarray``, but behaves differently when subsetting with a ``str``.


    Examples
    --------

        
    .. code:: python

        # let's create a regular numpy ndarray:
        >>> a = idlwrap.indgen(4, 5)
        >>> a[2:3, 1:2]
        array([[9]])

        # transform b:
        >>> b = idlwrap.subsetify_(a)
        
        # b behaves like a regular numpy ndarray:
        >>> b[2:3, 1:2]
        array([[9]])
        >>> b.mean()
        9.5

        # but when subsetting with a ``str``, it behaves like IDL's subset:
        >>> b["2:3, 1:2"]
        array([[ 6,  7],
              [10, 11]])
        >>> b["*"]
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
               17, 18, 19])
        
        # it also works for setting elements:
        >>> b["1:2,1:3"] = 0
        >>> b
        array([[ 0,  1,  2,  3],
               [ 4,  0,  0,  7],
               [ 8,  0,  0, 11],
               [12,  0,  0, 15],
               [16, 17, 18, 19]])


    """
    return _IDLarray(arr)


class _IDLarray:
    def __init__(self, array):
        self.array = array
    def __getitem__(self, key):
        if type(key)==str:
            return subset_(self.array, key)
        else:
            return self.array.__getitem__(key)
    def __setitem__(self, key, what):
        if type(key)==str:
            set_subset_(self.array, key, what)
        else:
            self.array.__setitem__(key, what)
    def __getattr__(self, name):
        return getattr(self.array, name)
    def __str__(self):
        return self.array.__str__()



def subset_(arr, subset, debug=False):
    """
    Get a subset of an array.

    Parameters
    ----------
    arr : ndarray
        The input array.
    subset : str
        Subset as it would have been passed to IDL, as string. See examples.

    Returns
    -------
    res : ndarray


    Notes
    -----
    In IDL, subset ranges are inclusive: ``[1:3]`` returns 3 elements, while it would only return 2
    elements in python.

    ``idlwrap.subsetify_`` provides an alternative interface to the same functionality.

    Examples
    --------

    .. code:: python

        >>> a = idlwrap.indgen(4, 4)
        >>> idlwrap.subset_(a, "*")
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])

        >>> idlwrap.subset_(a, "[14]")
        14

        >>> idlwrap.subset_(a, "[1:2]")
        array([1, 2])

        >>> idlwrap.subset_(a, "[1:2,2:3]")
        array([[ 9, 10],
               [13, 14]])

    These are not yet implemented:

    .. code:: python

        # idlwrap.subset_(a, "[-1]")   # negative subset
        # idlwrap.subset_(a, "[1.5]")  # float-type subset


    """
    parts = _transform_subset(subset, debug=debug)
    if len(parts)==1:
        return arr.flatten()[parts[0]]
    else:
        return arr.__getitem__(parts)

def set_subset_(arr, subset, what):
    """
    Assign an array subset to a value. The ``arr`` is modified in place. An alternative interface to
    the same functionalities is provided by the ``idlwrap.subsetify_`` function.

    Parameters
    ----------
    arr : ndarray
        The array to use.
    subset : str
        A string with the subset notation, as you would use it in IDL, e.g. ``"[1:4,*]"``. You can
        also omit the brackets ``[]``.
    what : ndarray, numeric
        The value(s) to assign to the selected subset.

    Returns
    -------
    None

    Examples
    --------

    .. code:: python

        a = idlwrap.indgen(10, 10)

        idlwrap.set_subset(a, "[1:4]", 0)
        idlwrap.set_subset(a, "*", 0)
        idlwrap.set_subset(a, "2:5,2:5", 0)

        # the following are valid IDL, but are not yet implemented in idlwrap:

        # idlwrap.set_subset(a, "[1.5]", 0)  # float-type subset
        # idlwrap.set_subset(a, "-1", 0)     # negative subset


    """
    parts = _transform_subset(subset)
    if len(parts)==1:
        raise NotImplementedError("single-element setting not implemented!")
    arr.__setitem__(parts, what)



def matrix_multiply_(a, b, atranspose=False, btranspose=False):
    """


    """

    if atranspose:
        a = a.T
    if btranspose:
        b = b.T

    return H(a,b)


def H(a, b):
    """
    matrix multiplication ("hash"), corresponds to IDL ``A # B``

    Parameters
    ----------
    a : np.ndarray
        supported shapes: (n,) or (n,m)
    b : np.ndarray
        supported shapes: (n,) or (n,m)

    Returns
    -------
    mat : ndarray
        Multiplication of the arrays, as defined by the IDL documentation.

    Notes
    -----
    
    **porting to python**

    Quite complicated, as the numpy function depends on the dimensions of the inputs. Look at the
    source code.

    """
    # TODO: Handle exotic shapes, like (1,5) and (5,1)

    #Maybe also something like np.mgrid, if we only pass ``INDGEN`` as arrays?


    if a.ndim==1 and b.ndim==1:
        return np.multiply(a, b[np.newaxis].T)
    elif a.ndim==2 and b.ndim==2:
        return np.matmul(a.T, b.T).T
    else:
        raise NotImplementedError("only 1d and 2d inputs are supported!")


def HH(a, b):
    """
    matrix multiplication, corresponds to IDL ``A ## B``

    Parameters
    ----------
    a : np.ndarray
        supported shapes: (n,) or (n,m)
    b : np.ndarray
        supported shapes: (n,) or (n,m)

    Returns
    -------
    mat : ndarray
        Multiplication of the arrays, as defined by the IDL documentation.

    Notes
    -----
    
    **porting to python**

    Quite complicated, as the numpy function depends on the dimensions of the inputs. Look at the
    source code.
    """
    if a.ndim==1 and b.ndim==1:
        return np.outer(a, b)
    elif a.ndim==2 and b.ndim==2:
        return np.matmul(a, b) # same as np.dot for 2d arrays
    else:
        raise NotImplementedError("only 1d and 2d inputs are supported!")






# shorthand functions for operator_(...). Can also be used as `operator`
# argument to operator_()
def LE(a, b):
    """
    less-than-or-equal-to relational operator, corresponds to IDL ``a LE b``
    """
    return operator_(a, "LE", b)

def GE(a, b):
    """
    greater-than-or-equal-to relational operator, corresponds to IDL ``a GE b``
    """
    return operator_(a, "GE", b)

def LT(a, b):
    """
    less-than relational operator, corresponds to IDL ``a LT b``
    """
    return operator_(a, "LT", b)

def GT(a, b):
    """
    greater-than relational operator, corresponds to IDL ``a GT b``
    """
    return operator_(a, "GT", b)

def EQ(a, b):
    """
    equals-to relational operator, corresponds to IDL ``a EQ b``
    """
    return operator_(a, "EQ", b)

def NE(a, b):
    """
    not-equal-to relational operator, corresponds to IDL ``a NE b``
    """
    return operator_(a, "NE", b)






def operator_(a, operator, b):
    """
    Special IDL operations.
    
    Parameters
    ----------
    a : numeric or ndarray
    operator : str
        Operation. The following IDL operations are supported:

        - `minimum and maximum operators <http://www.harrisgeospatial.com/docs/Minimum_and_Maximum_Oper.html>`_:
            - ``'<'``: minimum operator
            - ``'>'``: maximum operator
        - `relational operators <http://www.harrisgeospatial.com/docs/Relational_Operators.html>`_:
            - ``'EQ'``: equal to
            - ``'NE'``: not equal to
            - ``'GE'``: greater than or equal to
            - ``'GT'``: greater than
            - ``'LE'``: less than or equal to
            - ``'LT'``: less than
        - `matrix operators <http://www.harrisgeospatial.com/docs/Matrix_Operators.html>`_:
            - ``'#'``: multiplies columns of ``a`` with rows of ``b``. ``b`` must have the same
              number of columns as ``a`` has rows. The resulting array has the same number of
              columns as ``a`` and the same number of rows as ``b``.
            - ``'##'``: multiplies rows of ``a`` with columns of ``b``. ``b`` must have the same
              number of rows as ``a`` has columns. The resulting array has the same number of rows
              as ``a`` and the same number of columns as ``b``.
    
    Returns
    -------
    res : numeric / ndarray

    
    Notes
    -----
    In idlwrap, the relational operators (``EQ``, ``NE``, ``GE``, ``GT``, ``LE``, ``LE``) are also
    available as functions: ``EQ(a, b)``, ...
    
    **Porting to python**

    - the relational operators can be replaced with its python equivalent
    - the minimum and maximum operators ``<`` and ``>`` can be replaced with ``np.minimum(a,b)`` and
      ``np.maximum(a,b)``, respectively ``a < b``
    - the matrix operators are more complex. Please refer to the documentation of ``H`` and
      ``HH``

    Examples
    --------

    .. code:: IDL

        A < B   ->  operator_(a, "<", b)
                ->  np.minimum(a, b)

        A LE B  ->  operator_(a, "LE", b)
                ->  LE(a, b)
                ->  a <= b

        A # B   ->  operator_(a, "#", b)
                ->  H(a, b)

        A ## B  ->  operator_(a, "##", b)
                ->  HH(a, b)
    
    
    """
    # minimum and maximum operators
    # http://www.harrisgeospatial.com/docs/Minimum_and_Maximum_Oper.html
    if operator in ["<"]:
        return np.minimum(a, b)
    elif operator in [">"]:
        return np.maximum(a, b)

    # relational operators
    # http://www.harrisgeospatial.com/docs/Relational_Operators.html
    elif operator in ["EQ", EQ]:
        return a == b
    elif operator in ["NE", NE]:
        return a != b
    elif operator in ["GE", GE]:
        return a >= b
    elif operator in ["GT", GT]:
        return a > b
    elif operator in ["LE", LE]:
        return a <= b
    elif operator in ["LT", LT]:
        return a < b
    
    # matrix operators
    # http://www.harrisgeospatial.com/docs/Matrix_Operators.html
    elif operator in ["#"]:
        return H(a,b)
    elif operator in ["##"]:
        return HH(a,b)

    # unknown operators
    else:
        raise RuntimeError("supported operators: >, <, " # min/max
                           "EQ, NE, GE, GT, LE, LT, " # relational
                           "#, ##") # matrix







#                    888               888
#       o            888               888
#      d8b           888               888
#     d888b          88888b.   .d88b.  888 88888b.
# "Y888888888P"      888 "88b d8P  Y8b 888 888 "88b
#   "Y88888P"        888  888 88888888 888 888  888
#   d88P"Y88b        888  888 Y8b.     888 888 d88P
#  dP"     "Yb       888  888  "Y8888  888 88888P"
#                                          888
#                                          888
#                                          888


def _list_r_trim(l, what=1):
    l = list(l)
    popped = False
    while len(l)>0 and l[-1] == what:
        popped = True
        l.pop()

    if popped and len(l)==0:
        l = [what]

    return l



def _int_list(l):
    return [int(_) for _ in l]
