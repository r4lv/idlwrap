# idlwrap

*idlwrap* is a python package which provides many functions known from Harris Geospatial's IDL (Interactive Data Language), all implemented in `scipy`/`numpy`.

> No IDL is required to run *idlwrap*, as it is pure python!

With `numpy` and `scipy`, there are powerful and open-source tools available for scientific computing in python. Currently, still lots of scientific projects — especially in astrophysics — rely on the proprietary and expensive IDL instead of moving foward to open and reproducible science. There are many reasons for chosing python over IDL, but transition is not that easy. At least it was until now!

This package aims to abstract away all differences in IDL and python and provide the interface and functions you know from IDL, but using `scipy` and `numpy` under the hood.



## Installation


Install *idlwrap* with pip:

``` bash
pip install idlwrap
```


## Usage

One of the main differences between IDL and python is how arrays and indices are handled. Let's create an array:

``` IDL
IDL> a = INDGEN(3, 4)
IDL> a
       0       1       2
       3       4       5
       6       7       8
       9      10      11
```

That is easy in *idlwrap*:

``` python
>>> a = idlwrap.indgen(3,4)
>>> a
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
```

In IDL, array-indices are *inclusive*:

``` IDL
IDL> a[1:2, 1:2]
       4       5
       7       8
```

while they are *exclusive* in python:

``` python
>>> a[1:2, 1:2]
array([[4]])
```

*idlwrap* can help here too, by making IDL subsetting available as a function:

``` python
>>> idlwrap.subset_(a, "[1:2, 1:2]") 
array([[4, 5],
       [7, 8]])
```


*idlwrap* provides many more functions. Make sure you check the [documentation](https://r4lv.github.io/idlwrap), which is filled with many examples on how to use *idlwrap*, but also provides general information on how to port existing IDL code to python!