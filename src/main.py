#!/usr/bin/env python2
from __future__ import division, print_function
from astroML.decorators import pickle_results
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy


@pickle_results("x-squared.pkl")
def square(x):
    return x*x


def main():
    x = np.arange(10)

    print(square(x))


if __name__ == "__main__":
    main()
