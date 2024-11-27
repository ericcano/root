# Author: Aaron Jomy CERN 09/2024
# Author: Vincenzo Eduardo Padulano CERN 09/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
/**
\class TRandom
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
## PyROOT

The TRandom class has several additions for its use from Python, which are also
available in its subclasses TRandom2 and TRandom3.



First, TRandom instance can be initialized with user-defined Python functions. Given a generic Python callable,
the following can performed:

\code{.py}
def func(x: numpy.ndarray, pars: numpy.ndarray) -> float:
    return pars[0] * x[0] * x[0] + x[1] * pars[0]

my_func = ROOT.TF1("my_func", func, -10, 10, npar=2, ndim=2)
\endcode

Second, after performing the initialisation with a Python functor, the TF1 instance can be evaluated using the Pythonized
`TF1::EvalPar` function. The pythonization allows passing in 1D(single set of x variables) or 2D(a dataset) NumPy arrays.

The following example shows how we can create a TF1 instance with a Python function and evaluate it on a dataset:

\code{.py}
import ROOT
import math
import numpy as np

def pyf_tf1_coulomb(x, p):
    return p[1] * x[0] * x[1] / (p[0]**2) * math.exp(-p[2] / p[0])

rtf1_coulomb = ROOT.TF1("my_func", pyf_tf1_coulomb, -10, 10, ndims = 2, npars = 3)

# x dataset: 5 pairs of particle charges
x = np.array([
    [1.0, 10, 2.0],
    [1.5, 10, 2.5],
    [2.0, 10, 3.0],
    [2.5, 10, 3.5],
    [3.0, 10, 4.0]
])

params = np.array([
    [1.0],       # Distance between charges r
    [8.99e9],    # Coulomb constant k (in N·m²/C²)
    [0.1]        # Additional factor for modulation
])

# Slice to avoid the dummy column of 10's
res = rtf1_coulomb.EvalPar(x[:, ::2], params)
        
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

# Pythonic wrappers to TRandom based on the interface of numpy.random.Generator.random()
# random.Generator.random(size=None, dtype=np.float64, out=None)

# out / size mismatch leads to:
# >>> g.random(out=out, size=s2)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "numpy/random/_generator.pyx", line 356, in numpy.random._generator.Generator.random
#   File "_common.pyx", line 304, in numpy.random._common.double_fill
#   File "_common.pyx", line 288, in numpy.random._common.check_output
# ValueError: size must match out.shape when used together

from . import pythonization

def _TRandom_Gaus(self, size=None, out=None, mean=0, sigma=1):

    import ROOT
    import numpy

    if size == None and out == None:
        return self._Gaus(mean, sigma)
    if out != None:
        sz = numpy.shape(out)
        if size != None and size != sz:
            raise ValueError("size must match out.shape when used together")
        out = out.reshape(shape=-1)
        self.GausN(out.size(), out, mean, sigma)
        out = out.reshape(shape=sz)
        return None
    else:
        sz = numpy.prod(size)
        ret = numpy.zeros(sz)
        self.GausN(sz, ret, mean, sigma)
        return ret.reshape(shape=size)
    

@pythonization('TRandom')
def pythonize_trandom(klass):   

    # Pythonizations for TH1::EvalPar
    klass._Gaus = klass.Gaus
    klass.Gaus = _TRandom_Gaus
