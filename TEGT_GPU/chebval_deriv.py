#import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import elementwise_grad
def chebyshev_t(x):
    c = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352, -0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906])
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray):
        c = c.reshape(c.shape + (1,)*x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    return c0 + c1 * x    
#what i think it is
r = np.linspace(-0.95,0.95,100)
dr = 1e-3
lpp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352, -0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906])
dV_pppi = np.array([(chebyshev_t( yi+dr)-chebyshev_t( yi-dr))/2/dr for yi in r])
plt.plot(r,dV_pppi,label="maybe deriv")

#what it actually is
r = np.linspace(-0.95,0.95,100)
lpp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352, -0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906])
V_pppi = np.array([chebyshev_t( yi) for yi in r])
dV_pppi_fd = np.gradient(V_pppi)

plt.plot(r,dV_pppi_fd,label = "ref deriv fd")
plt.plot(r,V_pppi,label = "chebval")

#autograd
r = np.linspace(-0.95,0.95,100)
lpp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352, -0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906])
dV_pppi = elementwise_grad(chebyshev_t)(r)
plt.plot(r,dV_pppi,label="autograd deriv")

plt.legend()
plt.savefig("chbval_deriv.png")
plt.clf()