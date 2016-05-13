"""
Markov Chain Monte Carlo functions.
"""
from numpy.polynomial import polynomial as P

def smoothing_poly_lnprior(poly, degree, xmin, xmax, gamma=1):
    """
    A smoothing prior that suppresses higher order derivatives of a polynomial,
    poly = a + b x + c x*x + ..., described by a vector of its coefficients,
    [a, b, c, ...].

    Functional form is:

    ln p(poly coeffs) =
      -gamma * integrate( (diff(poly(x), x, degree))^2, x, xmin, xmax)

    So it takes the `degree`th derivative of the polynomial, squares it,
    integrates that from xmin to xmax, and scales by -gamma.
    """
    # Take the `degree`th derivative of the polynomial.
    poly_diff = P.polyder(poly, m=degree)
    # Square the polynomial.
    poly_diff_sq = P.polypow(poly_diff, 2)
    # Take the indefinite integral of the polynomial.
    poly_int_indef = P.polyint(poly_diff_sq)
    # Evaluate the integral at xmin and xmax to get the definite integral.
    poly_int_def = (
        P.polyval(xmax, poly_int_indef) - P.polyval(xmin, poly_int_indef)
    )
    # Scale by -gamma to get the log prior
    lnp = -gamma * poly_int_def

    return lnp
