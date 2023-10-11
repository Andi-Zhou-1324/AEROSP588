"""
This script tests your uncon_optimizer on two easy problems:
- Slanted quadratic function
- Bean function

To run tests, put this file in the same folder as your `uncon_optimizer.py`, and run
$ python tests_uncon_optimizer.py

NOTE:
- For your report, you should test your optimizer on the slanted quadratic function and the n-dimensional Rosenbrock function. (Not the bean function)
- Your optimizer must pass these tests; otherwise, you'll lose significant points. The autograder will run the same tests.
- I set loose tolerance here, but the competition uses tight tolerance of |g|_infinity <= 1e-6.
- You don't need to understand or edit this script.
- Do not upload this script to autograder when submitting. Only upload `uncon_optimizer.py` and other Python files called by `uncon_optimizer.py`.
"""

import numpy as np
import unittest

# import your optimizer
from uncon_optimizer import uncon_optimizer


class FunctionWrapperWithCounter():
    """
    Attach a function call counter to a user-defined function

    To use:
    --------------------------------------------------
    # instantiate
    func = FunctionWrapperWithCounter(user_defined_function)

    # call function
    func(x)

    # get function calls
    fcalls = func.get_fcalls()
    --------------------------------------------------
    """

    def __init__(self, func):
        self._func = func
        self._fcalls = 0   # initialize function call counter
        self._max_fcalls = 10000   # maximum iterations

    def __call__(self, x):
        """ calls the user-provided function and count the function calls"""
        self._fcalls += 1

        # kill and exit if the function calls reached the limit
        if self._fcalls >= self._max_fcalls:
            raise RuntimeError('You reached the function call limit of ' + str(self._max_fcalls) + '. Exit.')
        
        return self._func(x)

    def get_fcalls(self):
        """ get function calls"""
        return self._fcalls


def func_slanted_quad(x):
    """
    Slanted quadratic function from Appendix D of the book

    Parameters
    ----------
    x : ndarray, shape (2,)
        design variables
    
    Returns
    -------
    f : float
        function value
    g : ndarray, shape (2,)
        objective gradient
    """

    beta = 1.5

    f = x[0]**2 + x[1]**2 - beta * x[0] * x[1]

    g = np.zeros(2)
    g[0] = 2 * x[0] - beta * x[1]
    g[1] = 2 * x[1] - beta * x[0]

    return f, g


def func_bean(x):
    """
    Bean function from Appendix D of the book

    Parameters
    ----------
    x : ndarray, shape (2,)
        design variables
    
    Returns
    -------
    f : float
        function value
    g : ndarray, shape (2,)
        objective gradient
    """

    x1 = x[0]
    x2 = x[1]

    f = (1. - x1)**2 + (1. - x2)**2 + 0.5 * (2 * x2 - x1**2)**2

    g = np.zeros(2)
    g[0] = -2. * (1 - x1) - 2. * x1 * (2 * x2 - x1**2)
    g[1] = -2. * (1 - x2) + 2. * (2 * x2 - x1**2)

    return f, g


class TestsYouMustPass(unittest.TestCase):

    def _run_actual_test(self, func_in, x0, xopt_ref, fopt_ref):
        # add function call counter to the function
        func = FunctionWrapperWithCounter(func_in)

        # run optimizaiton
        xopt, fopt, output = uncon_optimizer(func, x0, epsilon_g=1e-6, options=None)

        # check optimal x and f
        np.testing.assert_allclose(xopt, xopt_ref, rtol=0, atol=1e-3, err_msg='x_opt is wrong')
        np.testing.assert_allclose(fopt, fopt_ref, rtol=0, atol=1e-3, err_msg='f_opt is wrong')

        # check objective gradient at the optimized point
        _, gopt = func_in(xopt)   # call original function to not increment the call counter
        gnorm = np.max(np.abs(gopt))
        self.assertTrue(gnorm <= 1e-5, msg='Norm of objective gradient at x_opt did not satisfy |g|_infinity < 1e-6')

        # check if alias is provided
        self.assertIn('alias', output, msg='Could not get your alias from `output`. Make sure to set your alias as instructed in the template.')

        # get function calls
        ### print('fcalls =', func.get_fcalls())
        
    def test_slanted_quad(self):
        """Slanted quadratic function"""

        # optimal solution
        xopt_ref = np.zeros(2)
        fopt_ref = 0.

        # initial point
        x0 = np.array([0., 3.])

        # run test
        self._run_actual_test(func_slanted_quad, x0, xopt_ref, fopt_ref)

        print('-----------------------------------------------------------------------------')
        print('   Your optimizer successfully solved the slanted quadratic function problem.')
        print('-----------------------------------------------------------------------------\n')

    def test_bean(self):
        """Bean function"""

        # optimal solution
        xopt_ref = np.array([1.21341, 0.824123])
        fopt_ref = 0.09194

        # initial point
        x0 = np.array([0., 0.])

        # run test
        self._run_actual_test(func_bean, x0, xopt_ref, fopt_ref)

        print('-----------------------------------------------------------------------------')
        print('   Your optimizer successfully solved the bean problem.')
        print('-----------------------------------------------------------------------------\n')


if __name__ == '__main__':
    unittest.main()
