# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Given an ill-posed inverse problem
    x = arg min{||Ax-C||^2} (1)
one can use regularization schemes can be used to stabilize the system and find a numerical solution.
    x_lambda = arg min{||Ax-C||^2 + lambda*R(x)} (2)
where R(x) represents the penalization term.
"""

import numpy as np
from typing import Optional, Union, List

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import logging
logger = logging.getLogger(__name__)
from typing import List


from qiskit.aqua.operators.gradients.gradient import Gradient
from qiskit.aqua.operators.gradients.qfi.qfi import QFI
from qiskit.circuit import Parameter, ParameterVector

from qiskit.aqua.operators import (OperatorBase, ListOp)


class NaturalGradient(GradientBase):
    """Convert an operator expression to the first-order gradient."""

    # pylint: disable=arguments-differ
    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[ParameterVector, Parameter, List[Parameter]]] = None,
                method: str = 'param_shift',
                regularization: Optional[str] = None,
                approx: Optional[str] = None
                ) -> OperatorBase:
        r"""
        Args:
            operator: The operator we are taking the gradient of
            params: The parameters we are taking the gradient with respect to.
            method: The method used to compute the state/probability gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``. Deprecated for observable gradient.
            regularization: Use the following regularization with an lstsq method to solve the underlying SLE
                Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``
                ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search
                If regularization is None but the metric is ill-conditioned or singular then a lstsq solver is
                used without regularization
            approx: Which approximation of the QFI to use: [None, 'diagonal', 'block_diagonal']

        Returns:
            An operator whose evaluation yields the NaturalGradient.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
        """
        grad = Gradient().convert(operator, params, method)
        metric = QFI().convert(operator, params, approx) * 0.25

        def combo_fn(x):
            c = x[0]
            a = x[1]
            if regularization:
                nat_grad = regularized_lse_solver(a, c, regularization=regularization)
            else:
                try:
                    nat_grad = np.linalg.solve(a, c)
                except np.LinAlgError:
                    nat_grad = np.linalg.lstsq(a, c)
            return nat_grad

        return ListOp([grad, metric], combo_fn=combo_fn)

@staticmethod
def reg_term_search(A: np.ndarray,
                    C: np.ndarray, reg_method, lambda1=1e-3, lambda4=1., tol=1e-8):
    """
    This method implements a search for a regularization parameter lambda by finding for the corner of the L-curve
    More explicitly, one has to evaluate a suitable lambda by finding a compromise between the error in the
    solution and the norm of the regularization.
    This function implements a method presented in
    `A simple algorithm to find the L-curve corner in the regularization of inverse problems
     <https://arxiv.org/pdf/1608.04571.pdf>`
    Args:
        A (2D array): see (1) and (2)
        C (1D array): see (1) and (2)
        reg_method (funct): Given A, C and lambda the regularization method must return x_lambda - see (2)
        lambda1 (float): left starting point for L-curve corner search
        lambda4 (float): right starting point for L-curve corner search
        tol (float): termination threshold

    Returns:
        int: regularization term - lambda
    """
    def get_curvature(x_lambda):
        """
        Calculate Menger curvature
        Menger, K. (1930).  Untersuchungen  ̈uber Allgemeine Metrik. Math. Ann.,103(1), 466–501

        Args:
            x_lambda (list): [x_lambdaj, x_lambdak, x_lambdal]
            lambdaj < lambdak < lambdal

        Returns: Menger Curvature

        """
        eps=[]
        eta=[]
        for x in x_lambda:
            try:
                eps.append(np.log(np.linalg.norm(np.matmul(A, x) - C)**2))
            except Exception:
                eps.append(np.log(np.linalg.norm(np.matmul(A, np.transpose(x)) - C) ** 2))
            eta.append(np.log(np.linalg.norm(x)**2))
        p_temp = 1
        C_k = 0
        for i in range(3):
            try:
                p_temp *= (eps[np.mod(i+1, 3)] - eps[i])**2 + (eta[np.mod(i+1, 3)] - eta[i])**2
            except Exception:
                pass
            C_k += eps[i] * eta[np.mod(i+1, 3)] - eps[np.mod(i+1, 3)] * eta[i]
        C_k = 2 * C_k / np.sqrt(p_temp)
        return C_k

    def get_lambda2_lambda3(lambda1, lambda4):
        gold_sec = (1+np.sqrt(5))/2.
        lambda2 = 10**((np.log10(lambda4) + np.log10(lambda1)*gold_sec) / (1 + gold_sec))
        lambda3 = 10**(np.log10(lambda1) + np.log10(lambda4) - np.log10(lambda2))
        return lambda2, lambda3

    lambda2, lambda3 = get_lambda2_lambda3(lambda1, lambda4)
    lambda_ = [lambda1, lambda2, lambda3, lambda4]
    x_lambda = []
    for l in lambda_:
        x_lambda.append(reg_method(A, C, l))
    counter = 0
    while (lambda_[3]-lambda_[0])/lambda_[3] >= tol:
        counter += 1
        C2 = get_curvature(x_lambda[:-1])
        C3 = get_curvature(x_lambda[1:])
        while C3 < 0:
            lambda_[3] = lambda_[2]
            x_lambda[3] = x_lambda[2]
            lambda_[2] = lambda_[1]
            x_lambda[2] = x_lambda[1]
            lambda2, _ = get_lambda2_lambda3(lambda_[0], lambda_[3])
            lambda_[1] = lambda2
            x_lambda[1] = reg_method(A, C, lambda_[1])
            C3 = get_curvature(x_lambda[1:])

        if C2 > C3:
            lambda_mc = lambda_[1]
            x_mc = x_lambda[1]
            lambda_[3] = lambda_[2]
            x_lambda[3] = x_lambda[2]
            lambda_[2] = lambda_[1]
            x_lambda[2] = x_lambda[1]
            lambda2, _ = get_lambda2_lambda3(lambda_[0], lambda_[3])
            lambda_[1] = lambda2
            x_lambda[1] = reg_method(A, C, lambda_[1])
        else:
            lambda_mc = lambda_[2]
            x_mc = x_lambda[2]
            lambda_[0] = lambda_[1]
            x_lambda[0] = x_lambda[1]
            lambda_[1] = lambda_[2]
            x_lambda[1] = x_lambda[2]
            _, lambda3 = get_lambda2_lambda3(lambda_[0], lambda_[3])
            lambda_[2] = lambda3
            x_lambda[2] = reg_method(A, C, lambda_[2])
    return lambda_mc, x_mc

@staticmethod
def ridge(A: np.ndarray,
          C: np.ndarray,
          lambda_=1., auto_search=True, lambda1=1e-3, lambda4=1., tol_search=1e-8,
          fit_intercept=True, normalize=False, copy_A=True, max_iter=None, tol=0.0001, solver='auto',
          random_state=None):
    """
    Ridge Regression with automatic search for a good regularization term lambda
    x_lambda = arg min{||Ax-C||^2 + lambda*||x||_2^2} (3)
    `Scikit Learn Ridge Regression
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`
    Args:
        A: see (1) and (2)
        C: see (1) and (2)
        lambda_ (float): regularization parameter used if auto_search = False
        auto_search (bool): if True then use reg_term_search to find a good regularization parameter
        lambda1 (float): left starting point for L-curve corner search
        lambda4 (float): right starting point for L-curve corner search
        tol_search (float): termination threshold for regularization parameter search
        fit_intercept (bool): if True calculate intercept
        normalize (bool): deprecated if fit_intercept=False, if True normalize A for regression
        copy_A (bool): if True A is copied, else overwritten
        max_iter (int): max. number of iterations if solver is CG
        tol (float): precision of the regression solution
        solver (str): solver {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}
        random_state (int): seed for the pseudo random number generator used when data is shuffled

    Returns:
        array(int): Solution to the regularization inverse problem

    """

    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=lambda_, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_A, max_iter=max_iter,
                tol=tol, solver=solver, random_state=random_state)
    if auto_search:
        def reg_method(A, C, l):
            reg.set_params(alpha=l)
            reg.fit(A, C)
            return reg.coef_
        lambda_mc, x_mc = reg_term_search(A, C, reg_method, lambda1=lambda1, lambda4=lambda4, tol=tol_search)
    else:
        lambda_mc = lambda_
        reg.fit(A, C)
        x_mc = reg.coef_
    return lambda_mc, np.transpose(x_mc)

@staticmethod
def lasso(A, C, lambda_=1., auto_search=True, lambda1=1e-4, lambda4=1., tol_search=1e-8,
          fit_intercept=True, normalize=False, precompute=False, copy_A=True, max_iter=1000, tol=0.0001,
          warm_start=False, positive=False, random_state=None, selection='random'):
    """
    Lasso Regression with automatic search for a good regularization term lambda
    x_lambda = arg min{||Ax-C||^2/(2*n_samples) + lambda*||x||_1} (4)
    `Scikit Learn Lasso Regression
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`
    Args:
        A: mxn matrix
        C: m vector
        lambda_ (float): regularization parameter used if auto_search = False
        auto_search (bool): if True then use reg_term_search to find a good regularization parameter
        lambda1 (float): left starting point for L-curve corner search
        lambda4 (float): right starting point for L-curve corner search
        tol_search (float): termination threshold for regularization parameter search
        fit_intercept (bool): if True calculate intercept
        normalize (bool): deprecated if fit_intercept=False, if True normalize A for regression
        precompute (bool or array-like): If True compute and use Gram matrix to speed up calculations.
                                         Gram matrix can also be given explicitly
        copy_A (bool): if True A is copied, else overwritten
        max_iter (int): max. number of iterations if solver is CG
        tol (float): precision of the regression solution
        warm_start (bool): if True reuse solution from previous fit as initialization
        positive (bool): if True force positive coefficients
        random_state (int): seed for the pseudo random number generator used when data is shuffled
        selection (str): {'cyclic', 'random'}
    Returns:
        array(int): Solution to the regularization inverse problem

    """
    from sklearn.linear_model import Lasso
    reg = Lasso(alpha=lambda_, fit_intercept=fit_intercept, normalize=normalize, precompute=precompute,
                copy_X=copy_A, max_iter=max_iter, tol=tol, warm_start=warm_start, positive=positive,
                random_state=random_state, selection=selection)
    if auto_search:
        def reg_method(A, C, l):
            reg.set_params(alpha=l)
            # reg.set_params({'alpha': l})
            reg.fit(A, C)
            return reg.coef_
        lambda_mc, x_mc = reg_term_search(A, C, reg_method, lambda1=lambda1, lambda4=lambda4, tol=tol_search)
    else:
        lambda_mc = lambda_
        reg.fit(A, C)
        x_mc = reg.coef_
    return lambda_mc, x_mc

@staticmethod
def regularized_lse_solver(A: np.ndarray,
                           C: np.ndarray,
                           regularization: str = 'perturb_diag',
                           lambda1: float = 1e-3,
                           lambda4: float = 1.,
                           alpha: float = 0.,
                           tol_norm_x: Union[tuple, float] = (1e-8, 5.),
                           tol_cond_A: float = 1000.) -> np.ndarray:
    """
    Solve a linear system of equations with a regularization method and automatic lambda fitting
    cite lambda fitting
    Args:
        A: mxn matrix
        C: m vector
        regularization: Regularization scheme to be used: 'ridge', 'lasso', 'perturb_diag_elements' or 'perturb_diag'
        lambda1: left starting point for L-curve corner search (for 'ridge' and 'lasso')
        lambda4: right starting point for L-curve corner search (for 'ridge' and 'lasso')
        alpha: perturbation coefficient for 'perturb_diag_elements' and 'perturb_diag'
        tol_norm_x: tolerance for the norm of x
        tol_cond_A: tolerance for the condition number of A

    Returns:

    """
    if regularization == 'ridge':
        _, x = ridge(A, C, lambda1=lambda1)
    elif regularization == 'lasso':
        _, x = lasso(A, C, lambda1=lambda1)
    elif regularization == 'perturb_diag_elements':
        alpha = 1e-7
        while np.linalg.cond(A + alpha * np.diag(A)) > tol_cond_A:
            alpha *= 10
        # include perturbation in A to avoid singularity
        x, res, rank, sv = np.linalg.lstsq(A + alpha * np.diag(A), C, rcond=None)
    else:
        alpha = 1e-7
        while np.linalg.cond(A + alpha * np.eye(len(C))) > tol_cond_A:
            alpha *= 10
        # include perturbation in A to avoid singularity
        x, _, _, _ = np.linalg.lstsq(A + alpha * np.eye(len(C)), C, rcond=None)

    if np.linalg.norm(x) > tol_norm_x[1] or np.linalg.norm(x) < tol_norm_x[0]:
        if regularization == 'ridge':
            lambda1 = lambda1 / 10.
            _, x = ridge(A, C, lambda1=lambda1, lambda4=lambda4)
        elif regularization == 'lasso':
            lambda1 = lambda1 / 10.
            _, x = lasso(A, C, lambda1=lambda1)
        elif regularization == 'perturb_diag_elements':
            while np.linalg.cond(A + alpha * np.diag(A)) > tol_cond_A:
                if alpha == 0:
                    alpha = 1e-7
                else:
                    alpha *= 10
            # include perturbation in A to avoid singularity
            x, _, _, _ = np.linalg.lstsq(A + alpha * np.diag(A), C, rcond=None)
        else:
            if alpha == 0:
                alpha = 1e-7
            else:
                alpha *= 10
            while np.linalg.cond(A + alpha * np.eye(len(C))) > tol_cond_A:
                # include perturbation in A to avoid singularity
                x, res, rank, sv = np.linalg.lstsq(A + alpha * np.eye(len(C)), C, rcond=None)
                alpha *= 10
    return x
