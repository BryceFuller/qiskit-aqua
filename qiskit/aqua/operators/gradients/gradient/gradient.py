# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The base interface for Aqua's gradient."""



from functools import partial, reduce
from collections.abc import Iterable
from typing import Optional, Union, Tuple, List, Callable
from copy import deepcopy
import numpy as np
import sympy as sy
import jax.numpy as jnp
from jax import grad, jit, vmap

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression, Instruction
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from ..gradient_base import GradientBase
from ...operator_base import OperatorBase
from ...primitive_ops.primitive_op import PrimitiveOp
from ...primitive_ops.pauli_op import PauliOp
from ...primitive_ops.circuit_op import CircuitOp
from ...list_ops.list_op import ListOp
from ...list_ops.composed_op import ComposedOp
from ...list_ops.summed_op import SummedOp
from ...list_ops.tensored_op import TensoredOp
from ...state_fns.state_fn import StateFn
from ...state_fns.circuit_state_fn import CircuitStateFn
from ...operator_globals import H, S, I, Zero, One
from ...converters.converter_base import ConverterBase

class Gradient(GradientBase):
    r"""
    Converter for changing parameterized circuits into operators
    whose evaluation yields the first-order gradient with respect to the circuit parameters.
    """

    # pylint: disable=too-many-return-statements
    def convert(self,
        operator: OperatorBase = None,
        params: Optional[List] = None,
        method: str = 'param_shift',
        natural_gradient: bool = False) -> OperatorBase:

        r"""
        Args:
            operator: The measurement operator we are taking the gradient of
            params: The parameters we are taking the gradient with respect to
            method: The method used to compute the state/probability gradient. ['param_shift', 'ancilla']
                    Deprecated for observable gradient
        Returns:
            gradient_operator: An operator whose evaluation yields the Gradient
        """

        if isinstance(params, (ParameterVector, List)):
            param_grads = [self.convert(operator, param) for param in params]
            #If autograd returns None, then the corresponding parameter was probably not present in the operator. 
            # This needs to be looked at more carefully as other things can probably trigger a return of None. 
            absent_params = [params[i] for i,grad_ops in enumerate(param_grads) if grad_ops is None]
            if len(absent_params) > 0:
                raise ValueError("The following parameters do not appear in the provided operator: ", absent_params)
            return ListOp(param_grads)

        param = params

        """
        All of this logic is outside of autograd because autograd doesn't currently handle 
        the coefficient of the outermost operator. Currently autograd assumed that 
        all operator it sees have coeff == 1. It checks and throws an error if this is not true.
        The reason I haven't put this logic into autograd is because it will eliminate this assumption
        that all coeffs are 1 and I like having that check there to catch any weird behavior. I will probably
        move it inside before release though. 
        """
        # Separate the operator from the coefficient
        coeff = operator._coeff
        op = operator/coeff
        #Get derivative of the operator (recursively)
        d_op = self.autograd(op, param, method)
        if d_op is None:
            #I need this term to evaluate to 0, but it needs to be an OperatorBase type
            #We should find a more elegant solution for this.
            d_op = ~Zero@One
            
        grad_op = coeff*d_op
            
        if isinstance(coeff, ParameterExpression):
            #..get derivative of the coefficient
            d_coeff = self.parameter_expression_grad(coeff, param)
            grad_op += d_coeff*op
            
        return grad_op

    #NOTE, the coeff of the highest level operator is not handled by any of this code. 
    #Can assume only 1 parameter. If more are passed in we immediately reduce to the case of 1 at a time. 
    def autograd(self, operator, params, method='param_shift'):
        
        
        def is_coeff_one(coeff):  
            if isinstance(coeff, ParameterExpression):
                expr = coeff._symbol_expr
                return expr==1.0
            return coeff==1
        
        assert is_coeff_one(operator._coeff)

        if isinstance(params, (ParameterVector, List)):
            param_grads = [self.autograd(operator, param, method) for param in params]
            #If autograd returns None, then the corresponding parameter was probably not present in the operator. 
            # This needs to be looked at more carefully as other things can probably trigger a return of None. 
            absent_params = [params[i] for i,grad_ops in enumerate(param_grads) if grad_ops is None]
            if len(absent_params) > 0:
                raise ValueError("The following parameters do not appear in the provided operator: ", absent_params)
            return ListOp(param_grads)
        
        #by this point, it's only one parameter
        param = params
        
        #Base Case, you've hit a ComposedOp!
        #Prior to execution, the composite operator was standardized and coefficients were collected.
        #Any operator measurements were converted to Pauli-Z measurements and rotation circuits were applied. 
        #Additionally, all coefficients within ComposedOps were collected and moved out front.
        if isinstance(operator, ComposedOp):
            
            assert is_coeff_one(operator._coeff), "Operator pre-processing failed. Coefficients were not properly collected inside the ComposedOps"
            
            #Do some checks to make sure operator is sensible
            if isinstance(operator[-1], CircuitStateFn):
                #Do some checks and deicde how you're planning on taking the gradient. 
                #for now we do param shift    
                if method == 'param_shift':
                    return self.parameter_shift(operator, param)
            elif isinstance(operator[-1], (VectorStateFn, DictStateFn)):
                pass
                #Do LCU logic
            else:
                pass
                #@CHRISTA, here is where you'd check if you need to
                # decompose some operator into circuits or do 
                # something other than the parameter shift rule. 
            
        #This is the recursive case where the chain rule is handled
        elif isinstance(operator, ListOp):  
            ops = operator.oplist
            grad_ops = []        
            for oper in ops:
                #ListOp( {c(x_0)g(x_0), c(x_1)g(x_1)  }  )
                #For the ith operator in this ListOp...
                # ...get the coefficient
                coeff = oper._coeff
                # ...and separate the operator from the coefficient
                op = oper/coeff
                #..get derivative of the coefficient
                d_coeff = self.parameter_expression_grad(coeff, param)
                #get derivative of the operator
                #This will be a recursive call in practice.
                d_op = self.autograd(op, param, method)
                #Think harder about this
                if d_op is None:
                    #I need this term to evaluate to 0, but it needs to be an OperatorBase type
                    #We should find a more elegant solution for this.
                    d_op = ~Zero@One
                
                grad_op = d_coeff*op + coeff*d_op
                grad_ops.append(grad_op)
                
            #Note that this check to see if the ListOp has a default combo_fn
            # will fail if the user manually specifies the default combo_fn. 
            # I.e operator = ListOp([...], combo_fn=lambda x:x) will not pass this check and
            # later on jax will try to differentiate it and fail. 
            # An alternative is to check the byte code of the operator's combo_fn against the default one.
            # This will work but look very ugly and may have other downsides I'm not aware of
            if operator._combo_fn == ListOp([])._combo_fn:
                print("default combo fn")
                return ListOp(oplist=grad_ops)
            elif isinstance(operator, SummedOp):
                print("SummedOp combo fn")
                return SummedOp(oplist=grad_ops)
            elif isinstance(operator, TensoredOp):
                raise NotImplementedError
                #TODO!
                
            #NOTE! This will totally break if you try to pass a DictStateFn through a combo_fn
            # (for example, using probability gradients)
            # I think this is a problem more generally, not just in this subroutine. 
            grad_combo_fn = self.get_grad_combo_fn(operator)
            return ListOp(oplist=ops+grad_ops, combo_fn=grad_combo_fn)
            
        elif isinstance(operator, StateFn):
            if operator._is_measurement:
                raise Exception 
                #Doesn't make sense to have a StateFn measurement
                # at the end of the tree.
        else:
            print(type(operator))
            print(operator)
            raise Exception("Control Flow should never have reached this point")
            #This should never happen. The base case in our traversal is reaching a ComposedOp or a StateFn.
            #If a leaf of the computational graph is not one of these two, then the original operator
            # we are trying to differentiate is not an expectation value or a state. Thus it's not clear what
            # the user wants.
            
        
    def get_grad_combo_fn(self, operator: ListOp) -> Callable:
        """
        Get the derivative of the operator combo_fn
        Args:
            operator: The operator for whose combo_fn we want to get the gradient.
            
        Returns:
            function which evaluates the partial gradient of operator._combo_fn 
            with respect to each element of operator.oplist

        """
        if isinstance(operator, ComposedOp):
            raise Exception("FIGURE OUT HOW THIS CODE WAS REACHED")
        
        n = len(operator.oplist)
        indices = [i for i in range(n)]
        #jax needs the combo_fn to have n inputs, rather than a list of n inputs
        wrapped_combo_fn = lambda *x: operator._combo_fn(list(x))
        
        try:
            grad_combo_fn = jit(grad(wrapped_combo_fn, indices))
            #Test to see if the grad function breaks for a trivial input
            grad_combo_fn(*[0. for i in range(n)])
        except Exception as e:
            raise Exception("An error occurred when attempting to differentiate a combo_fn")
              
        #ops will be the concatenation of the original oplist with the gradients 
        # of each operator in the original oplist.
        #If your original ListOp contains k elements, then ops will have length 2k.
        def chain_rule_combo_fn(ops, grad_combo_fn):
            #Get the first half of the values in ops and convert them to floats (or jax breaks)
            opvals = [np.float(np.real(val)) for val in ops[:int(len(ops)/2)]]
            #Get the derivatives of each op in oplist w.r.t the relevant parameter
            derivs = [np.float(np.real(val)) for val in ops[int(len(ops)/2):]]
            #Get the partial derivatives of the combo_fn with respect to each op in oplist
            pderivs = [partial.tolist() for partial in grad_combo_fn(*opvals)]
            #return the dot product to compute the final derivative of the operator with 
            # respect to the specified parameter.
            return np.dot(pderivs, derivs)
        
        return partial(chain_rule_combo_fn, grad_combo_fn=grad_combo_fn)

    # TODO get ParameterExpression in the different gradients
    # Working title
    def _chain_rule_wrapper_sympy_grad(self,
                                  param: ParameterExpression) -> List[Union[sy.Expr, float]]:
        """
        Get the derivative of a parameter expression w.r.t. the underlying parameter keys
        :param param: Parameter Expression
        :return: List of derivatives of the parameter expression w.r.t. all keys
        """
        expr = param._symbol_expr
        keys = param._parameter_symbols.keys()
        grad = []
        for key in keys:
            grad.append(sy.Derivative(expr, key).doit())
        return grad

    def _get_gates_for_param(self,
                             param: ParameterExpression,
                             qc: QuantumCircuit) -> List[Instruction]:
        """
        Check if a parameter is used more often than once in a quantum circuit and return a list of quantum circuits
        which enable independent adding of pi/2 factors without affecting all gates in which the parameters is used.
        :param param:
        :param qc:
        :return:
        """
        # TODO check if param appears in multiple gates of the quantum circuit.
        # TODO deepcopy qc and replace the parameters by independent parameters such that they can be shifted
        # independently by pi/2
        return qc._parameter_table[param]
