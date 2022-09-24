from __future__ import annotations
import torch, math, copy
from typing import Optional, Any, Callable, Union, Iterable

__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Education': 'Amity University, Noida : Btech CSE (Final Year)',
    'Mail': [
        'sawhney.rahulofficial@outlook.com', 
        'rahulsawhney321@gmail.com'
    ]
}

#@: date = 25 September, 2022
__license__: str = r'''
    MIT License
    Copyright (c) 2022 Rahul Sawhney
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

class Optimizer_Interface(torch.optim.Optimizer):
    def __init__(self, parameters: Iterable[Any], learning_rate: Optional[Union[float, NoneType]] = None, 
                                                  beta: Optional[Union[int, float]] = 1,
                                                  lips_contant: Optional[Union[int, float]] = 1) -> None:
        
        if beta < 0.0 or lips_contant < 0.0:
            raise ValueError
        
        self.default_params: dict[str, Any] = {
            key : value for key, value in locals().items() if not key in [
                'self', 'parameters'
            ]
        }
        super(Optimizer_Interface, self).__init__(parameters, default_params)
        
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            key : value for key, value in zip(
                ['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))]
            )
        })
    
    
    def __str__(self) -> str(dict[str, Any]):
        return str(self.default_params)
   
   
    def step(self, *args: Iterable[Any], **kwargs: Mapping[Any, Any]) -> NotImplementedError:
        raise NotImplementedError
   
   



class AdaptiveAcceleratedSGD(Optimizer_Interface):
    def __init__(self, parameters: Iterable[Any], learning_rate: Optional[Union[float, NoneType]] = None,
                                                  beta: Optional[Union[int, float]] = 10,
                                                  lips_contant: Optional[Union[int, float]] = 10) -> None:
        
        if beta < 0.0 or lips_contant < 0.0:
            raise ValueError
        
        self.default_params: dict[str, Any] = {
            key : value for key, value in locals().items() if not key in [
                'self', 'parameters'
            ]
        }
        
        super(AdaptiveAcceleratedSGD, self).__init__(parameters, self.default_params)
        
    
    
    def step(self, closure: Optional[Callable[Any], float]) -> Union[int, float, NoneType]:
        loss: Union[int, float, NoneType] = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad: Any = p.grad.data
                state: Any = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['alpha_k'] = 1
                    state['v_k'] = 0
                    state['avg_grad'] = copy.deepcopy(grad)
                    state['x_k'] = copy.deepcopy(p.data)
                    
                gamma_k: Union[int, float, Any] = 2 * group['lips_constant'] / (state['step'] + 1)
                
                avg_grad: Union[int, float, Any] = state['avg_grad']
                avg_grad.mul_(state['step'])
                avg_grad.add_(grad)
                avg_grad.div_(state['step'] + 1)
                
                delta_k: Union[int, float, Any] = torch.add(grad, avg_grad, alpha = -1)
                state['v_k'] += torch.sum(delta_k * delta_k).item()
                
                h_k: Union[int, float, Any] = math.sqrt(state['v_k'])
                alpha_k_1: Union[int, float, Any] = 2 / (state['step'] + 3)
                coef: Union[int, float, Any] = 1 / (gamma_k + group['beta'] * h_k)
                x_k_1: Union[int, float, Any] = state['x_k']
                x_k_1.add_(grad, alpha = -coef)
                
                p.data.mul_(1 - alpha_k_1)
                p.data.add_(x_k_1, alpha = alpha_k_1)
                p.data.add_(grad, alpha = - (1 - alpha_k_1) * state['alpha_k'] * coef)
                
                state['alpha_k'] = alpha_k_1
                state['step'] += 1
                
        return loss
                

