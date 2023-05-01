
from __future__ import annotations
import time 

#@: NOTE : Software Design Patterns 
#@: NOTE : Pattern 2: Decorator Pattern 

__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Mail': [
        'sawhney.rahulofficial@outlook.com', 
        'rahulsawhney321@gmail.com'
    ]
}

__license__: str = r'''
    MIT License
    Copyright (c) 2023 Rahul Sawhney
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


__doc__: str = r'''
    The Decorator Pattern is a structural design pattern that involves a set of decorator classes that are used to wrap 
    concrete components. Decorator classes mirror the type of the components they decorate (they have the same interface) 
    but add or override behavior. In this way, the system can be dynamically composed at runtime with any number of decorators 
    in any order, which means that functionality can be composed in any combination required.

    The main purpose of the decorator pattern is to attach additional responsibilities to an object dynamically. Decorators 
    provide a flexible alternative to subclassing for extending functionality. Instead of static inheritance, where an object 
    gets its behavior up-front from a class, decorators provide a way to add behavior dynamically, a slice at a time.
    

    Here's how it works:

        *   Both the main object and the decorators implement a common interface.

        *   The main object is wrapped with the decorators.

        *   When a method of the interface is called, it is forwarded by the decorators to the inner object, possibly after 
            adding some functionality. Since the decorators also implement the same interface, a decorator can wrap another 
            decorator, which wraps another decorator, and so on. The innermost decorator wraps the main object.

    
    This pattern is used when you need to be able to assign extra behaviors to objects at runtime without breaking the code that 
    uses these objects. The additional behavior can be removed just as easily as it was added. This is a good choice when:


        *   You need to add responsibilities to individual objects dynamically and transparently, that is, without affecting 
            other objects.

        *   You need to add responsibilities to objects without creating a highly coupled system with many subclasses to cover 
            all possible combinations.


    In Python, decorators are a very powerful and expressive feature. Python decorators are not exactly the same as the decorator 
    pattern, but they achieve similar goals, and they use the "@" symbol in the code.
'''


'''
    >>>  theory behind '@' and __call__ in Python
    
    The @ Symbol:
        *   The @ symbol in Python is syntactic sugar for function decorators. A decorator is a higher-order function that takes a 
            function as input and returns a new function with (usually) extended behavior. The @ symbol makes it easy to apply a 
            decorator to a function.

    The __call__ Method:
        *   The __call__ method in Python is a special method that allows a class's instance to be called as a function, not a method. 
            When an instance of the class is called as a function, the __call__ method is invoked.

        *   This makes it possible to create classes where the instances behave like functions and can be called like a function.

        *   In the context of decorators, the __call__ method is used to implement the behavior that should be added to the decorated function. 
            When you apply a decorator to a function, Python creates an instance of the decorator class, passing the function. When you later 
            call the decorated function, Python calls the __call__ method of the decorator instance.


    Together, the @ symbol and the __call__ method provide a powerful way to modify a function's behavior without changing its source code. 
    This is useful in a variety of scenarios, such as adding logging or timing code, modifying input or output values, and more.
'''


#@: Printer Logger Decorator 

# PrinterLogger class is an example of a decorator class in Python. It is initialized with a function and then, through 
# the __call__ method, it prints information about the function call, calls the function itself, prints the result, and 
# returns the result.

class PrinterLogger:    
    def __init__(self, func: Callable[Any]) -> None:
        self.func = func
        
    
    def __call__(self, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
        print(f'Calling function {self.func.__name__}')
        print(f'With arguments {args} and {kwargs}')
        result: Any = self.func(*args, **kwargs)
        print(f'Function {self.func.__name__} returned {result}')
        return result


# This example shows how decorators can be used to add extra functionality (in this case, logging) to a function 
# without modifying the function itself.

@PrinterLogger
def add(a: int, b: int) -> int:
    # When you run this code, add(1, 2) will print something like:
    # Calling function add
    # With arguments (1, 2) and {}
    # Function add returned 3
    # 3
    return a + b



class Milk:
    def __init__(self, beverage: Callable[Any, float]) -> None:
        self.beverage = beverage


    def __call__(self) -> float:
        return self.beverage() + 0.2



def coffee() -> float:
    return 1.0



@Milk
def coffee() -> float:
    # Output: 1.2
    return 1.0



class Sugar:
    def __init__(self, beverage: Callable[Any, float]) -> None:
        self.beverage = beverage


    def __call__(self) -> float:
        return self.beverage() + 0.1



@Sugar
@Milk
def coffee() -> float:
    # Output: 1.3
    return 1.0




class LoggingDecorator:
    def __init__(self, func: Callable[Any]) -> None:
        self.func = func


    def __call__(self, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
        print(f"Calling function {self.func.__name__}")
        print(f"With arguments {args} and {kwargs}")
        
        start_time: float = time.time()
        result: Any = self.func(*args, **kwargs)
        end_time: float = time.time()
        
        print(f"Function {self.func.__name__} returned {result}")
        print(f"Execution time: {end_time - start_time:.6f} seconds")
        
        return result




@LoggingDecorator
def add_two(a: int, b: int) -> int:
    """Add two integers."""
    time.sleep(2)  # Let's simulate a time-consuming operation
    return a + b





class LoggingDecoratorTwo:
    
    # The class LoggingDecoratorTwo is a Python decorator class that has been extended to include several 
    # additional functionalities.

    #     *   Function Call Counting: The decorator keeps a count of the number of times the decorated function has been called. 
    #                                 This is done using a class variable call_count. Each time the __call__ method is invoked 
    #                                 (which happens each time the decorated function is called), call_count is incremented.


    #     *   Function Call Timing: The decorator also measures the time it takes to execute the decorated function. It records 
    #                               the time immediately before and after the function call, and then prints the difference, 
    #                               which is the execution time.


    #     *   Result Alteration: After the function has been called, the decorator alters the result by squaring it, which is done 
    #                            in the alter_result method.


    #     *   Call Count Retrieval: The decorator provides a class method get_call_count that returns the number of times the decorated 
    #                               function has been called. As a class method, it's not tied to any specific instance of the decorator, 
    #                               but instead operates on the class as a whole.
    
    call_count: int = 0

    def __init__(self, func: Callable[Any]) -> None:
        self.func = func



    def __call__(self, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
        print(f"Calling function {self.func.__name__}")
        print(f"With arguments {args} and {kwargs}")

        start_time: float = time.time()
        result: Any = self.func(*args, **kwargs)
        end_time: float = time.time()

        LoggingDecoratorTwo.call_count += 1

        print(f"Function {self.func.__name__} returned {result}")
        print(f"Execution time: {end_time - start_time:.6f} seconds")

        result: int | float = self.alter_result(result)
        return result



    @classmethod
    def get_call_count(cls) -> int:
        return cls.call_count



    def alter_result(self, result: int | float) -> int | float:
        return result ** 2
    
    

@LoggingDecoratorTwo
def add_two(a: int, b: int) -> int:
    # Calling function add_two    
    # With arguments (3, 5) and {}
    # Function add_two returned 8
    # Execution time: 1.012020 seconds
    # 64
    time.sleep(1)  # Let's simulate a time-consuming operation
    return a + b





#@: Driver Code 
if __name__.__contains__('__main__'):
    print(add(1, 3))
    print(add_two(3, 5))
    print(f'Total function calls: {LoggingDecoratorTwo.get_call_count()}')