from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type

#@: NOTE : Software Design Patterns 
#@: NOTE : Design Pattern 6: Bridge Pattern

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

__pattern_doc__: str = r'''
    The Bridge design pattern is a structural pattern that aims to decouple an abstraction from its implementation, 
    allowing the two to evolve independently. This pattern is especially useful when you need to create a bridge 
    between an abstract interface and its concrete implementations so that they can be developed and modified separately. 
    The Bridge pattern achieves this decoupling by using composition instead of inheritance.

    The Bridge pattern consists of the following components:

        *   Abstraction: This is an interface or abstract class that defines the high-level abstraction. It contains a 
                         reference to an object of the Implementation class and provides an interface for the client code. 
                         The Abstraction class delegates the actual work to the Implementation object it references.

        *   RefinedAbstraction: This class extends the Abstraction and provides more specific functionality. It can override 
                                or add new methods to the base Abstraction class. This class is also responsible for delegating 
                                the work to the concrete Implementation class it references.

        *   Implementation: This is an interface or abstract class that defines the low-level operations the concrete 
                            implementations need to perform. It provides the basic structure for concrete implementation classes.

        *   ConcreteImplementation: These are the concrete classes that implement the Implementation interface or inherit from 
                                    the abstract Implementation class. They provide the actual implementation of the low-level 
                                    operations defined in the Implementation class.


    The Bridge pattern provides several benefits:

        *   Decoupling of abstraction and implementation: The Bridge pattern allows the abstraction and implementation to evolve 
                                                          independently. This means that you can change or add new implementations 
                                                          without modifying the Abstraction, and you can modify the Abstraction 
                                                          without affecting the existing implementations.

        *   Improved code organization: The Bridge pattern promotes better organization of code by separating the high-level 
                                        abstraction from the low-level implementation details. This makes the code easier to 
                                        understand, maintain, and extend.

        *   Reusability: The Bridge pattern enables you to reuse the same abstraction with different implementations and vice versa. 
                         This reduces code duplication and promotes code reusability.

        *   Flexibility: The Bridge pattern allows you to switch between different implementations at runtime. This is particularly 
                         useful when you need to support multiple platforms or configurations.


    the Bridge design pattern is a structural pattern that separates an abstraction from its implementation, allowing them to evolve 
    independently. It achieves this by using composition instead of inheritance, promoting decoupling, improved code organization, 
    reusability, and flexibility.
'''


#@: NOTE: Example 1: Basic Bridge Pattern

# In this example, we're implementing a basic Bridge pattern using Python. The key components of this pattern are the Abstraction and 
# Implementation classes, which are separated to allow them to evolve independently. Let's break down the code:

#         *   Abstraction class: This is an abstract class that represents the high-level abstraction in the Bridge pattern. It has a 
#                                constructor that takes an implementation parameter of type Callable[Any]. The operation method calls the 
#                                operation_implementation method on the provided implementation object.

        
#         *   Implementation class: This is another abstract class representing the low-level implementation in the Bridge pattern. It has a 
#                                   single abstract method operation_implementation, which needs to be implemented by the concrete implementation 
#                                   classes.
                                  
                                  
#         *   ConcreteImplementationA and ConcreteImplementationB classes: These are the concrete classes that implement the Implementation interface. 
#                                                                          They provide the actual implementation of the operation_implementation method. 
#                                                                          In this case, they return a simple string to indicate which implementation is 
#                                                                          being used.

# The main idea behind the Bridge pattern is to separate the high-level abstraction (in this case, Abstraction) from the low-level implementation details 
# (in this case, ConcreteImplementationA and ConcreteImplementationB). This allows you to change or add new implementations without modifying the 
# Abstraction, and vice versa.

class Abstraction(ABC):
    def __init__(self, implementation: Callable[Any]) -> None:
        self._implementation = implementation


    def operation(self) -> str:
        return self._implementation.operation_implementation()



class Implementation(ABC):
    @abstractmethod
    def operation_implementation(self) -> str:
        ...


class ConcreteImplementationA(Implementation):
    def operation_implementation(self) -> str:
        return "ConcreteImplementationA"



class ConcreteImplementationB(Implementation):
    def operation_implementation(self) -> str:
        return "ConcreteImplementationB"
    
    

#@: NOTE: Example 2: Bridge pattern with a refined abstraction:

# In this example, we extend the basic Bridge pattern with a "refined abstraction". A refined abstraction is a class that extends the base Abstraction 
# and provides additional or more specific functionality. It can override or add new methods to the base Abstraction class.

#         *   RefinedAbstraction class: This class extends the Abstraction class and overrides the operation method. The new implementation of the 
#                                       operation method in RefinedAbstraction adds a custom string prefix "Refined: " to the result of the 
#                                       operation_implementation method called on the implementation object. This demonstrates how a refined 
#                                       abstraction can provide additional functionality without affecting the base abstraction or the implementation 
#                                       classes.

# In this example, the RefinedAbstraction class can be used with both ConcreteImplementationA and ConcreteImplementationB. When the operation method 
# is called on a RefinedAbstraction object, it will return a string that includes the "Refined: " prefix, followed by the result of the 
# operation_implementation method of the implementation object.


# By introducing a refined abstraction, you can provide more specific functionality or variations of the abstraction without modifying the base 
# abstraction or implementation classes. This helps to maintain the separation of concerns and allows the abstraction and implementation to evolve 
# independently.

class RefinedAbstraction(Abstraction):
    def operation(self) -> str:
        return f'Refined: {self._implementation.operation_implementation()}'



#@: NOTE: Example 3: Bridge pattern with multiple methods in the abstraction and implementation:

# In this example, we further extend the Bridge pattern by adding multiple methods to both the abstraction and the implementation classes. This 
# demonstrates how the Bridge pattern can be adapted to more complex scenarios with additional functionality.

#         *   AdvancedAbstraction class: This class extends the base Abstraction class and adds a new method called additional_operation. 
#                                        This method calls the additional_operation_implementation method on the implementation object. 
#                                        This demonstrates how a more advanced abstraction can work with an extended implementation interface.


#         *   AdvancedImplementation class: This class extends the base Implementation class and introduces a new abstract method called 
#                                           additional_operation_implementation. Concrete implementations of the AdvancedImplementation 
#                                           class should provide an implementation for this new method.


#         *   AdvancedConcreteImplementationA class: This class is a concrete implementation of the AdvancedImplementation interface. It 
#                                                    provides the actual implementations for both the operation_implementation and 
#                                                    additional_operation_implementation methods. In this example, the methods return 
#                                                    simple strings to indicate which method and implementation are being used.


# This example demonstrates how the Bridge pattern can be adapted to more complex scenarios, where both the abstraction and implementation 
# classes have multiple methods. By extending the base abstraction and implementation classes, you can maintain the separation of concerns 
# and allow the abstraction and implementation to evolve independently.

class AdvancedAbstraction(Abstraction):
    def additional_operation(self) -> str:
        return self._implementation.additional_operation_implementation()


class AdvancedImplementation(Implementation):
    @abstractmethod
    def additional_operation_implementation(self) -> str:
        ...
        

class AdvancedConcreteImplementationA(AdvancedImplementation):
    def operation_implementation(self) -> str:
        return "AdvancedConcreteImplementationA: operation"


    def additional_operation_implementation(self) -> str:
        return "AdvancedConcreteImplementationA: additional_operation"
    

    

#@: NOTE: Example 4: Bridge pattern with a shared implementation for multiple abstractions:

# In this example, we demonstrate how the Bridge pattern allows multiple abstractions to share a common implementation. The idea is that different 
# abstractions can utilize the same implementation without modifying the implementation or other abstractions.

#         *   AnotherAbstraction class: This class extends the base Abstraction class and introduces a new method called another_operation. 
#                                       This method returns a string that includes the prefix "Another: ", followed by the result of the 
#                                       operation_implementation method of the implementation object. This demonstrates that a completely 
#                                       different abstraction can still work with the same implementation classes as the original Abstraction class.


# In this example, the AnotherAbstraction class can be used with both ConcreteImplementationA and ConcreteImplementationB. When the another_operation 
# method is called on an AnotherAbstraction object, it will return a string that includes the "Another: " prefix, followed by the result of the 
# operation_implementation method of the implementation object.


# This demonstrates a key benefit of the Bridge pattern: multiple abstractions can share a common implementation, and the implementation can be changed 
# or extended without affecting the abstractions. This helps maintain the separation of concerns and allows the abstractions and implementation to evolve 
# independently.

class AnotherAbstraction(Abstraction):
    def another_operation(self) -> str:
        return f'Another: {self._implementation.operation_implementation()}'
    
    

#@: NOTE: Example 5: Bridge pattern with a factory for creating abstractions:


# In this example, we introduce an AbstractionFactory to create abstractions with a specified implementation. The factory pattern is used to encapsulate the 
# creation of objects, making it easier to manage the process and reduce complexity in the client code.

#         *   AbstractionFactory class: This class provides a static method create_abstraction that takes two arguments - abstraction_type and implementation. 
#                                       The abstraction_type is the type of abstraction you want to create (a subclass of Abstraction), and implementation is 
#                                       the specific implementation object you want to associate with the abstraction. The method returns a new instance of 
#                                       the specified abstraction type with the given implementation.


# Using the AbstractionFactory, you can create instances of different abstractions (like Abstraction, RefinedAbstraction, AdvancedAbstraction, or AnotherAbstraction)
# with various implementations (like ConcreteImplementationA, ConcreteImplementationB, or AdvancedConcreteImplementationA) in a more streamlined and organized way. 
# This can make the client code cleaner and easier to maintain.


# The AbstractionFactory demonstrates how the Bridge pattern can be combined with other design patterns (in this case, the Factory pattern) to create more 
# organized and maintainable code.


class AbstractionFactory:
    @staticmethod
    def create_abstraction(abstraction_type: Type[Abstraction], implementation: Implementation) -> Abstraction:
        return abstraction_type(implementation)
    
    

#@: Driver Code
if __name__.__contains__('__main__'):
    #@: example 1:
    implementation_a = ConcreteImplementationA()
    abstraction_a = Abstraction(implementation_a)
    print(abstraction_a.operation())  # Output: ConcreteImplementationA

    implementation_b = ConcreteImplementationB()
    abstraction_b = Abstraction(implementation_b)
    print(abstraction_b.operation())  # Output: ConcreteImplementationB
    
    #@: Example 2:
    refined_abstraction_a = RefinedAbstraction(implementation_a)
    print(refined_abstraction_a.operation())  # Output: Refined: ConcreteImplementationA
    
    #@: Example 3:
    advanced_implementation_a = AdvancedConcreteImplementationA()
    advanced_abstraction_a = AdvancedAbstraction(advanced_implementation_a)
    print(advanced_abstraction_a.operation())  # Output: AdvancedConcreteImplementationA: operation
    print(advanced_abstraction_a.additional_operation())  # Output: AdvancedConcreteImplementationA: additional_operation
    
    #@: Example 4:
    another_abstraction_a = AnotherAbstraction(implementation_a)
    print(another_abstraction_a.another_operation())  # Output: Another: ConcreteImplementationA
    
    
    #@: Example 5:
    factory_abstraction_a = AbstractionFactory.create_abstraction(Abstraction, implementation_a)
    print(factory_abstraction_a.operation())  # Output: ConcreteImplementationA

    factory_refined_abstraction_a = AbstractionFactory.create_abstraction(RefinedAbstraction, implementation_a)
    print(factory_refined_abstraction_a.operation())  # Output: Refined: ConcreteImplementationA

    factory_advanced_abstraction_a = AbstractionFactory.create_abstraction(AdvancedAbstraction, advanced_implementation_a)
    print(factory_advanced_abstraction_a.operation())  # Output: AdvancedConcreteImplementationA: operation
    print(factory_advanced_abstraction_a.additional_operation())  # Output: AdvancedConcreteImplementationA: additional_operation
    