from __future__ import annotations
from abc import ABC, abstractmethod

#@: NOTE : Software Design Patterns 
#@: NOTE : Design Pattern 8: Template Method Pattern

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
    The Template Method pattern is a behavioral design pattern that defines the basic structure of an algorithm in a base class 
    (also called a template class or an abstract class) and lets derived classes (also called concrete classes or subclasses) implement 
    the details of the algorithm without changing the overall structure.

    
    The Template Method pattern involves the following components:


            *   Abstract Base Class (Template Class): This class contains an abstract method called the "template method," which defines 
                                                      the high-level steps of an algorithm. The template method calls several other methods, 
                                                      some of which may be abstract (also known as "primitive methods" or "hook methods") and 
                                                      need to be implemented by subclasses. The template method can also contain default 
                                                      implementations for some methods, allowing subclasses to optionally override them.


            *   Concrete Classes (Subclasses): These classes inherit from the abstract base class and provide implementations for the abstract 
                                               methods defined in the base class. Each concrete class represents a different variation of the 
                                               algorithm, while the structure of the algorithm remains the same.


    The Template Method pattern allows you to:

            *   Define the skeleton of an algorithm in the base class and let subclasses provide the specific implementations for some steps.

            *   Enforce a consistent structure for the algorithm across all variations.

            *   Promote code reuse by defining common steps in the base class, reducing code duplication in the concrete classes.

            *   Provide extension points in the algorithm, allowing subclasses to customize certain aspects without affecting the overall structure.

    
    In summary, the Template Method pattern is useful when you have an algorithm with a fixed structure and multiple variations. It enables you to 
    define the common steps of the algorithm in a base class while letting subclasses implement the variable parts, ensuring a consistent structure 
    and promoting code reuse.
'''

#@: NOTE: Type 1: Basic Template Method Pattern:

# *   This code demonstrates the implementation of the Template Method pattern using inheritance and polymorphism in Python.


# *   The AbstractClass defines an abstract base class with a template_method() that calls two abstract methods step1() and step2(). The template_method() 
#     method defines the basic structure of the algorithm but leaves the implementation details of the step1() and step2() methods to the concrete subclasses.


# *   Two concrete classes ConcreteClass1 and ConcreteClass2 inherit from AbstractClass and provide their own implementation for the step1() and step2() methods. 
#     These concrete classes represent different variations of the algorithm that can be used in different situations.


# *   ConcreteClass1 provides a step1() method that returns the string "ConcreteClass1: Step 1" and a step2() method that returns the string "ConcreteClass1: Step 2".


# *   ConcreteClass2 provides a step1() method that returns the string "ConcreteClass2: Step 1" and a step2() method that returns the string "ConcreteClass2: Step 2".


# *   When the template_method() is called on an instance of either ConcreteClass1 or ConcreteClass2, it executes the algorithm defined in the AbstractClass by calling 
#     step1() and step2() in the appropriate order. The results of the step1() and step2() methods are concatenated and returned as a single string.

class AbstractClass(ABC):
    def template_method(self) -> str:
        result: str = self.step1() + self.step2()
        return result


    @abstractmethod
    def step1(self) -> str:
        ...

    @abstractmethod
    def step2(self) -> str:
        ...
        


class ConcreteClass1(AbstractClass):
    def step1(self) -> str:
        return "ConcreteClass1: Step 1\n"


    def step2(self) -> str:
        return "ConcreteClass1: Step 2\n"



class ConcreteClass2(AbstractClass):
    def step1(self) -> str:
        return "ConcreteClass2: Step 1\n"


    def step2(self) -> str:
        return "ConcreteClass2: Step 2\n"



#@: NOTE: Type 2: Template Method with Default Behavior:

# *   This code shows an implementation of the Template Method pattern that includes a default behavior for one of the abstract methods.


# *   The AbstractClassWithDefault is an abstract base class that defines a template_method() which calls two methods: step1() and step2(). The step1() method has a 
#     default implementation, whereas the step2() method is left for the concrete subclass to define.


# *   ConcreteClassWithDefault is a concrete subclass that inherits from AbstractClassWithDefault and provides its implementation of the step2() method.


# *   When template_method() is called on an instance of ConcreteClassWithDefault, it first executes the default implementation of step1(), followed by the implementation 
#     of step2() in ConcreteClassWithDefault. The results of step1() and step2() are concatenated and returned as a single string.


# *   This type of Template Method pattern provides a default implementation for some of the methods, which can be overridden by the concrete subclasses. It is useful 
#     when some parts of the algorithm are common across different implementations, while other parts need to be specialized.


class AbstractClassWithDefault(ABC):
    def template_method(self) -> str:
        result: str = self.step1() + self.step2()
        return result


    def step1(self) -> str:
        return "AbstractClassWithDefault: Default Step 1\n"


    @abstractmethod
    def step2(self) -> str:
        ...



class ConcreteClassWithDefault(AbstractClassWithDefault):
    def step2(self) -> str:
        return "ConcreteClassWithDefault: Step 2\n"
    
    


#@: NOTE: Type 3: Template Method with Hook Methods:

# *   This code demonstrates a Template Method pattern with hook methods. Hook methods are optional methods in the Template Method pattern that provide a way to modify 
#     or extend the behavior of the algorithm without changing its structure.


# *   AbstractClassWithHook is an abstract base class that defines a template_method() which calls three methods: step1(), step2(), and hook(). step1() and step2() are 
#     abstract methods, whereas hook() has a default implementation.


# *   ConcreteClassWithHook is a concrete subclass that inherits from AbstractClassWithHook and provides its implementation of the step1() and step2() methods. It also 
#     overrides the hook() method to provide custom behavior.


# *   When template_method() is called on an instance of ConcreteClassWithHook, it executes the algorithm defined in AbstractClassWithHook by calling step1(), step2(), 
#     and hook() in the appropriate order. The results of step1(), step2(), and hook() are concatenated and returned as a single string.


# *   The hook method hook() in AbstractClassWithHook has a default implementation that can be overridden by concrete subclasses. In this way, concrete subclasses can 
#     extend or modify the behavior of the algorithm without changing its structure.

class AbstractClassWithHook(ABC):
    def template_method(self) -> str:
        result: str = self.step1() + self.step2() + self.hook()
        return result


    @abstractmethod
    def step1(self) -> str:
        ...


    @abstractmethod
    def step2(self) -> str:
        ...


    def hook(self) -> str:
        return "AbstractClassWithHook: Default Hook\n"



class ConcreteClassWithHook(AbstractClassWithHook):
    def step1(self) -> str:
        return "ConcreteClassWithHook: Step 1\n"


    def step2(self) -> str:
        return "ConcreteClassWithHook: Step 2\n"


    def hook(self) -> str:
        return "ConcreteClassWithHook: Custom Hook\n"
    



#@: NOTE: Type 4: Template Method with Template State:

# *   This code demonstrates a Template Method pattern with template state. In this type of Template Method pattern, the algorithm can change its behavior based on the 
#     state of the object.


# *   AbstractClassWithState is an abstract base class that defines a template_method() which calls three methods: step1(), step2(), and state. step1() and step2() 
#     are abstract methods, whereas set_state() sets the state of the object.


# *   ConcreteClassWithState is a concrete subclass that inherits from AbstractClassWithState and provides its implementation of the step1() and step2() methods.


# *   When template_method() is called on an instance of ConcreteClassWithState, it executes the algorithm defined in AbstractClassWithState by calling step1(), 
#     step2(), and state() in the appropriate order. The results of step1(), step2(), and state() are concatenated and returned as a single string.


# *   ConcreteClassWithState can change the behavior of the algorithm by calling set_state() and setting the state of the object. The state can be used to modify the 
#     behavior of the algorithm in template_method(), for example, by changing the order in which methods are called or by modifying the output of a method based on the state.


# *   This type of Template Method pattern is useful when the algorithm needs to adapt its behavior based on the state of the object, and the state can be changed 
#     dynamically during runtime.


class AbstractClassWithState(ABC):
    def __init__(self) -> None:
        self.state: str = ''


    def template_method(self) -> str:
        result: str = self.step1() + self.step2() + self.state
        return result


    @abstractmethod
    def step1(self) -> str:
        ...


    @abstractmethod
    def step2(self) -> str:
        ...


    def set_state(self, state: str) -> None:
        self.state = state



class ConcreteClassWithState(AbstractClassWithState):
    def step1(self) -> str:
        return "ConcreteClassWithState: Step 1\n"


    def step2(self) -> str:
        return "ConcreteClassWithState: Step 2\n"
    
    

#@: NOTE: Type 5: Template Method with Multiple Template Classes:

# *   This code demonstrates a Template Method pattern with multiple template classes. In this type of Template Method pattern, multiple abstract base classes can be used 
#     together to create a more complex algorithm.


# *   AbstractClassA is an abstract base class that defines a template_method() which calls two methods: step1() and step2(). step1() and step2() are abstract methods.


# *   ConcreteClassA is a concrete subclass that inherits from AbstractClassA and provides its implementation of the step1() method.


# *   When template_method() is called on an instance of ConcreteClassA, it executes the algorithm defined in AbstractClassA by calling step1() and step2() in the appropriate 
#     order. The results of step1() and step2() are concatenated and returned as a single string.


# *   This is a simple example of a Template Method pattern with multiple template classes, as only one subclass is used. However, more complex algorithms can be created by 
#     combining multiple abstract base classes in a single algorithm.

class AbstractClassA(ABC):
    def template_method(self) -> str:
        result: str = self.step1() + self.step2()
        return result


    @abstractmethod
    def step1(self) -> str:
        ...


    @abstractmethod
    def step2(self) -> str:
        ...


class ConcreteClassA(AbstractClassA):
    def step1(self) -> str:
        return 'ConcreteClassA: Step 1\n'
    
    
    
    
    
#@: Driver Code
if __name__.__contains__('__main__'):
    #@: Ex: 1
    concrete_class_1 = ConcreteClass1()
    print(concrete_class_1.template_method())  # Output: ConcreteClass1: Step 1\nConcreteClass1: Step 2\n

    concrete_class_2 = ConcreteClass2()
    print(concrete_class_2.template_method())  # Output: ConcreteClass2: Step 1\nConcreteClass2: Step 2\n
    
    #@: Ex: 2
    concrete_class_with_default = ConcreteClassWithDefault()
    print(concrete_class_with_default.template_method())  # Output: AbstractClassWithDefault: Default Step 1\nConcreteClassWithDefault: Step 2\n
    
    #@: Ex: 3
    concrete_class_with_hook = ConcreteClassWithHook()
    print(concrete_class_with_hook.template_method())  # Output: ConcreteClassWithHook: Step 1\nConcreteClassWithHook: Step 2\nConcreteClassWithHook: Custom Hook\n
    
    
    #@: etc ....