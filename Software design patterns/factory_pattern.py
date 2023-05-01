
from __future__ import annotations
from abc import ABC, abstractmethod

#@: NOTE : Software Design Patterns 
#@: NOTE : Pattern 3: Factory Pattern 

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
    The Factory Method pattern is a creational design pattern that provides an interface for creating objects in a superclass 
    but allows subclasses to alter the type of objects that will be created. The pattern is used to handle the process of object 
    creation and encapsulate the object instantiation logic from the client code. It promotes loose coupling and code reusability
    by delegating object creation to subclasses instead of directly instantiating objects in the client code.

    
    The rationale behind the Factory Method pattern is to achieve the following:

        *   Separate object creation from the client code: By separating the object creation logic from the client code, it becomes 
                                                           easier to maintain and modify the object creation process without affecting 
                                                           the client code.


        *   Promote code reusability: The Factory Method pattern allows creating reusable object factories that can be shared across 
                                      multiple modules or projects. This can reduce code duplication and make it easier to maintain 
                                      and extend your code.


        *   Encapsulate complex object creation logic: When objects have complex creation logic, it's helpful to encapsulate that logic 
                                                       in a factory method. This way, the client code doesn't need to know about the intricate 
                                                       details of object creation, making it simpler and more maintainable.


        *   Enable extensibility: Factory Method pattern promotes extensibility by allowing new types of objects to be added without changing 
                                  the client code. When a new type of object is introduced, you can create a new factory method for it and 
                                  easily plug it into the existing system.


        *   Achieve loose coupling: The pattern enables a loose coupling between the client code and the concrete classes that are instantiated. 
                                    This allows you to change the concrete classes being used without affecting the client code that relies on the 
                                    factory method.

'''



__doc_two__: str = r'''
    The factory pattern is a creational design pattern that provides an interface for creating objects, but allows subclasses to decide which 
    class to instantiate. It's useful in situations where you need to create objects without knowing the exact class of object that will be 
    created or when you want to delegate the responsibility of object creation to the subclasses.

    
    Here are some reasons why you might want to use the factory pattern:

        *   Encapsulation: By using a factory method to create objects, you can encapsulate the object creation logic in a separate class and 
                           abstract the client code from the implementation details of the object creation process. This allows for more flexible 
                           and extensible code that is easier to maintain and modify.


        *   Flexibility: By delegating object creation to subclasses, you can easily create new subclasses without changing the existing code. 
                         This allows for more flexibility in the design of your application and makes it easier to add new functionality or 
                         modify existing functionality.


        *   Abstraction: By using a factory method, you can abstract the client code from the implementation details of the object creation process. 
                         This allows you to create objects without having to know the details of how the objects are created, making your code more
                         modular and easier to maintain.


        *   Testability: By using a factory method, you can create mock objects for testing purposes. This makes it easier to test your code and 
                         ensures that your code is working correctly.


        *   Standardization: By using a factory method, you can ensure that all objects are created in a standardized way. This makes your code 
                             more consistent and easier to understand.


Overall, the factory pattern is a powerful tool for creating objects in a flexible and extensible way. It allows you to abstract the client code from 
the implementation details of the object creation process, making your code more modular, testable, and maintainable.

'''
#@: Example 1
'''
*   The Animal class is an abstract base class with an abstract method speak(), which is implemented by its subclasses Dog and Cat. The AnimalFactory 
    class has a static method create_animal() that takes an animal_type string and returns an instance of the corresponding subclass using a dictionary 
    lookup.

*   Each subclass has its own implementation of the speak() method, which returns a string representing the sound made by the animal. The AnimalFactory 
    class allows for easy creation of Dog or Cat objects based on the animal_type string passed to the create_animal() method.

*   Overall, this implementation follows the principles of the factory method pattern, which allows for more flexible and extensible code that is easier 
    to maintain and modify.
'''
class Animal(ABC):
    @abstractmethod
    def speak(self) -> None:
        ...
        
        
class Dog(Animal):
    def speak(self) -> str:
        return 'woof'
    

class Cat(Animal):
    def speak(self) -> str:
        return 'meow'
    

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        animal_map: dict[str, Animal] = {
            'dog': Dog(), 
            'cat': Cat()
        }
        return animal_map[animal_type]




#@: Example 2: 
'''
*   This code is an implementation of the factory method pattern in Python, which is used to create and customize different types of pizzas.

*   The Pizza class is an abstract base class that defines four abstract methods: prepare(), bake(), cut(), and box(). These methods will be 
    implemented by the subclasses CheesePizza and PepperoniPizza to provide the specific behavior of each type of pizza.

*   The CheesePizza and PepperoniPizza classes are subclasses of Pizza and provide their own implementations of the prepare(), bake(), cut(), 
    and box() methods. These methods print out messages indicating the steps involved in preparing, baking, cutting, and boxing the pizza.

*   The PizzaFactory class is responsible for creating instances of the appropriate Pizza subclass based on the pizza_type string passed to the 
    create_pizza() method. It uses a dictionary to map the pizza_type string to the appropriate subclass instance. If the pizza_type is not in 
    the dictionary, it raises a ValueError.

*   The PizzaStore class is responsible for taking orders for pizzas and using the PizzaFactory to create and customize the pizzas. The order_pizza() 
    method takes a pizza_type string, creates an instance of the appropriate Pizza subclass using the PizzaFactory, and then calls the prepare(), bake(),
    cut(), and box() methods to create and deliver the pizza. Finally, it prints a message indicating that the pizza is ready.


By using the factory method pattern, we can easily create and customize different types of pizzas without having to change the PizzaStore class or the 
implementation details of the subclasses. This allows for more flexible and extensible code that is easier to maintain and modify.
'''
class Pizza(ABC):
    @abstractmethod
    def prepare(self) -> None:
        ...
    
    @abstractmethod
    def bake(self) -> None:
        ...
    
    @abstractmethod
    def cut(self) -> None:
        ...
    
    @abstractmethod
    def box(self) -> None:
        ...
        
        
        
class CheesePizza(Pizza):
    def prepare(self) -> None:
        print("Preparing cheese pizza...")
        
    def bake(self) -> None:
        print("Baking cheese pizza...")
        
    def cut(self) -> None:
        print("Cutting cheese pizza...")
        
    def box(self) -> None:
        print("Boxing cheese pizza...")



class PepperoniPizza(Pizza):
    def prepare(self) -> None:
        print("Preparing pepperoni pizza...")
        
    def bake(self) -> None:
        print("Baking pepperoni pizza...")
        
    def cut(self) -> None:
        print("Cutting pepperoni pizza...")
        
    def box(self) -> None:
        print("Boxing pepperoni pizza...")




class PizzaFactory:
    @staticmethod
    def create_pizza(pizza_type: str) -> Pizza:
        pizza_map: dict[str, Pizza] = {
            "cheese": CheesePizza(),
            "pepperoni": PepperoniPizza()
        }
        if pizza_type not in pizza_map:
            raise ValueError(f"Invalid pizza type: {pizza_type}")
        return pizza_map[pizza_type]




class PizzaStore:
    def order_pizza(self, pizza_type: str) -> None:
        pizza = PizzaFactory.create_pizza(pizza_type)
        pizza.prepare()
        pizza.bake()
        pizza.cut()
        pizza.box()
        print(f"Here's your {pizza_type} pizza!")

        




#@: Driver Code 
if __name__.__contains__('__main__'):
    # Create an instance of Dog using the factory method
    animal_factory = AnimalFactory()
    dog = animal_factory.create_animal("dog")
    print(dog.speak())  # Output: Woof!

    # Create an instance of Cat using the factory method
    cat = animal_factory.create_animal("cat")
    print(cat.speak())  # Output: Meow!
    
    
    # Create a PizzaStore object and order a cheese pizza
    pizza_store = PizzaStore()
    pizza_store.order_pizza("cheese")

    # Create a PizzaStore object and order a pepperoni pizza
    pizza_store = PizzaStore()
    pizza_store.order_pizza("pepperoni")