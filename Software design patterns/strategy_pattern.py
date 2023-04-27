

from __future__ import annotations
from abc import ABC, abstractmethod

#@: NOTE : Software Design Patterns 
#@: NOTE : Pattern 1: Strategy Pattern

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


#@: NOTE : Pattern 1: Strategy Pattern

__doc__: str = r'''
    The Strategy pattern is a behavioral design pattern that lets you define a family of algorithms, put each of them into 
    a separate class, and make their objects interchangeable. It is also known as the Policy pattern.

    The main goal of the Strategy pattern is to enable a client to choose an algorithm from a family of algorithms at runtime and 
    provide a mechanism to swap out the 'strategy' or algorithm being used as business logic changes.

    Here are the key components of the Strategy pattern:

            *   Strategy (Interface): This is an interface that is common to all supported algorithms. Context uses this interface 
                                      to call the algorithm defined by a concrete strategy.


            *   Concrete Strategy: Implements the algorithm using the Strategy interface. There can be multiple concrete strategies 
                                   implementing the Strategy interface, each providing a different algorithm.


            *   Context: Holds a reference to a Strategy object, and is instantiated with a concrete strategy. It allows the interface 
                         to be interchangeable at runtime.


    The Strategy pattern encapsulates the implementation details of different algorithms, which are often subject to changes. This reduces 
    the coupling between the algorithm and the client code that uses it.

    Here are the advantages of using the Strategy pattern:


            *   Flexibility: The Strategy pattern provides an alternative to subclassing. Instead of implementing a behavior in subclasses, 
                             the behavior can be encapsulated using the Strategy pattern.


            *   Avoids conditional statements: The Strategy pattern provides an alternative to using conditional statements for selecting 
                                               desired behavior.


            *   Open/Closed Principle: The strategy pattern allows us to introduce new strategies without having to change the context. 
                                       This makes the strategy pattern compliant with the open/closed principle.


            *   Replace inheritance with delegation: Rather than rely on extensive inheritance hierarchies, the strategy pattern encourages 
                                                     composition and delegation.


    Here are some scenarios where the Strategy pattern is useful:

            *   When you want to use different variants of an algorithm within an object and be able to switch from one algorithm to 
                another during runtime.

            *   When you have a lot of similar classes that only differ in the way they execute some behavior.

            *   When you need to isolate the business logic of a class from the implementation details of algorithms that may not be as 
                important in the context of that logic.

            *   When your class has a massive conditional operator that switches between different variants of the same algorithm.

            *   One thing to keep in mind is that the Strategy pattern only makes sense when these strategies are used by the client interchangeably. 
                If a class uses a strategy object only once, there might be a simpler way to modify the class behavior, like using a function pointer.

'''


#@: NOTE : Example Code 1
#@: Interface 
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: list[int]) -> list[int]: 
        ...



#@: Concrete strategy algo: 1
class BubbleSortStrategy(SortStrategy):
    def sort(self, data: list[int]) -> list[int]:
        sorted_data = data.copy()
        n = len(sorted_data)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if sorted_data[j] > sorted_data[j + 1]:
                    sorted_data[j], sorted_data[j + 1] = sorted_data[j + 1], sorted_data[j]
        return sorted_data




#@: Concrete strategy algo: 2
class QuickSortStrategy(SortStrategy):
    def sort(self, data: list[int]) -> list[int]:
        if len(data) <= 1:
            return data
        else:
            pivot = data.pop()
            greater: list[int] = [x for x in data if x > pivot]
            lesser: list[int] = [x for x in data if x < pivot]
            return self.sort(lesser) + [pivot] + self.sort(greater)



#@: concrete strategy
#@: The concrete strategy to be used can be changed dynamically at runtime.
class Context:
    def __init__(self, strategy: SortStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: SortStrategy) -> None:
        self._strategy = strategy

    def execute_strategy(self, data: List[int]) -> List[int]:
        return self._strategy.sort(data)





#@: NOTE Exmaple code 2:
#@: implementing a shopping cart with various discount strategies.


class Item:
    def __init__(self, name: str, price: float) -> None:
        self.name = name
        self.price = price


#@: Abstract strategy 
class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, items: list[Item]) -> float:
        ...


#@: Algo 1:
class NoDiscountStrategy(DiscountStrategy):
    def calculate(self, items: List[Item]) -> float:
        return sum(item.price for item in items)


#@: Algo 2:
class FivePercentDiscountStrategy(DiscountStrategy):
    def calculate(self, items: List[Item]) -> float:
        total = sum(item.price for item in items)
        return total * 0.95  # apply 5% discount


#@: Algo 3:
class TenPercentDiscountStrategy(DiscountStrategy):
    def calculate(self, items: List[Item]) -> float:
        total = sum(item.price for item in items)
        return total * 0.90  # apply 10% discount



#@: Main Context
class ShoppingCart:
    def __init__(self, discount_strategy: DiscountStrategy) -> None:
        self._discount_strategy = discount_strategy
        self.items = []

    def add_item(self, item: Item) -> None:
        self.items.append(item)


    def calculate_total(self) -> float:
        return self._discount_strategy.calculate(self.items)


    def set_discount_strategy(self, discount_strategy: DiscountStrategy) -> None:
        self._discount_strategy = discount_strategy
        
        



#@: Driver Code
if __name__.__contains__('__main__'):
    #@: Client code
    #@: ex 1 
    data = [9, 5, 2, 7, 1, 6, 8, 0, 3, 4]
    context: Context = Context(BubbleSortStrategy())
    print(context.execute_strategy(data))

    context.set_strategy(QuickSortStrategy())
    print(context.execute_strategy(data))
    
    #@: ex 2
    item1 = Item("Apple", 10)
    item2 = Item("Banana", 20)
    item3 = Item("Cherry", 30)

    cart = ShoppingCart(NoDiscountStrategy())
    cart.add_item(item1)
    cart.add_item(item2)
    cart.add_item(item3)

    print("No discount strategy total: $", cart.calculate_total())

    cart.set_discount_strategy(FivePercentDiscountStrategy())
    print("5% discount strategy total: $", cart.calculate_total())

    cart.set_discount_strategy(TenPercentDiscountStrategy())
    