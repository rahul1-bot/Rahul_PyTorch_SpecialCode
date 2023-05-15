from __future__ import annotations
from dataclasses import dataclass, field
import torch
from torch import nn
from torchvision import transforms


#@: NOTE : Python Data Classes

__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Mail': [
        'sawhney.rahulofficial@outlook.com', 
        'rahulsawhney321@gmail.com'
    ]
}


_license__: str = r'''
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

# Python's dataclasses are a way of simplifying the creation of classes which primarily exist to hold values. 

# Dataclasses use decorator syntax to automatically add special methods to your classes, such as __init__, __repr__, and __eq__, based 
# on variable annotations. This can make your code cleaner and more efficient by reducing boilerplate.

@dataclass
class Point:
    x: int
    y: int
    


# The @dataclass decorator tells Python that this is a dataclass. The x: int and y: int lines are variable annotations that indicate the 
# class has two attributes, x and y, both of which are integers.

# The above code is roughly equivalent to the following:
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


    def __repr__(self) -> str:
        return f'Point(x={self.x}, y={self.y})'


    def __eq__(self, other: Callable[Any]) -> bool | NotImplemented:
        if other.__class__ is self.__class__:
            return (self.x, self.y) == (other.x, other.y)
        return NotImplemented
    

# Dataclasses also provide a number of other features:

#         *   Default values: You can provide default values for fields, which will be used if no value is provided when 
#                             creating an instance of the class.


#         *   Immutable dataclasses: By setting frozen= True in the dataclass decorator, you can create an immutable dataclass, 
#                                    where the values cannot be changed after creation.


#         *   Inheritance: Dataclasses support inheritance, meaning a dataclass can inherit from another dataclass or a regular 
#                          class, and a regular class can inherit from a dataclass.


#         *   Field customization: The field() function can be used to customize individual fields, such as specifying a default 
#                                  factory function for a field, or controlling whether the field is included in the generated 
#                                  __repr__ and __eq__ methods.


@dataclass
class InventoryItem:
        # *   @dataclass is a decorator that automatically adds special methods like __init__, __repr__, and __eq__ to the class based 
        #     on the class attributes defined.


        # *   InventoryItem is the class name. This class represents an item in an inventory.


        # *   The class attributes are name, unit_price, quantity_on_hand, and categories.

        #         *   name is the name of the inventory item and it is of type str.

        #         *   unit_price is the cost per unit of the inventory item and it is of type float.

        #         *    quantity_on_hand is the number of units available in the inventory. It's of type int and defaults to 0 
        #              if not provided.

        #         *   categories is a list of categories that the item belongs to. It's a list of strings (list[str]) and defaults 
        #             to an empty list if not provided. This is established using field(default_factory=list), which is a way to 
        #             provide a default value for fields that are mutable.

        
        # *   The class also has a method total_cost:


        # *   total_cost(self) -> float: is a method that calculates the total cost of the inventory items on hand by multiplying the 
        #                                unit_price with the quantity_on_hand. It returns a float. This method can be called on an instance 
        #                                of InventoryItem.
    
    name: str
    unit_price: float
    quantity_on_hand: int = 0
    categories: list[str] = field(default_factory= list)

    
    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
    
    




# Dataclasses are mainly used when you need to bundle data with behavior. They are particularly suited to situations where you have classes 
# that primarily store values and you want to minimize the boilerplate associated with them. Here are a few examples:

#         *   Modeling domain entities: In many applications, you need to model entities from your problem domain as classes. For instance, 
#                                       if you're building a customer management system, you might have Customer, Order, and Product entities. 
#                                       These classes primarily store data, so they can be modeled as dataclasses.


#         *   As DTOs (Data Transfer Objects): In software architectures, DTOs are often used to carry data between processes or between tiers 
#                                              of an application. They are simple objects that hold data and do not contain business logic.


#         *   Representing mathematical or geometric objects: You can use dataclasses to represent mathematical or geometric objects. 
                                                            

#         *   Configuration: Dataclasses can be used to represent configuration. For instance, if you're writing a web scraper, you might have 
#                            a ScraperConfig dataclass that stores all the settings for a scraper.
        
        
#         *   Immutable Data Structures: When you want to create an immutable data structure, you can use dataclasses with the frozen=True parameter.
#                                        This will make the generated class immutable, meaning you can't change the field values after instantiation. 
#                                        This can be useful for example in multithreaded programs where you want to ensure that data doesn't get changed 
#                                        after creation. 
                           
@dataclass
class Customer:
    identifier: int
    name: str
    email: str


@dataclass
class UserDto:
    username: str
    email: str


@dataclass
class Point:
    x: int
    y: int


@dataclass
class ScraperConfig:
    url: str
    retry_times: Optional[int] = 3
    timeout: Optional[int] = 30


@dataclass(frozen= True)
class ImmutablePoint:
    x: int
    y: int



#@: NOTE : Example 1: Simple Data Class
@dataclass
class Person:
    name: str
    age: int
    address: str
    

#@: NOTE: Example 2: Dataclass with Default Values
@dataclass
class Car:
    make: str
    model: str
    year: Optional[int] = 2022


#@: NOTE: Example 3: Dataclass with Methods
@dataclass
class Rectangle:
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height
    

#@: NOTE: Example 4: Dataclass with Post-Initialization Processing
@dataclass
class Student:
    name: str
    grade: float
    status: Optional[str] = 'Inactive'


    def __post_init__(self) -> None:
        if self.grade >= 70:
            self.status = 'Active'


#@: NOTE: Example 5: Nested Dataclasses with Default Factory Method
@dataclass
class Book:
    title: str
    author: str


@dataclass
class Library:
    name: str
    books: list[Book] = field(default_factory= list)
    


#@: NOTE: Example 6: Dataclasses with list, dict
@dataclass
class School:
    name: str
    students: list[dict[str, str]] = field(default_factory= list)
    teachers: dict[str, str] = field(default_factory= dict)


    def add_student(self, name: str, grade: str) -> None:
        self.students.append({'name': name, 'grade': grade})


    def add_teacher(self, subject: str, name: str) -> None:
        self.teachers[subject] = name
        
        
        
#@: NOTE: Dataclasses can be used in PyTorch to define configurations, encapsulate data, or 
# even define more complex structures like custom layers or models

@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    output_size: int
    learning_rate: float


@dataclass
class TrainConfig:
    batch_size: int
    num_epochs: int
    device: Optional[str] = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class DatasetConfig:
    root: str
    transform: transforms.Compose
    
    
    

#@: Driver Code 
if __name__.__contains__('__main__'):
    #@: NOTE: Use Case ex:
    school = School("Awesome School")

    school.add_student("John Doe", "5th Grade")
    school.add_student("Jane Doe", "6th Grade")

    school.add_teacher("Math", "Mr. Smith")
    school.add_teacher("English", "Ms. Johnson")

    print(school.students)
    print(school.teachers)


    model_config = ModelConfig(
        input_size= 784, 
        hidden_size= 500, 
        output_size= 10, 
        learning_rate= 0.001
    )
    
    train_config = TrainConfig(
        batch_size= 64, 
        num_epochs= 5
    )
    
    dataset_config = DatasetConfig(
        root= './data', 
        transform= transforms.ToTensor()
    )

    model: nn.Module = nn.Sequential(
        nn.Linear(model_config.input_size, model_config.hidden_size),
        nn.ReLU(),
        nn.Linear(model_config.hidden_size, model_config.output_size),
        nn.LogSoftmax(dim=1)
    )

    print(model)


    #@: OUTPUT:
    # [
    #     {'name': 'John Doe', 'grade': '5th Grade'}, 
    #     {'name': 'Jane Doe', 'grade': '6th Grade'}
    # ]
    
    # {'Math': 'Mr. Smith', 'English': 'Ms. Johnson'}
    
    # Sequential(
    #     (0): Linear(in_features=784, out_features=500, bias=True)
    #     (1): ReLU()
    #     (2): Linear(in_features=500, out_features=10, bias=True) 
    #     (3): LogSoftmax(dim=1)
    # )