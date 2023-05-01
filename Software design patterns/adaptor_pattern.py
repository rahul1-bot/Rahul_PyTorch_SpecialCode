from __future__ import annotations
from typing import Protocol
import argparse

#@: NOTE : Software Design Patterns 
#@: NOTE : Pattern 4: Adaptor Pattern 

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
    The Adapter Design Pattern is a structural design pattern that allows objects with incompatible interfaces to work
    together. This is accomplished by wrapping an interface around the existing object to create a new interface that is 
    compatible with the client code.

    The Adapter pattern is especially useful when you want to integrate some existing component into your application, but 
    the component's interface doesn't match the rest of your code.

    The Adapter acts as a bridge between two incompatible interfaces by wrapping an "adaptee" and transforming the interface 
    of the adaptee into an interface that the client can understand.

    There are two types of adapter patterns:

        *   Object Adapter Pattern: This pattern uses composition to wrap the adaptee. It creates an adapter class that wraps 
                                    the adaptee object. This adapter class will redefine the interface exposed by the adaptee. 
                                    If the adaptee's interface changes, only the adapter code will need to change to match it.


        *   Class Adapter Pattern: This pattern uses inheritance. It extends the classes of both the target and the adaptee. 
                                   Since Python supports multiple inheritance, we could use this pattern in Python. However, 
                                   it's generally recommended to use composition over inheritance, so the object adapter pattern
                                   is more commonly used.

'''

#@: Example 1:
'''
This code is an implementation of the Adapter design pattern in Python. The purpose of this design pattern is to 
allow classes with incompatible interfaces to work together. Here's how each component works:

    *   Target Interface (Target): This is the interface that the client expects to work with. In this code, it's 
                                   represented by a Python Protocol, which is a way of defining an interface that a 
                                   class must adhere to. The Target interface declares a method named request() that 
                                   returns a string.


    *   Adaptee (Adaptee): This is a class that provides some useful behavior, but its interface is not compatible with 
                           the client code. The client code expects to call request() on objects it works with, but this 
                           class provides a specific_request() method instead.


    *   Adapter (Adapter): This is a class that's able to "adapt" the Adaptee's interface to the Target interface. It 
                           does this by containing an instance of the Adaptee and implementing the Target interface. 
                           The Adapter's request() method calls specific_request() on the Adaptee, then reverses the 
                           resulting string (this is the [::-1] part) to transform the Adaptee's behavior into a form 
                           that the client code can work with.


    *   Client Code (client_code function): This is the part of the code that relies on objects that implement the Target 
                                            interface. It calls request() on whatever object it's given, and expects to receive 
                                            a string in return. Because the Adapter class adapts the Adaptee to the Target 
                                            interface, the client code can work with an instance of the Adaptee via an Adapter.


In summary, the Adapter class allows an instance of the Adaptee to be used where an instance of a Target is expected. It does this 
by transforming the specific_request() of the Adaptee into a request() that the client code can use. This is the purpose of the 
Adapter design pattern: to provide a way for classes with incompatible interfaces to work together.
'''
class Target(Protocol):
    def request(self) -> str:
        ...
        
        
class Adaptee:
    def specific_request(self) -> str:
        return '.eetpadA eht fo roivaheb laicepS'


class Adapter(Target):
    def __init__(self, adaptee: Adaptee) -> None:
        self.adaptee = adaptee
    
    
    def request(self) -> str:
        return self.adaptee.specific_request()[::-1]
    


def client_code(target: Target) -> str:
    return target.request()



#@: Example 2:
# This is the TargetTwo interface, which is the interface the Client uses.
'''
This code is another example of the Adapter design pattern, this time used to handle command-line arguments and set them as parameters 
in another class. Here's a breakdown:

    *   Target Interface (TargetTwo): This is the interface that client objects expect to interact with. The TargetTwo interface 
                                      declares a method named set_params(), which takes a dictionary of parameters.


    *   Adaptee (CmdParser): This class represents a command-line argument parser. It uses Python's built-in argparse library to 
                             define and parse command-line arguments. Its get_params() method returns a dictionary of the parsed 
                             arguments. However, the interface of CmdParser isn't compatible with TargetTwo interface which the 
                             client code expects, because CmdParser doesn't have a set_params() method.


    *   Adapter (ParserAdapter): This class adapts the CmdParser's interface to the TargetTwo interface. It contains an instance 
                                 of CmdParser and implements the set_params() method. In set_params(), it calls get_params() on 
                                 the CmdParser to get the command-line arguments, then assigns them to the input params dictionary. 
                                 This allows the ParserAdapter to be used where a TargetTwo is expected.


    *   Client Class (MyClass): This class represents a part of your application that needs to use the parsed command-line arguments. 
                                It has a params dictionary to store the parameters and a set_params() method to set them. It also 
                                has a display_params() method to return its parameters.


In this code, an instance of MyClass would use a ParserAdapter to set its parameters from the command-line arguments. The ParserAdapter 
adapts the interface of the CmdParser to the interface that MyClass expects (TargetTwo), allowing MyClass to use CmdParser to parse 
command-line arguments.

In summary, the ParserAdapter class allows an instance of CmdParser to be used where an instance of a TargetTwo is expected. It does this 
by transforming the get_params() of CmdParser into a set_params() that the client code (MyClass in this case) can use. This is the purpose 
of the Adapter design pattern: to provide a way for classes with incompatible interfaces to work together.
'''
class TargetTwo(Protocol):
    def set_params(self, params: Any) -> None:
        ...


# This is an Adaptee class which has a different interface from the Target.
class CmdParser:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--param1')
        self.parser.add_argument('--param2')
        self.args = self.parser.parse_args()


    def get_params(self) -> dict[str, Any]:
        return vars(self.args)



# This is the Adapter class which adapts the interface of the Adaptee to the Target interface.
class ParserAdapter(TargetTwo):
    def __init__(self, parser: CmdParser) -> None:
        self.parser = parser


    def set_params(self, params: Any) -> None:
        args = self.parser.get_params()
        for key, value in args.items():
            params[key] = value



# This is an example class that would use the parameters.
class MyClass:
    def __init__(self):
        self.params: dict[str, Any] = {}

    def set_params(self, params: Any) -> None:
        self.params = params


    def display_params(self) -> Any:
        return self.params



    
#@: Driver Code
if __name__.__contains__('__main__'):
    adaptee = Adaptee()
    print(f"Adaptee: {adaptee.specific_request()}")

    print("Client: But I can work with it via the Adapter:")
    adapter = Adapter(adaptee)
    print(client_code(adapter))
    
    #@: Parsing the command line arguments
    cmd_parser = CmdParser()
    
    #@: Adapting the parsed command line arguments to the interface expected by MyClass
    adapter = ParserAdapter(cmd_parser)

    my_class = MyClass()
    adapter.set_params(my_class.params)

    #@: Displaying the parameters
    print(my_class.display_params())
    
    