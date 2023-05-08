from __future__ import annotations
from abc import ABC, abstractmethod

#@: NOTE : Software Design Patterns 
#@: NOTE : Design Pattern 7: Proxy Pattern

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
    The Proxy design pattern is a structural pattern that involves an intermediary object, called a proxy, that acts 
    as a surrogate or placeholder for another object, called the real subject. The proxy and the real subject share a 
    common interface, so they can be used interchangeably by the client code. The proxy controls access to the real subject 
    by intercepting requests, adding some additional behavior, and then forwarding the requests to the real subject as necessary.

    
    The main idea behind the Proxy pattern is to provide a level of indirection between the client and the real subject. This 
    indirection allows for various types of additional behavior to be introduced, such as access control, lazy initialization, 
    or remote communication, without modifying the client code or the real subject.


    There are several types of proxies, each serving a specific purpose:

            *   Virtual Proxy: This type of proxy delays the creation and initialization of an expensive object until it is actually 
                               needed. This can improve performance and reduce memory usage by creating objects only when necessary.


            *   Remote Proxy: A remote proxy provides a local representation of an object that resides in a different address space, 
                              such as on a remote server. The remote proxy communicates with the remote object behind the scenes, 
                              hiding the complexity of remote communication from the client code.


            *   Protection Proxy: This type of proxy controls access to the real subject based on certain conditions, such as user 
                                  permissions or resource availability. This can be useful for implementing security mechanisms or 
                                  managing resource usage.


            *   Smart Reference Proxy: A smart reference proxy adds additional functionality when an object is accessed, such as 
                                       reference counting, caching, or logging. This can be useful for optimizing resource usage 
                                       or monitoring system behavior.


    In summary, the Proxy design pattern allows for the separation of concerns between the client code and the real subject by 
    introducing an intermediary object that controls access to the real subject. This enables various types of additional behavior 
    to be added without modifying the client code or the real subject, leading to more maintainable and flexible systems.

'''

#@: NOTE: Example 1: Basic Proxy Pattern

# This example demonstrates the basic Proxy pattern in Python.

#         *   Subject class: It is an abstract base class that defines the common interface for both the RealSubject and Proxy classes. It has an 
#                            abstract method request() that returns a string. Any concrete subclass of Subject must implement this method.


#         *   RealSubject class: This class is a concrete implementation of the Subject interface. It represents the actual object that the proxy 
#                                is meant to stand in for. The request() method of this class returns a string indicating that the request is being 
#                                handled by the RealSubject.


#         *   Proxy class: This class is another concrete implementation of the Subject interface. It acts as a proxy for the RealSubject. The Proxy 
#                          class has a constructor that takes a RealSubject object as an argument and stores it in the instance variable _real_subject.
#                          The request() method of the Proxy class returns a string that includes the result of the RealSubject's request() method, 
#                          demonstrating that the proxy is delegating the request to the real subject.
                         

# In this basic example, the Proxy pattern is used to provide a level of indirection between the client code and the real subject. The client code can 
# interact with the Proxy object, which in turn delegates the request to the RealSubject. While this example doesn't add any additional behavior to the 
# proxy, it lays the groundwork for more complex use cases, such as access control, lazy initialization, or remote communication, which can be built upon 
# this basic structure.

class Subject(ABC):
    @abstractmethod
    def request(self) -> str:
        ...


class RealSubject(Subject):
    def request(self) -> str:
        return "RealSubject: Handling request."



class Proxy(Subject):
    def __init__(self, real_subject: RealSubject) -> None:
        self._real_subject = real_subject


    def request(self) -> str:
        return f"Proxy: {self._real_subject.request()}"



#@: NOTE: Example 2: Protection Proxy

# This example demonstrates the Protection Proxy pattern in Python, which is a variation of the basic Proxy pattern.

#         *   ProtectionProxy class: This class is a concrete implementation of the Subject interface and acts as a protection proxy for the RealSubject. 
#                                    It has a constructor that takes a RealSubject object as an argument and stores it in the instance variable _real_subject.
                                   
        
#         *   request() method: The request() method of the ProtectionProxy class checks if the client has access to the RealSubject by calling the check_access()
#                               method. If access is granted, the request is delegated to the RealSubject, and the result is returned with a "ProtectionProxy" 
#                               prefix. If access is denied, the method returns a string indicating that access has been denied.
                              
        
#         *   check_access() method: This method is responsible for checking if the client has access to the RealSubject. In this example, it is a simple placeholder
#                                    that always returns True, but in a real-world scenario, it should be replaced with a proper access control mechanism, such as 
#                                    checking user permissions or roles.
                                   

# In this example, the Protection Proxy pattern is used to provide an access control mechanism for the RealSubject. The client code interacts with the ProtectionProxy 
# object, which checks if the client has access before delegating the request to the RealSubject. This allows for the implementation of security mechanisms without
# modifying the client code or the RealSubject.


class ProtectionProxy(Subject):
    def __init__(self, real_subject: RealSubject) -> None:
        self._real_subject = real_subject


    def request(self) -> str:
        if self.check_access():
            return f"ProtectionProxy: {self._real_subject.request()}"
        else:
            return "ProtectionProxy: Access denied."


    def check_access(self) -> bool:
        # Replace with a proper access check, e.g., user permissions
        return True



#@: NOTE: Example 3: Virtual Proxy

# This example demonstrates the Virtual Proxy pattern in Python, which is another variation of the basic Proxy pattern.

#         *   VirtualProxy class: This class is a concrete implementation of the Subject interface and acts as a virtual proxy for the RealSubject. In the constructor, 
#                                 it initializes an instance variable _real_subject with a value of None. This variable will store the RealSubject object once it is 
#                                 created.
        
        
#         *   request() method: The request() method of the VirtualProxy class checks if the _real_subject instance variable is None. If it is, it creates a new RealSubject 
#                               object and assigns it to _real_subject. This means that the RealSubject object is created and initialized only when it is actually needed 
#                               (i.e., when the request() method is called for the first time). After ensuring that the _real_subject instance variable contains a RealSubject 
#                               object, the method delegates the request to the RealSubject and returns the result with a "VirtualProxy" prefix.        


# In this example, the Virtual Proxy pattern is used to delay the creation and initialization of the RealSubject object until it is actually needed. This can be useful for 
# improving performance and reducing memory usage in cases where the RealSubject object is expensive to create or initialize. The client code interacts with the VirtualProxy 
# object, which handles the lazy initialization of the RealSubject and delegates the request to it when required.                          

class VirtualProxy(Subject):
    def __init__(self) -> None:
        self._real_subject: Optional[RealSubject] = None

    
    def request(self) -> str:
        if not self._real_subject:
            self._real_subject = RealSubject()
        return f"VirtualProxy: {self._real_subject.request()}"
    
    
    
#@: NOTE: Example 4: Remote Proxy

# This example demonstrates the Remote Proxy pattern in Python, which is another variation of the basic Proxy pattern.

#             *   RemoteSubject class: This class is a concrete implementation of the Subject interface and represents a remote object. In this example, the request() 
#                                      method simulates making a request to a remote server using the requests.get() method (you would need to import the requests library 
#                                      to use this). The request() method returns the text content of the server's response.
                                     
            
#             *   RemoteProxy class: This class is another concrete implementation of the Subject interface and acts as a remote proxy for the RemoteSubject. The RemoteProxy 
#                                    class has a constructor that takes a RemoteSubject object as an argument and stores it in the instance variable _remote_subject.
                                   
                                   
#             *   request() method: The request() method of the RemoteProxy class delegates the request to the _remote_subject object (i.e., the RemoteSubject instance) and 
#                                   returns the result with a "RemoteProxy" prefix.


# In this example, the Remote Proxy pattern is used to provide a level of indirection between the client code and the remote object (in this case, the RemoteSubject). The 
# client code interacts with the RemoteProxy object, which in turn delegates the request to the RemoteSubject. This pattern can help encapsulate the complexity of remote 
# communication and make it easier to handle errors or retries.


class RemoteSubject(Subject):
    def request(self) -> str:
        # Simulating a request to a remote server
        # response = requests.get("https://example.com/data")
        # return response.text
        return ...
    
    

class RemoteProxy(Subject):
    def __init__(self, remote_subject: RemoteSubject) -> None:
        self._remote_subject = remote_subject


    def request(self) -> str:
        return f"RemoteProxy: {self._remote_subject.request()}"
    


#@: NOTE: Example 5: Smart Reference Proxy (with caching)

# This example demonstrates the Smart Reference Proxy pattern in Python, which is another variation of the basic Proxy pattern. In this case, the Smart Reference 
# Proxy is used for caching.

#                 *   SmartReferenceProxy class: This class is a concrete implementation of the Subject interface and acts as a smart reference proxy for the RealSubject. 
#                                                In the constructor, it initializes an instance variable _real_subject with a RealSubject object and an instance variable 
#                                                _cache with a value of None. The _cache variable will store the result of the RealSubject's request once it is fetched.


#                 *   request() method: The request() method of the SmartReferenceProxy class first checks if the _cache instance variable is None. If it is, it calls 
#                                       the request() method of the _real_subject object (i.e., the RealSubject instance) and stores the result in the _cache. After 
#                                       ensuring that the _cache instance variable contains the result of the RealSubject's request, the method returns the result with 
#                                       a "SmartReferenceProxy (cache)" prefix.


# In this example, the Smart Reference Proxy pattern is used to cache the result of the RealSubject's request, preventing redundant requests to the RealSubject and improving
# performance. The client code interacts with the SmartReferenceProxy object, which handles the caching and returns the cached result when available. This pattern can be 
# particularly useful when dealing with expensive operations or slow resources, such as remote services or large data sets.

class SmartReferenceProxy(Subject):
    def __init__(self, real_subject: RealSubject) -> None:
        self._real_subject = real_subject
        self._cache: Optional[str] = None

    
    def request(self) -> str:
        if not self._cache:
            self._cache = self._real_subject.request()
        return f"SmartReferenceProxy (cache): {self._cache}"
    
    

#@: Driver code
if __name__.__contains__('__main__'):
    #@: Ex: 1
    real_subject = RealSubject()
    proxy = Proxy(real_subject)

    response = proxy.request()
    print(response)
    
    #@: Ex: 2
    real_subject = RealSubject()
    protection_proxy = ProtectionProxy(real_subject)

    response = protection_proxy.request()
    print(response)
    
    #@: Ex: 3
    virtual_proxy = VirtualProxy()
    response = virtual_proxy.request()
    print(response)
    
    
    #@: Ex: 4
    remote_subject = RemoteSubject()
    remote_proxy = RemoteProxy(remote_subject)

    response = remote_proxy.request()
    print(response)
    
    
    #@: Ex: 5
    real_subject = RealSubject()
    smart_reference_proxy = SmartReferenceProxy(real_subject)

    # First request - the result will be fetched from the RealSubject and cached
    response = smart_reference_proxy.request()
    print(response)

    # Second request - the result will be fetched from the cache
    response = smart_reference_proxy.request()
    print(response)
    
    
    
    