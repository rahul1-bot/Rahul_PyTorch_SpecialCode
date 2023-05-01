from __future__ import annotations

#@: NOTE : Software Design Patterns 
#@: NOTE : Pattern 5: Facade Pattern 

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
    *   The Facade design pattern is a structural design pattern that provides a simplified interface to a 
        complex subsystem. The term 'facade' translates to 'frontage' or 'face' in French, which is a fitting 
        description of this pattern's functionality. It serves as a front-facing interface masking more complex 
        underlying or structural code.


    *   The Facade pattern plays a crucial role in promoting system organization and functionality with a high-level 
        interface. It can help manage and partition a system into subsections, each with its own set of interfaces, 
        which can be simplified with the Facade.


    *   The pattern involves a single class called the Facade, which provides simplified methods required by the client. 
        These methods delegate the call to methods of existing system classes. The Facade doesn't encapsulate these subsystem
        classes but composes them, meaning the Facade has an association relationship with the subsystem classes, not an 
        inheritance relationship.


    *   The main advantage of the Facade pattern is that it shields clients from complex subsystem components, reducing 
        dependencies and promoting loose coupling. It hides the intricacies of the complex subsystems, providing a simpler 
        interface to the client. It typically involves a single wrapper class that contains a set of members required by the 
        client. These members access the system on behalf of the facade client and hide the implementation details.

    
    Let's consider an example. Imagine a home theater system with several components: a DVD player, a projector, speakers, lights, 
    and a popcorn machine. Each of these components has its own set of controls and interfaces, and they all need to work together 
    to create the perfect movie-watching experience.

    For someone unfamiliar with the system, it would be a daunting task to understand how to properly operate each component 
    and in what sequence. This is where the Facade pattern comes into play.

    
    Instead of expecting users to interact with each component individually, we can create a HomeTheaterFacade that provides 
    simple methods like 'watchMovie()' and 'endMovie()'. The 'watchMovie()' method, for instance, could turn down the lights, 
    start the popcorn machine, put the DVD in the player, start the player, and set the projector to the DVD input.


    The 'endMovie()' method could turn on the lights, stop the DVD player, stop the popcorn machine, and turn off the projector.
    Users don't need to know the specifics of how to perform these actions for each component; they only interact with the 
    simplified interface provided by the HomeTheaterFacade.


    The Facade pattern offers many benefits. It provides a simple interface to a complex subsystem, making it easier for users and 
    reducing potential errors. It decouples a client implementation from the complex subsystem, making it less prone to errors that 
    can occur when a subsystem evolves over time.


    It also promotes subsystem independence and portability because clients only communicate with the subsystems through the facade. 
    The facade can also help divide responsibilities in a subsystem and delegate to internal classes, making it easier to manage the 
    subsystem.


    However, it's important to remember that while the Facade pattern simplifies the interface, it doesn't prevent the client from using
    the subsystem classes directly if they need to. It just provides a simpler alternative.
'''
#@: NOTE : Example 1

# This code is a Python implementation of the Facade design pattern. The pattern is used to provide a simplified interface to a complex subsystem, 
# and in this case, the subsystem is a computer with all of its components: CPU, Memory, and HardDrive.

# Here's a breakdown of what each class represents and does:

#     1.  CPU Class: This class represents the Central Processing Unit of a computer. It has three methods:

#             *   freeze(): This method simulates the CPU freezing its current tasks.

#             *   jump(position: Any): This method simulates the CPU jumping to a different position or instruction in the memory.

#             *   execute(): This method simulates the CPU executing an instruction.

    
#     2.  Memory Class: This class represents the Memory of a computer. It has one method:

#             *   load(position: Any, data: Any): This method simulates the memory loading data at a given position.


#     3.  HardDrive Class: This class represents the Hard Drive of a computer. It has one method:

#             *   read(lba: Any, size: Any): This method simulates the hard drive reading data of a certain size at a given logical block 
#                                            addressing (lba).


#     4.  Computer Class: This class acts as the Facade in this design pattern. It simplifies the interface of the subsystem (CPU, Memory, 
#                         HardDrive) and provides a simple method to start the computer. It has two methods:

#             *   __init__(): This constructor initializes new instances of CPU, Memory, and HardDrive when a Computer object is created.

#             *   start_computer(): This method simulates the process of starting a computer. It first freezes the CPU, then loads the boot 
#                                   address from the hard drive into the memory, makes the CPU jump to the boot address, and finally executes 
#                                   the CPU.

# The start_computer method inside the Computer class hides the complexities involved in the process of starting a computer. The client, which uses 
# this Computer class, doesn't need to know about the sequence of operations or the interaction between CPU, Memory, and HardDrive. It can just call 
# the start_computer method to start the computer.

# In summary, this code represents a simplified computer system where the Computer class (the facade) encapsulates the complexities of the subsystem 
# (CPU, Memory, and HardDrive). It provides a simpler interface to the client, making the subsystem easier to use.

class CPU:
    def freeze(self) -> None:
        print("CPU is freezing")
        

    def jump(self, position: Any) -> None:
        print(f"CPU jumps to: {position}")
        

    def execute(self) -> None:
        print("CPU is executing")



class Memory:
    def load(self, position: Any, data: Any) -> None:
        print(f"Memory loads data: {data} at position: {position}")
        


class HardDrive:
    def read(self, lba: Any, size: Any) -> str:
        return f"Hard Drive reads {size} data at {lba}"



class Computer:
    def __init__(self) -> None:
        self.cpu = CPU()
        self.memory = Memory()
        self.hard_drive = HardDrive()
        
    def start_computer(self) -> None:
        self.cpu.freeze()
        self.memory.load("BOOT_ADDRESS", self.hard_drive.read("BOOT_SECTOR", "SECTOR_SIZE"))
        self.cpu.jump("BOOT_ADDRESS")
        self.cpu.execute()


#@: NOTE : Example 2
# This code provides a Python implementation of the Facade design pattern, applied to a hypothetical home automation system. The home automation system 
# consists of various devices like Light, AirConditioner, Television, and MusicSystem. Each of these devices can be controlled independently, but to 
# simplify their operation, a Facade (HomeAutomationFacade) is provided.

# Here's a breakdown of the classes and their functionalities:

#         *   Light: This class represents a light in the home automation system. It has two methods, on and off, which print messages indicating whether
#                    the light is turned on or off.


#         *   AirConditioner: This class represents an air conditioner in the home automation system. Similar to the Light class, it has on and off methods 
#                             to control the air conditioner.


#         *   Television: This class represents a television in the home automation system. It also has on and off methods to control the television.


#         *   MusicSystem: This class represents a music system. It has a couple of additional methods apart from on and off:


#                 *   setCD: This method takes a CD name as input and prints a message indicating that the CD has been set in the music system.


#                 *   setVolume: This method takes a volume level as input and prints a message indicating that the volume has been set.


#         *   HomeAutomationFacade: This class represents the Facade in the design pattern. It provides a simplified interface to control all the devices 
#                                   in the home automation system. It has the following methods:


#                 *   partyMode: This method turns on the light and the air conditioner, turns off the television, and sets the music system to play party 
#                                songs at a volume level of 10.


#                 *   sleepMode: This method turns off the light, the television, and the music system, and turns on the air conditioner.


#                 *   offEverything: This method turns off all devices.


# When a user interacts with the home automation system, they don't need to control each device independently. Instead, they can use the methods provided by the 
# HomeAutomationFacade, which encapsulate the complexity of controlling multiple devices. For instance, if the user wants to start a party, they just need to call 
# the partyMode method, and the Facade takes care of setting up all devices appropriately. This is the power of the Facade design pattern.

class Light:
    def on(self) -> None:
        print("Light is on")
    
    def off(self) -> None:
        print("Light is off")



class AirConditioner:
    def on(self) -> None:
        print("AirConditioner is on")

    def off(self) -> None:
        print("AirConditioner is off")



class Television:
    def on(self) -> None:
        print("Television is on")

    def off(self) -> None:
        print("Television is off")



class MusicSystem:
    def on(self) -> None:
        print("MusicSystem is on")

    def off(self) -> None:
        print("MusicSystem is off")

    def setCD(self, cd: Any) -> None:
        print(f"MusicSystem is set with {cd}")

    def setVolume(self, volume: int) -> None:
        print(f"MusicSystem volume is set to {volume}")



class HomeAutomationFacade:
    def __init__(self) -> None:
        self.light = Light()
        self.ac = AirConditioner()
        self.tv = Television()
        self.music_system = MusicSystem()


    def partyMode(self) -> None:
        print("Switching to Party Mode...")
        self.light.on()
        self.ac.on()
        self.tv.off()
        self.music_system.on()
        self.music_system.setCD("Party Songs")
        self.music_system.setVolume(10)
        print("Party Mode is ON!")


    def sleepMode(self) -> None:
        print("Switching to Sleep Mode...")
        self.light.off()
        self.ac.on()
        self.tv.off()
        self.music_system.off()
        print("Sleep Mode is ON!")


    def offEverything(self) -> None:
        print("Switching off Everything...")
        self.light.off()
        self.ac.off()
        self.tv.off()
        self.music_system.off()
        print("Everything is OFF!")




#@: Driver Code
if __name__.__contains__('__main__'):
    computer = Computer()
    computer.start_computer()
    
    home_automation = HomeAutomationFacade()
    home_automation.partyMode()
    home_automation.sleepMode()
    home_automation.offEverything()
    
    