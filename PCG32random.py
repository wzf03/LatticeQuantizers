## PCG32 Random Number Generator in Python (Micropython port) ##
# Original Source:   PCG Random Number Generation for C
# Minimal C Implementation
# Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
# http://www.pcg-random.org
#
# This is the Python implementation of the PCG32 random number generator, ported
# from the C  for use on embedded systems like ESP32, RP2040, Pybaord.
# This implementation uses the PCG32 (Permuted Congruential Generator) method. 
# https://github.com/MicroControleurMonde
# PCG32_Minimal_Port_PyBoard.py

import machine
import time

class PCG32Random:
    """
    PCG32 Random Number Generator class.
    Implements the PCG32 algorithm, a fast and high-quality random number generator.
    The generator is initialized with a 64-bit seed and a 64-bit increment. If no 
    values are provided for the seed and increment, they are automatically generated 
    using the device's unique ID and the current time.
    Attributes:
        state (int): The internal state of the generator (64 bits).
        inc (int): The increment used for the generator's state update (64 bits).
    """

    def __init__(self, seed=None, inc=None):
        """
        Initializes the PCG32 random number generator with an optional seed and increment.
        If no seed is provided, the generator will use the device's unique ID 
        combined with the current time in milliseconds to generate a seed.
        If no increment is provided, it will default to 1.
        Args:
            seed (int, optional): The initial seed for the generator (64-bit). 
            inc (int, optional): The increment for the state update (64-bit).
        """
        if seed is None:
            # Use the device's unique ID and current time as a seed for more variability
            seed = int.from_bytes(machine.unique_id(), 'big') + int(time.ticks_ms())
        if inc is None:
            inc = 1  # Default increment

        # Convert seed and increment to 64-bit integers
        self.state = seed & 0xFFFFFFFFFFFFFFFF  # 64-bit seed
        self.inc = (inc << 1) | 1  # Ensure the increment is odd

    def random(self):
        """
        Generates a 32-bit pseudo-random value using the PCG32 algorithm.
        The internal state is updated, and a 32-bit random number is produced 
        based on the old state using the XSH RR output function.
        Returns:
            int: A 32-bit random number.
        """
        # Save the old state
        oldstate = self.state

        # Advance the internal state
        self.state = (oldstate * 6364136223846793005) + self.inc
        self.state &= 0xFFFFFFFFFFFFFFFF  # Mask to 64 bits

        # Xorshift and rotate (XSH RR)
        xorshifted = (oldstate >> 18) ^ oldstate
        xorshifted = (xorshifted >> 27) & 0xFFFFFFFF  # Mask to 32 bits
        rot = oldstate >> 59

        # Perform the rotation and return the result
        return ((xorshifted >> rot) | (xorshifted << (32 - rot))) & 0xFFFFFFFF

    def randint(self, start, end):
        """
        Generates a random integer in the specified range [start, end].
        This function uses the `random()` method to generate a number and then
        maps it to the given range.
        Args:
            start (int): The lower bound of the range.
            end (int): The upper bound of the range.
        Returns:
            int: A random integer in the range [start, end].
        """
        return start + (self.random() % (end - start + 1))

def generate_random_reals(n):
    """
    Generates n random real numbers in the range [0, 1) using the PCG32 algorithm.
    Args:
        n (int): The number of random numbers to generate.
    Returns:
        list: A list of n random real numbers in the range [0, 1).
    """
    # Initialize the PCG32 random number generator
    pcg = PCG32Random()

    # Generate n random real numbers in the range [0, 1)
    return [pcg.random() / 0xFFFFFFFF for _ in range(n)]

