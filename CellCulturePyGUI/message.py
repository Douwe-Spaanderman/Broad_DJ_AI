import sys
import time

program_message = \
'''
Thanks for using CellCulturePy

CellCulturePy tries to find the best growth media for you,
Outcoming tabel displays media with highest probability of growing.
Note that all media combinations can be saved afterwards

Here are the arguments you prodided:

-------------------------------------

{0}

Results: 



-------------------------------------

CellFindPy is an ongoing project, so feel free to submit any bugs you find to the
issue tracker on Github[1], or drop me a line at dspaande@broadinstitute.org if ya want.

[1](https://github.com/Douwe-Spaanderman/Broad_DJ_AI)

See ya!

^_^

'''


def display_message():
    message = program_message.format('\n'.join(sys.argv[1:])).split('\n')
    delay = 1.8 / len(message)

    for line in message:
        print(line)
        time.sleep(delay)
