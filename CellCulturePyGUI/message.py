import sys
import time
from colored import stylize, attr, fg

program_message = \
'''
\033[1m Thanks for using CellCulturePy \033[0m

CellCulturePy tries to find the best growth media for you,
Outcoming tabel displays media with highest probability of growing.
Note that all media combinations can be saved afterwards

Here are the arguments you prodided:

-------------------------------------

{0}

Warnings:

-------------------------------------

CellFindPy is an ongoing project, so feel free to submit any bugs you find to the
issue tracker on Github[1], or drop me a line at dspaande@broadinstitute.org if ya want.

[1](https://github.com/Douwe-Spaanderman/Broad_DJ_AI)

See ya!

^_^

'''


def display_message(part):
    if part == 1:
        message = program_message.format('\n'.join(sys.argv[1:])).split('\n')
        message = message[:len(message)-12]

    if part == 2:
        message = program_message.format('\n'.join(sys.argv[1:])).split('\n')
        message = message[-13:]

    delay = 1.8 / len(message)

    for line in message:
        print(line)
        time.sleep(delay)
