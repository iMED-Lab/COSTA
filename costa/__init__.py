from __future__ import absolute_import
import os

ascii_art = f"""
 ██████╗ ██████╗ ███████╗████████╗ █████╗ 
██╔════╝██╔═══██╗██╔════╝╚══██╔══╝██╔══██╗
██║     ██║   ██║███████╗   ██║   ███████║
██║     ██║   ██║╚════██║   ██║   ██╔══██║
╚██████╗╚██████╔╝███████║   ██║   ██║  ██║
         
A Multi-center Multi-vendor TOF-MRA Dataset 
and a Novel Cerebrovascular Segmentation Network
"""

terminal_width = int(os.get_terminal_size().columns)
border = "+" + "-" * (terminal_width - 2) + "+"

ascii_art_lines = ascii_art.strip().split("\n")
ascii_art_lines[0] = ' ' + ascii_art_lines[0]
padding_lines = (terminal_width - len(ascii_art_lines[0]) - 4) // 2
formatted_ascii_art = [
    f"| {' ' * ((terminal_width - len(line) - 4) // 2)}{line}{' ' * (terminal_width - len(line) - 4 - (terminal_width - len(line) - 4) // 2)} |"
    for line in ascii_art_lines]

print("\n")
print(border)
print(f"| {' ' * (terminal_width - 4)} |")
for line in formatted_ascii_art:
    print(line)
print(f"| {' ' * (terminal_width - 4)} |")
print(border)

# print(
#     "\n\n\033[5;31m COSTA: A Multi-center Multi-vendor TOF-MRA Dataset and A Novel Cerebrovascular Segmentation Network.\033[0m")

from . import *
