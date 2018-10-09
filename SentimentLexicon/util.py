#-*- encoding:utf-8 -*-
import re

def read_lines(path):
    all_lines = []
    with open(path, 'r', encoding='utf-8') as file:
        temp_lines = file.readlines()
        for line in temp_lines:
            line = line.strip()
            if line:
                all_lines.append(line)
    return all_lines

