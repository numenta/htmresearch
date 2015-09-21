#!/bin/bash
g++ -c extractMNIST.cpp
g++ -o extractMNIST extractMNIST.o
rm extractMNIST.o
