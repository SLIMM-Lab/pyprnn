#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:07:01 2022

@author: malvesmaia
"""

# Class for defining personal printing style

class PrintingStyle (  ):
    def __init__(self):
        maxc = 255.0
        self.c01 = [ 141/maxc, 211/maxc, 199.0/maxc ]
        self.c02 =  [ 255/maxc, 255/maxc, 179/maxc ]
        self.c03 =  [ 190.0/maxc, 186.0/maxc, 218/maxc  ]
        self.c04 = [ 251.0/maxc, 128.0/maxc, 114/maxc  ]
        self.c05 = [ 128.0/maxc, 177.0/maxc, 211/maxc  ]
        self.c06 =  [ 128.0/maxc, 177.0/maxc, 211/maxc  ]
        self.c07 = [ 253.0/maxc, 180.0/maxc, 98/maxc  ]
        self.c08 =  [ 179.0/maxc, 222.0/maxc, 105/maxc  ]
        self.c09 =  [ 252.0/maxc, 205.0/maxc, 229/maxc  ]
        self.c10 =  [ 217.0/maxc, 217.0/maxc, 217/maxc  ]
        self.c11 =  [ 102.0/maxc, 102.0/maxc, 102/maxc  ]
        self.c12 =  [ 90.0/maxc, 174.0/maxc, 159/maxc  ]
        self.c13 = [ 33.0/maxc, 201.0/maxc, 160/maxc  ]
        
        self.font = {'fontname':'serif', 'fontsize':10}
