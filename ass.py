#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:39:47 2019

@author: wenke_yang
"""
import math

def intersect_headtail(A,B):
    if len(A) == 0 or len(B) == 0:
        return []
    elif len(A) == 1 and len(B) == 1:
        if A == B:
            return A
        else:
            return []
    else:
        head_A = [A[0]]
        head_B = [B[0]]
        tail_A = A[1:]
        tail_B = B[1:]
        
        return intersect(head_A, head_B) + \
            intersect(head_A, tail_B) + \
            intersect(head_B, tail_A) + \
            intersect(tail_A, tail_B)
            
def intersect(A, B):
    if len(A) == 0 or len(B) == 0:
        return []
    elif len(A) == 1 and len(B) == 1:
        if A == B:
            return A
        else:
            return []
    else:
        half_len_A = math.floor(len(A)/2)
        half_len_B = math.floor(len(B)/2)
        
        first_half_A = A[:half_len_A]
        first_half_B = B[:half_len_B]
        
        second_half_A = A[half_len_A:]
        second_half_B = B[half_len_B:]
        
        return intersect(first_half_A, first_half_B) + \
            intersect(first_half_A, second_half_B) + \
            intersect(first_half_B, second_half_A) + \
            intersect(second_half_A, second_half_B)
            
def divide_list(A, B, k):
    if k > min(len(A), len(B)):
        print('k is larger than list size!')
        return False
    elif k == 1:
        return [A], [B]
    else:
        half_len_A = math.floor(len(A)/2)
        half_len_B = math.floor(len(B)/2)
                
        first_half_A = A[:half_len_A]
        first_half_B = B[:half_len_B]
        
        second_half_A = A[half_len_A:]
        second_half_B = B[half_len_B:]
        
        A1, B1 = divide_list(first_half_A, first_half_B, 
                             math.floor(k/2))
        A2, B2 = divide_list(second_half_A, second_half_B, 
                             k - math.floor(k/2))
        return A1 + A2, B1 + B2
        
    
    
a = [1,2,3,5,7,4]
b = [3,2,4,7,6]
print(intersect(a,b))
print(divide_list(a,b,3))