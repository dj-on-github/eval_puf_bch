#!/usr/bin/env python3

import sys

import gmpy2
from gmpy2 import mpfr
import math

from incomplete_beta import *
from binomial_quantile import *

precision = 300
gmpy2.get_context().precision=precision

# blksize,msg_len,correctable_bits
# n,k,t
bch_codes=[(7,4,1),(15,11,1),(15,7,2),(15,5,3),(31,26,1),(31,21,2),(31,16,3),(31,11,5),(31,6,7),(63,57,1),
(63,51,2),(63,45,3),(63,39,4),(63,36,5),(63,30,6),(63,24,7),(63,18,10),(63,16,11),(63,10,13),(63,7,15),
(127,120,1),(127,113,2),(127,106,3),(127,99,4),(127,92,5),(127,85,6),(127,78,7),(127,71,9),(127,64,10),(127,57,11),
(127,50,13),(127,43,14),(127,36,15),(127,29,21),(127,22,23),(127,15,27),(127,8,31),(255,247,1),(255,239,2),(255,231,3),
(255,223,4),(255,215,5),(255,207,6),(255,199,7),(255,191,8),(255,187,9),(255,179,10),(255,171,11),(255,163,12),(255,155,13),
(255,147,14),(255,139,15),(255,131,18),(255,123,19),(255,115,21),(255,107,22),(255,99,23),(255,91,25),(255,87,26),(255,79,27),
(255,71,29),(255,63,30),(255,55,31),(255,47,42),(255,45,43),(255,37,45),(255,29,47),(255,21,55),(255,13,59),(255,9,63),
(511,502,1),(511,493,2),(511,484,3),(511,475,4),(511,466,5),(511,457,6),(511,448,7),(511,439,8),(511,430,9),(511,421,10),
(511,412,11),(511,403,12),(511,394,13),(511,385,14),(511,376,15),(511,367,17),(511,358,18),(511,349,19),(511,340,20),(511,331,21),
(511,322,22),(511,313,23),(511,304,25),(511,295,26),(511,286,27),(511,277,28),(511,268,29),(511,259,30),(511,250,31),(511,241,36),
(511,238,37),(511,229,38),(511,220,39),(511,211,41),(511,202,42),(511,193,43),(511,184,45),(511,175,46),(511,166,47),(511,157,51),
(511,148,53),(511,139,54),(511,130,55),(511,121,58),(511,112,59),(511,103,61),(511,94,62),(511,85,63),(511,76,85),(511,67,87),
(511,58,91),(511,49,93),(511,40,95),(511,31,109),(511,28,111),(511,19,119),(511,10,127),(1023,1013,1),(1023,1003,2),(1023,993,3),
(1023,983,4),(1023,973,5),(1023,963,6),(1023,953,7),(1023,943,8),(1023,933,9),(1023,923,10),(1023,913,11),(1023,903,12),(1023,893,13),
(1023,883,14),(1023,873,15),(1023,863,16),(1023,858,17),(1023,848,18),(1023,838,19),(1023,828,20),(1023,818,21),(1023,808,22),(1023,798,23),
(1023,788,24),(1023,778,25),(1023,768,26),(1023,758,27),(1023,748,28),(1023,738,29),(1023,728,30),(1023,718,31),(1023,708,34),(1023,698,35),
(1023,688,36),(1023,678,37),(1023,668,38),(1023,658,39),(1023,648,41),(1023,638,42),(1023,628,43),(1023,618,44),(1023,608,45),(1023,598,46),
(1023,588,47),(1023,578,49),(1023,573,50),(1023,563,51),(1023,553,52),(1023,543,53),(1023,533,54),(1023,523,55),(1023,513,57),(1023,503,58),
(1023,493,59),(1023,483,60),(1023,473,61),(1023,463,62),(1023,453,63),(1023,443,73),(1023,433,74),(1023,423,75),(1023,413,77),(1023,403,78),
(1023,393,79),(1023,383,82),(1023,378,83),(1023,368,85),(1023,358,86),(1023,348,87),(1023,338,89),(1023,328,90),(1023,318,91),(1023,308,93),
(1023,298,94),(1023,288,95),(1023,278,102),(1023,268,103),(1023,258,106),(1023,248,107),(1023,238,109),(1023,228,110),(1023,218,111),(1023,208,115),
(1023,203,117),(1023,193,118),(1023,183,119),(1023,173,122),(1023,163,123),(1023,153,125),(1023,143,126),(1023,133,127),(1023,123,170),(1023,121,171),
(1023,111,173),(1023,101,175),(1023,91,181),(1023,86,183),(1023,76,187),(1023,66,189),(1023,56,191),(1023,46,219),(1023,36,223),(1023,26,239),
(1023,16,247),(1023,11,255)]



def maes_entropy_remaining(n,k,t,hp):
    entropy = k-(n*(1.0-hp))
    return entropy

def dodis_entropy_remaining(n,k,t,hp):
    entropy = k*hp
    return entropy

def number_of_bches(array_size,n):
    return math.floor(int(array_size)/int(n))

def fail_prob(n,k,t,ber,block_count):
    block_fail_prob = mpfr("1.0") - mpfr(BCDF(mpfr(n),mpfr(t),mpfr(ber)))
    array_fail_prob = mpfr("1.0") - mpfr(BCDF(mpfr(block_count),0,block_fail_prob))
    return block_fail_prob,array_fail_prob

array_sizes = [mpfr((n+1)*1024) for n in range(8)] # array sizes multiple of 1024.
mean_bers = [(mpfr(x+1.0)/100.0) for x in range(16)] # Check failure rates for list of BER rates
hdm_remaining = mpfr(0.8) # assume 20% dark mask
hdm_targeting_factor=0.8 # proportion of unstable bits hit bu HDM
hp = mpfr(0.9) # assume 0.9 raw entropy rate

def post_hdm_ber(hdm_remaining, ber, array_size,hdm_targeting_factor):
    post_hdm_bits = math.ceil(hdm_remaining*array_size)
    
    unstable_bits = math.ceil(ber*array_size)
    found_unstable_bits = math.floor(unstable_bits*hdm_targeting_factor)
    #misaimed_unstable_bits = (array_size-post_hdm_bits)-found_unstable_bits

    new_ber = (unstable_bits-found_unstable_bits)/array_size

    return new_ber
    
print("n,k,t,check_bits,percent_correction,array_size,ber,hdm_ber,block_count,block_fail_prob,array_fail_prob,maes_ent, maes_ent_rate, dodis_ent, dodis_ent_rate, total_maes_ent, total_dodis_ent")
for array_size in array_sizes:
    for bch in bch_codes:
        for ber in mean_bers:
            (tn,tk,tt) = bch
            n = mpfr(tn)
            k = mpfr(tk)
            t = mpfr(tt)
            cbits = n - k
            
            dodis_ent = dodis_entropy_remaining(n,k,t,hp)
            maes_ent = maes_entropy_remaining(n,k,t,hp)
           
            if maes_ent < dodis_ent:
                maes_ent = dodis_ent

            block_count = number_of_bches(array_size,n)

            total_maes_ent = block_count * maes_ent * hdm_remaining
            total_dodis_ent = block_count * dodis_ent * hdm_remaining
            
            maes_ent_rate = total_maes_ent/(array_size*block_count)
            dodis_ent_rate = total_dodis_ent/(array_size*block_count)
            worst_ent_rate = min(dodis_ent_rate,maes_ent_rate)

            hdm_ber = post_hdm_ber(hdm_remaining,ber,array_size,hdm_targeting_factor)
            bfp,afp = fail_prob(n,k,t,hdm_ber,block_count)
            
            percent_correction = 100.0*(t/n)
            #if (ber>0.155) and (afp < 0.000001) and (worst_ent_rate > 0.16):
            #    print(f"{int(n)},{int(k)},{int(t)},{int(cbits)},{percent_correction:0.2f}%,{int(array_size)},{float(ber):0.4f},{float(hdm_ber):0.4f}, {int(block_count)},{bfp:0.12f},{afp:0.12f},{maes_ent:0.4f}, {maes_ent_rate:0.4f},{dodis_ent:0.4f},{dodis_ent_rate:0.4f},{total_maes_ent:0.2f},{total_dodis_ent:0.2f}")
            print(f"{int(n)},{int(k)},{int(t)},{int(cbits)},{percent_correction:0.2f}%,{int(array_size)},{float(ber):0.4f},{float(hdm_ber):0.4f}, {int(block_count)},{bfp:0.12f},{afp:0.12f},{maes_ent:0.4f}, {maes_ent_rate:0.4f},{dodis_ent:0.4f},{dodis_ent_rate:0.4f},{total_maes_ent:0.2f},{total_dodis_ent:0.2f}")
            
            
            

