#!/usr/bin/env bash

for ((i=0;i<5;i+=1))
do
    python main.py --seed=$i --domain=humanoid --beta_UB=4.66 --delta=23.53 &
done