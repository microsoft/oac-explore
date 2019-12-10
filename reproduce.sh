#!/usr/bin/env bash

# RUN OAC
for ((i=0;i<5;i+=1))
do
    python main.py --seed=$i --domain=humanoid --beta_UB=4.66 --delta=23.53 &
done

# RUN SAC
for ((i=0;i<5;i+=1))
do
    python main.py --seed=$i --domain=humanoid &
done