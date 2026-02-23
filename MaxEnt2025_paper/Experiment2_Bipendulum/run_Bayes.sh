#!/bin/bash

for n in 5 7 10 25 50; do
  for s in 0.0 0.01 0.1 0.5; do
    python Ex2_Bayes.py $n $s 6
  done  
done
for n in 5 7; do
  for s in 0.0 0.01 0.1 0.5; do
    python Ex2_Bayes.py $n $s 2
  done  
done
