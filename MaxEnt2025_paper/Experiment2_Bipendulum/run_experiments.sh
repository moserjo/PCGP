#!/bin/bash

for n in 2 3 5 7; do
  for s in 0.0 0.01 0.1 0.5; do
    for r in {0..9}; do
    python Experiment2_PIGP/main.py $n $s 6 $r
    done
  done  
done

for n in 2 3 5 7 10 25 50; do
  for s in 0.0 0.01 0.1 0.5; do
    for r in {0..9}; do
    python Experiment2_PIGP/main.py $n $s 2 $r
    done
  done  
done


for n in 2 3 5 7; do
  for s in 0.0 0.01 0.1 0.5; do
    for r in {0..9}; do
    python Experiment2_NSB/main.py $n $s 6 $r
    done
  done  
done

for n in 2 3 5 7 10 25 50; do
  for s in 0.0 0.01 0.1 0.5; do
    for r in {0..9}; do
    python Experiment2_NSB/main.py $n $s 2 $r
    done
  done  
done
