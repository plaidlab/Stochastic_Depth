#!/bin/bash
DIVIDER=1
TRSIZE=$(expr 45000 / $DIVIDER)
VASIZE=$(expr 5000 / $DIVIDER)
WARMSTART=$(expr 10 \* $DIVIDER)
nohup th adaptive_main.lua -dataRoot cifar.torch/ -resultFolder results/ -deathRate 0.5 -trsize $TRSIZE -vasize $VASIZE -trainAlphas false -warmStartEpochs $WARMSTART &
