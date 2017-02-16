#!/bin/bash
DIVIDER=1
TRSIZE=$(expr 45000 / $DIVIDER)
VASIZE=$(expr 5000 / $DIVIDER)
WARMSTART=0
nohup th adaptive_main.lua -trainPerDev 5 -dataRoot cifar.torch/ -resultFolder results/ -deathRate 0.5 -trsize $TRSIZE -vasize $VASIZE -alphaLR 0.5 -warmStartEpochs $WARMSTART &
