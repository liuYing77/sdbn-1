#!/bin/bash
cd ../../

epoc=5
for i in `seq 0 $epoc 1085`;
do
    python -m code.1layerRBM.cd_k 1 1000 $epoc cdk $i
done

for file in /home/liuq/apt/2ndYear/sDBN/cdk/*.npy
do
    echo $file
    python -m code.1layerRBM.recon_error $file >> log_cdk.txt
done




