#!/bin/bash
cd ../../
python -m code.1layerRBM.1layerRBM 1 1 100 cd
python -m code.1layerRBM.1layerRBM 1 2 20 cd
python -m code.1layerRBM.1layerRBM 1 1000 1000 cd

for file in /home/liuq/apt/2ndYear/sDBN/theta/*.npy
do
    echo $file
    python -m code.1layerRBM.recon_error $file >> log.txt
done
        
#for file in /home/liuq/apt/2ndYear/sDBN/theta/5420_b1000_epoc10*.npy
#do
#    echo $file
#    python -m code.1layerRBM.recon_error $file >> log.txt
#done
