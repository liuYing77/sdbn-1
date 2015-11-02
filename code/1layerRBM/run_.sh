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
        
        
#echo '--Finish testing with' $sim_test >> log.txt
#date >> log.txt
#python test_analysis.py $num_cluster $sim_train $sim_test $dur $sum_rate>> log.txt
