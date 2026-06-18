*Ouvrir anaconda prompt:*



(base) C:\\Users\\Bicel service>**cd .\\Desktop\\USER\\Maria**



(base) C:\\Users\\Bicel service\\Desktop\\USER\\Maria>**conda activate sam\_env**



---------------------------------------------------------------------------------------



*Pour lancer une segmentation cellpose:*



(sam\_env) C:\\Users\\Bicel service\\Desktop\\USER\\Maria>**cd .\\codes**



(sam\_env) C:\\Users\\Bicel service\\Desktop\\USER\\Maria>**python seg\_cellpose.py**



---------------------------------------------------------------------------------------



*Pour lancer une segmentation avec modèle sam2 officiel:*



1. Autosegmentation mode



(sam\_env) C:\\Users\\Bicel service\\Desktop\\USER\\Maria>**cd .\\codes\\autoseg\_SAM2**



(sam\_env) C:\\Users\\Bicel service\\Desktop\\USER\\Maria>**python autoseg.py**





2\. Predictor mode 



(sam\_env) C:\\Users\\Bicel service\\Desktop\\USER\\Maria>**cd .\\codes\\predictor\_SAM2**



(sam\_env) C:\\Users\\Bicel service\\Desktop\\USER\\Maria>**python predictor\_before\_training.py**



---------------------------------------------------------------------------------------



*Pour lancer une segmentation avec modèle sam2 finetuné:*



(sam\_env) C:\\Users\\Bicel service\\Desktop\\USER\\Maria>**cd .\\codes\\predictor\_SAM2**



(sam\_env) C:\\Users\\Bicel service\\Desktop\\USER\\Maria>**python predictor\_finetuned.py**



---------------------------------------------------------------------------------------



*Pour finetuner sam2 :*



(sam\_env) C:\\Users\\Bicel service\\Desktop\\USER\\Maria>**cd .\\sam2**



(sam\_env) C:\\Users\\Bicel service\\Desktop\\USER\\Maria\\sam2>**python -m sam2.training.train -c sam2.1\_training/sam2.1\_hiera\_b+\_blanc\_finetune --use-cluster 0 --num-gpus 1**



\#-> sam2.1\_training/sam2.1\_hiera\_b+\_blanc\_finetune est le chemin vers le yaml indiquant le chemin vers le dataset (images et masques)

