This is a modified version of MPNET, a deep leanring based motion planning model. The modifications allows MPNET to run on python3.8 and plan with Fetch's 7 DoF robotic arm.
Directory Structure of Relavent Files:
├── AE
│   ├── CAE.py 
│   ├── CAE_JM.py
│   ├── data_loader.py
├── data_loader.py
├── infer.py
├── infer_EtE.py
├── model.py
├── train_JM.py
├── train_JM_ETE.py
└── train.py

All files run in python3.8. CAE_JM.py, train_JM.py and infer.py are the autoencoder, model trainer and inference program for fetch respectively; train_JM_EtE.py and infer_EtE.py are the training and inference programs for the end-to-end version of the model. In the original paper, it was not specified whether the end-to-end version of the model's autoencoder portion was trained with contrative loss or only the final L2 loss was used, so in this version only the final L2 loss was used. Please refer to model.py for the full end-to-end model (EMLP).

To run, simply using python3.8 in the container and run the files directly. Note that you will need to provide your own data in order to do so.
