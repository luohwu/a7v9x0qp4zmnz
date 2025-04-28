# UltraBones100k

Welcome to the **UltraBones100k Repository**!

This repository contains:
1. **Dataset Access**: Instructions for downloading the UltraBones100k dataset.
2. **Bone Segmentation in Ultrasound Images**: Code and pretrained models for accurate bone segmentation in ultrasound imaging.
3. **CT-US Alignment Optimization**: Code for optimizing the alignment between CT and ultrasound images.


# Requirements 
Run the following command to install all the packages listed in the `requirements.txt` file: 
```pip install -r requirements.txt```

The code has been tested on the following setup:

- **Operating System**: Windows 10
- **Python Version**: 3.10.0
- **CUDA Version**: 12.1
- **PyTorch Version**: 2.4.0
- **Processor**: Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz
- **GPU**: NVIDIA GeForce RTX 3060 (12GB) and NVIDIA V100

# Dataset downloading
1. Download Azure Storage Explorer [here](https://azure.microsoft.com/en-us/products/storage/storage-explorer).
2. Double-click on the downloaded file and start the installation.
3. On the main page, select **"Connect to resource"**.
4. Select "Storage account"
5. Select "Connection string", then press Next
6. Paste given URL (including BlobEndpoint=) into "Connection String" field and under  "Display name" write a wished name for the storage. This name is defined only for  you on your local machine and doesn't affect the storage itself
7. In the next page select "Connect"
8. By selecting the storage account you have named in step 6, then selecting "Blob  
containers", you will find the shared drive

The URL: BlobEndpoint=[https://rocs3.blob.core.windows.net/;QueueEndpoint=https://rocs3.queue.core.windows.net/;FileEndpoint=https://rocs3.file.core.windows.net/;TableEndpoint=https://rocs3.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2034-12-19T23:42:28Z&st=2024-12-19T15:42:28Z&spr=https&sig=KWLVjUi%2BBh2FA%2B6VAfUIUBlgQRz7yaQrduCSSBdVs0g%3D](https://rocs3.blob.core.windows.net/;QueueEndpoint=https://rocs3.queue.core.windows.net/;FileEndpoint=https://rocs3.file.core.windows.net/;TableEndpoint=https://rocs3.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2034-12-19T23:42:28Z&st=2024-12-19T15:42:28Z&spr=https&sig=KWLVjUi%2BBh2FA%2B6VAfUIUBlgQRz7yaQrduCSSBdVs0g%3D "https://rocs3.blob.core.windows.net/;queueendpoint=https://rocs3.queue.core.windows.net/;fileendpoint=https://rocs3.file.core.windows.net/;tableendpoint=https://rocs3.table.core.windows.net/;sharedaccesssignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2034-12-19t23:42:28z&st=2024-12-19t15:42:28z&spr=https&sig=kwlvjui%2bbh2fa%2b6vafuiublgqrz7yaqrducssbdvs0g%3d")


## Dataset File Structure

The dataset is organized as follows:

- **Root Folder**: `AI_Ultrasound_dataset` is the main directory.
- **Specimen Folders**: Each specimen folder (e.g., `cadaver01_F231091`, `cadaver02_F231218`) contains data for one specimen.
- **Record Folders**: Each specimen folder has subfolders for each record (e.g., `record01`, `record02`).
- **Labels**: Contains the label files for that record.
- **UltrasoundImages**: Contains the ultrasound image files for that record.

```
XXX:\AI_ULTRASOUND_DATASET
├───cadaver01_F231091
│   ├───record01
│   │   ├───UltrasoundImages
│   │   └───Labels
│   │   └───tracking.csv
│   ├───record02
│   │   ├───UltrasoundImages
│   │   └───Labels
│   │   └───tracking.csv
│   ├───record03
│   │   ├───UltrasoundImages
│   │   └───Labels
│   │   └───tracking.csv
│   ⋮
│   ⋮
├───cadaver02_F231218
├───cadaver03_S231783
⋮
⋮
```


# Train the Bone Segmentation model 
The training script is located at:
```AI_ultrasound_segmentation/train_lightning.py ```
Train the model using one NVIDIA V100 for 100 epochs, which typically takes around 10 hours. The training process leverages a ResNet-34 FPN architecture with a combination of DICE and BCE losses, and a learning rate of 1e-05.
By default, we assume the dataset folder located at ```./AI_ultrasound_segmentation/data/AI_Ultrasound_dataset/```

To train the model, just run `python AI_ultrasound_segmentation/train_lightning.py`


# Pretrained model
The pretrained model (trained on specimens [1,3,4,5,6,9,10,11,12,13,14]) is available at 
```AI_ultrasound_segmentation/models/train_on_1_3_4_5_6_9_10_11_12_13_14/epoch_100.pth```


# Evaluation
To quantitatively evaluate a trained model on specimens [2,7,8], run:

```python AI_ultrasound_segmentation/evaluation.py```


To qualitatively evaluate a trained model on some example ultrasound images, a notebook is available at:

```AI_ultrasound_segmentation/segment_example_images.ipynb```



# Running the CT-US Bone Alignment Optimization Pipeline
To run the optimization pipeline: `python CT-US bone alignment optimization/optimization_main.py`, example data are provided `CT-US bone alignment optimization/record_folder`. Given an ultrasound image, CT models and inital tracking data, the code output the initial bone alignment and optimized alignment under `CT-US bone alignment optimization/record_folder/optimization_results`
