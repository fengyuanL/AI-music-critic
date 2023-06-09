# AI-music-critic
What makes a “Brilliant” Classical Music Piece: an AI classical piano music critic <br>

Acknowledge: this project if forked from https://github.com/wazenmai/MIDI-BERT (folders MidiBERT, data_creations) and modified for our own tasks.<br>
Note: all the files are still in developing stage and some functions in the original repo may not be available here. <br>
Files may include different absolute paths or relative path as the code are runned on different machines. 

## Data collection:
MidiFiles: https://github.com/bytedance/GiantMIDI-Piano <br>
Create the dictionary from the midifiles: DataGather.ipynb <br>
Gather popularity index from Spotify: GetPopularity folder <br>
Scrape the year of composition: get_composition folder <br>

## Preprocess data:
Get the files and set things up: data_creation/prepare_data/main.py <br>
Process each file: data_creation/prepare_data/model.py <br>
command to preprocess data: <br>
python3 data_creation/prepare_data/main.py --task acclaimed --dataset GMP_1960 --name GMP_1960 <br>
python3 data_creation/prepare_data/main.py --task acclaimed --dataset GMP_Post1960 --name GMP_Post1960 <br>
Rescale and merge the data: merge.py <br>

## Pretrain:
Load the data, setup the model, iterate all epochs: MidiBERT/main.py <br> 
Perform the training / validation / testing: MidiBERT/trainer.py <br>
The model: MidiBERT/model.py <br>
command to pretrain: <br>
python3 MidiBERT/main.py --datasets GMP_1960 --name Pretrain1960 --batch_size 2 --epochs 30 --accum_grad 16 --cuda_devices 0 <br>
Codes are modified to take in our new dataset, implement accumulated gradient, learning rate warmup, load from previous training.

## Finetune:
Get dataset, load pretrained model, setup finetune model: MidiBERT/finetune.py <br>
Perform the training / validation / testing for finetune: MidiBERT/trainer.py The file should only call newly added functions start with reg_ for our regression tasks. <br>
The finetune model: MidiBERT/finetune_model.py The function should only call the newly added SequenceRegression <br>
command to finetune: <br>
python3 MidiBERT/finetune.py --task acclaimed --name Acclaimed1960 --ckpt result/pretrain/Pretrain1960/model_best.ckpt --datasets GMP_1960 --num_workers 8 --class_num 2 --batch_size 16 --epochs 30 --cuda_devices 0 <br>

## Ensemble and evaluation:
Main function to call ensemble for MidiBERT/Ensemble.py <br>
Collect data with augmentation and call the model: MidiBERT/EnsembleCP.py <br>
Execute evaluation: MidiBERT/EnsembleFT.py <br>

## Analysis:
Compute the ensembled results: Ensemble/ compute accuracy and loss for dataset of each time node <br>
T-test and p-values: Entropy and t_test/t_test.py <br>
Analysis results: entropy caluclation: Entropy and t_test/entropy.py <br>

If you are interested in the processed datasets, trained models, please contact the author directly.
