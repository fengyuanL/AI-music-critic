### AI-music-critic
What makes a “Brilliant” Classical Music Piece: an AI classical piano music critic <br>

Acknowledge: this project if forked from https://github.com/wazenmai/MIDI-BERT <br>
Note: all the files are still in developing stage and some functions in the original repo may not be available here. <br>

## Preprocess data:
Get the files and set things up: data_creation/prepare_data/main.py <br>
Process each file: data_creation/prepare_data/model.py <br>
command to preprocess data: <br>
python3 data_creation/prepare_data/main.py --task acclaimed --dataset GMP_1960 --name GMP_1960 <br>
python3 data_creation/prepare_data/main.py --task acclaimed --dataset GMP_Post1960 --name GMP_Post1960 <br>

## Pretrain:
Load the data, setup the model, iterate all epochs: MidiBERT/main.py <br> 
Perform the training / validation / testing: MidiBERT/trainer.py <br>
The model: MidiBERT/model.py <br>
command to pretrain: <br>
python3 MidiBERT/main.py --datasets GMP_1960 --name Pretrain1960 --batch_size 2 --epochs 30 --accum_grad 16 --cuda_devices 0 <br>

## Finetune:
Get dataset, load pretrained model, setup finetune model: MidiBERT/finetune.py <br>
Perform the training / validation / testing for finetune: MidiBERT/trainer.py The file should only call functions start with reg_ for regression tasks. <br>
The finetune model: MidiBERT/finetune_model.py <br>
command to finetune: <br>
python3 MidiBERT/finetune.py --task acclaimed --name Acclaimed1960 --ckpt result/pretrain/Pretrain1960/model_best.ckpt --datasets GMP_1960 --num_workers 8 --class_num 2 --batch_size 16 --epochs 30 --cuda_devices 0 <br>
