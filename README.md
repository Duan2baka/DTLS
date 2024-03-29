## Dataset
To prepare FFHQ dataset, you can follow: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

## Training
Follow the command lines below

**DTLS (16 --> 128)**
```
python main.py --mode train --hr_size 128 --lr_size 16 --stride 4 --train_steps 100001 --save_folder 'DTLS_16_128' --data_path 'your_dataset_directory' --batch_size 16
```


## Evaluation

Follow the command lines below


**DTLS 16 --> 128**
```
python main.py --mode eval --hr_size 128 --lr_size 16 --load_path 'pretrained_weight/DTLS_128.pt' --save_folder 'DTLS_16_128_results' --input_image 'your_images_folder'
```

my train
python main.py --mode train --hr_size 128 --lr_size 16 --stride 4 --train_steps 100001 --save_folder 'DTLS_16_128' --data_path ./training_set/ --batch_size 16

python main_smiling.py --mode train --hr_size 128 --lr_size 16 --stride 4 --train_steps 100001 --save_folder 'DTLS_smiling' --data_path ./fake_dataset_128/ --batch_size 16