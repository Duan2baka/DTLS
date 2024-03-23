## Dataset
To prepare FFHQ dataset, you can follow: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

## Training
Follow the command lines below

**Mapper (1024x1 --> 32x32)**
```
python mapper.py --mode train --lr_size 32 --train_steps 100001 --save_folder 'DTLS_mapper' --data_path 'your_dataset_directory' --batch_size 32
```

**DTLS (16 --> 128)**
```
python main_128.py --mode train --hr_size 128 --lr_size 16 --stride 4 --train_steps 100001 --save_folder 'DTLS_16_128' --data_path 'your_dataset_directory' --batch_size 16
```



## Evaluation

Follow the command lines below

**Generate 32x32 face images by mapper**
```
python mapper.py --mode eval --hr_size 32 --lr_size 32 --save_folder 'fake_faces' --data_path 'your_dataset_directory' --load_path 'DTLS_mapper/model_last.pt' --n_samples 30
```

**DTLS 16 --> 128**
```
python main.py --mode eval --hr_size 128 --lr_size 16 --load_path 'DTLS_16_128/model_last.pt' --save_folder 'DTLS_16_128_results' --input_image 'fake_faces'
```



python main_128.py --mode eval --hr_size 128 --lr_si
ze 16 --load_path ./pre_trained_weight/DTLS_128.pt --save_folder '128_16_128' --input_ima
ge ../restyle-encoder/unet/result_16/128_10/