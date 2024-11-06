#### Training
```python MSVD
python train.py --exp_name=MSVD-train --save_memory_mode --dataset_name=MSVD --num_epochs=5 --log_step=10 --evals_per_epoch=1 --batch_size=8 --num_workers=8 --videos_dir=MSVD/videos/ --noclip_lr=1e-5  
```

```python MSRVTT-ViT-B/32
python train.py --exp_name=MSRVTT-train --save_memory_mode --dataset_name=MSRVTT  --clip_arch=ViT-B/32  --num_epochs=5 --log_step=50 --evals_per_epoch=5 --batch_size=32 --num_workers=8 --videos_dir=MSRVTT/videos/ --noclip_lr=1e-5  
```

```python MSRVTT-ViT-B/16
python train.py --exp_name=MSRVTT-train --save_memory_mode --dataset_name=MSRVTT  --clip_arch=ViT-B/16 --num_epochs=5 --log_step=50 --evals_per_epoch=5 --batch_size=32 --num_workers=8 --videos_dir=MSRVTT/videos/ --noclip_lr=1e-5  
```

```python DiDeMo
python train.py --exp_name=DiDeMo-train --save_memory_mode --dataset_name=DiDeMo --num_epochs=5 --log_step=50 --num_frames=12 --evals_per_epoch=1 --batch_size=8 --num_workers=0 --videos_dir=DiDeMo/videos/ --noclip_lr=1e-5  
```

```python Charades
python train.py --exp_name=Charades-train --save_memory_mode --dataset_name=Charades --num_epochs=5 --num_frames=12 --log_step=10 --evals_per_epoch=1 --batch_size=8 --num_workers=8 --videos_dir=Charades/videos/ --noclip_lr=1e-5  
```

```python VATEX
python train.py --exp_name=VATEX-train --save_memory_mode --dataset_name=VATEX --num_epochs=5 --num_frames=12 --log_step=10 --evals_per_epoch=1 --batch_size=32 --num_workers=8 --videos_dir=VATEX/videos/ --noclip_lr=1e-5  
```

```python LSMDC
python train.py --exp_name=LSMDC-train --save_memory_mode --dataset_name=LSMDC --num_epochs=5 --num_frames=12 --log_step=10 --evals_per_epoch=1 --batch_size=32 --num_workers=8 --videos_dir=LSMDC --noclip_lr=1e-5  
```

#### Testing
```python
python test.py --exp_name=MSVD-test --save_memory_mode --dataset_name=MSVD --batch_size=32 --num_workers=0 --videos_dir=MSVD/videos/ --noclip_lr=1e-5 --load_epoch=0 --datetime=test
```
