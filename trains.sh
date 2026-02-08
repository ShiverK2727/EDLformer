python /app/MultiAnn/EDLformer/codes/train_simple_maskformer_v2_edl_seg.py \
    --config_path /app/MultiAnn/EDLformer/codes/configs/simple_maskformer_v2_edl.yaml \
    --gpu 0


python codes/train_simple_maskformer_v2_edl_seg.py \
  --config_path codes/configs/simple_maskformer_v2_edl2.yaml \
  --gpu 0 \
  --save_dir ./exp/my_run2


python codes/train_simple_maskformer_v2_edl_seg.py \
  --config_path codes/configs/simple_maskformer_v2_edl3.yaml \
  --gpu 0 \
  --save_dir ./exp/my_run3


python codes/train_simple_maskformer_v2_edl_seg.py \
  --config_path codes/configs/simple_maskformer_v2_edl4.yaml \
  --gpu 0 \
  --save_dir ./exp/my_run3_test_1 \
  --learning_rate 1e-4

python codes/train_simple_maskformer_v2_edl_seg.py \
  --config_path codes/configs/simple_maskformer_v2_edl4.yaml \
  --gpu 1 \
  --save_dir ./exp/my_run3_test_2 \
  --learning_rate 5e-4


python codes/train_simple_maskformer_v2_edl_seg.py \
  --config_path codes/configs/simple_maskformer_v2_edl5.yaml \
  --gpu 2 \
  --save_dir ./exp/my_run5_test_1 \
  --learning_rate 1e-4

python codes/train_simple_maskformer_v2_edl_seg.py \
  --config_path codes/configs/simple_maskformer_v2_edl5.yaml \
  --gpu 3 \
  --save_dir ./exp/my_run5_test_2 \
  --learning_rate 5e-4