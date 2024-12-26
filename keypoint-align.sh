#!/bin/zsh
source venv/bin/activate

OMP_NUM_THREADS=2 python keypoint-aligner.py  \
    --image_folder test_images \
    --reference_image ../test_images_aligned/IMG_20241023_122517.jpg \
    --step_counter Pedometer_Backup.txt \
    --crop_width 3000 \
    --crop_height 1800 \
    --audio one_fine_day.mp3 \
    --stop_motion parkbridge.mp4



# 1    --reference_image IMG_20240529_073834.jpg \
