#!/bin/bash

links=(
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0001.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0002.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0003.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0004.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0005.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0006.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0007.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0008.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0009.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0010.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0011.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0012.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0013.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0014.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0015.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0016.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0017.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0018.mp4"
    "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0019.mp4"
)

#for link in ${links[@]}; do
#    wget $link
#done

wget -N --recursive --no-parent -nH --cut-dirs=1 -R "index.html*" https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/ -P /media/kuriatsu/InternalHDD/PIE
