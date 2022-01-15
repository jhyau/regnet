for f in $(cat ckpt/$exp/multi_sample_mel.txt); do
    name=$(basename $f | sed 's/.npy//g');
    video=$(find data/features/ASMR/asmr_both_vids/videos_10s/ -name "$name*");
    cmd="ffmpeg -y -i $video -i ckpt/$exp/${name}_synthesis.wav -c:v copy -map 0:v:0 -map 1:a:0 ckpt/$exp/$name.newaudio.mp4";
    #echo $cmd;
    $cmd;
done
