for entry in "data/ASMR/orig_asmr_by_material"/*.mp4
do
	echo "$entry"
	python video-splitter/ffmpeg-split.py -f $entry -v libx264 -s 10 
done
