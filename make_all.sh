#! /bin/bash
# Helper script to remake all the images from input sources

find inputs -name '*.json' | sed -e 's/^inputs//g' | sed -e 's/.json//g' | xargs -I {} cargo run --release inputs{}.json images{}.png

ANIMATIONS=`find images -name '1.png' | sed -e 's/\/1.png//g'`
for folder in $ANIMATIONS 
do
  cd $folder
  rm out.mp4
  ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4
  rm *.png
  cd -
done