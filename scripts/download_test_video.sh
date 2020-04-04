
# please run this at the top level of this repository
# Download a test video
# about 7 MB

mkdir -p test/
wget https://next.cs.cmu.edu/data/VIRAT_S_000008.short.mp4 -O test/test_video.mp4

# make a video list
cd test/
ls $PWD/*.mp4 > test_videos.lst
cd ..

