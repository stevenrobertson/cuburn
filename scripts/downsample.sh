#!/bin/zsh

encode () {
    [ ! -e "${dir}/ref.avi" ] && return
    [ -e "downsampled/${dir}.ts" -a \
         "downsampled/${dir}.ts" -nt "${dir}/ref.avi" ] && return
    ffmpeg -y -i $dir/ref.avi -vcodec libx264 -q 20 -g 360 -x264opts \
crf=20:bluray-compat:aud:ref=4:level=4.1:threads=0:scenecut=-1:\
vbv_maxrate=40000:vbv_bufsize=30000:vbv_init=0.95:nr=100 \
         -preset slow -profile high -vf scale=-1:540 /tmp/enc.avi && \
    gst-launch-0.10 filesrc location=/tmp/enc.avi ! avidemux ! h264parse \
        ! mpegtsmux ! filesink location=downsampled/${dir}.ts && rm /tmp/enc.avi
}

for dir in `ls --sort time -r | grep -v '.sh'`; do encode $dir || break; done
