#!/bin/zsh
set -x
encode () {
    f=$1
    [ -e "enc/$f" -a "enc/$f" -nt "ref/$f" ] && return
    flock /tmp/enc.lock \
        ffmpeg -y -i ref/$f -vcodec libx264 -q 18 -g 360 -x264opts \
        crf=18:threads=0:nocabac -profile high /tmp/enc.avi && \
    gst-launch-0.10 filesrc location=/tmp/enc.avi ! avidemux ! h264parse \
        ! mpegtsmux ! filesink location=enc/$f && rm /tmp/enc.avi #&& \
}

mkdir -p enc
for f in `ls --sort time -r ref/`; do encode $f || break; done
