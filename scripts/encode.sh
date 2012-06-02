#!/bin/zsh

encode () {
    echo $dir unpackt/${dir}-00.idx
    [ ! -e "${dir}/ref.avi" ] && return
    [ -e "unpackt/${dir}-00.idx" -a \
         "unpackt/${dir}-00.idx" -nt "${dir}/ref.avi" ] && return
    echo $dir unpackt/${dir}-00.idx
    ffmpeg -y -i $dir/ref.avi -vcodec libx264 -q 20 -g 360 -x264opts \
crf=21:bluray-compat:aud:ref=4:level=4.1:threads=0:scenecut=-1:\
vbv_maxrate=40000:vbv_bufsize=30000:vbv_init=0.95:nr=100 \
         -preset slow -profile high /tmp/enc.avi && \
    gst-launch-0.10 filesrc location=/tmp/enc.avi ! avidemux ! h264parse \
        ! mpegtsmux ! filesink location=unpackt/${dir}.ts && rm /tmp/enc.avi #&& \
    #ts_split /tmp/${dir}.ts unpackt/ && rm /tmp/${dir}.ts
}

for dir in `ls --sort time -r | grep -v '.sh'`; do encode $dir || break; done

