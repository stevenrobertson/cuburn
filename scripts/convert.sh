#!/bin/zsh

convert () {
    dir=$1
    [ -e $dir/NFRAMES ] || return
    [ -e $dir/COMPLETE -a -e ref/$dir.ts ] && return
    NFRAMES="$(cat $dir/NFRAMES)"
    for i in `seq 1 $NFRAMES`
        [ -e $dir/$(printf '%05d' $i).jpg ] || return
    mencoder -mf fps=23.976 "mf://$dir/*.jpg" -ovc x264  \
        -x264encopts profile=high:bluray-compat:aud:crf=15:ref=4:level=4.1:threads=0 \
        -o $dir/ref.avi && \
    gst-launch-0.10 filesrc location=${dir}/ref.avi ! avidemux ! h264parse \
        ! mpegtsmux ! filesink location=ref/${dir}.ts && rm ${dir}/ref.avi && \
    touch $dir/COMPLETE
}
for dir in `ls --sort time -r | grep -v '.sh'`; do echo $dir; convert $dir; sleep 0.1 || break; done
