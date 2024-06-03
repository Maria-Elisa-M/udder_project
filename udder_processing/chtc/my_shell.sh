#!/bin/bash
num=$1
#
# First, copy the compressed tar file from /staging into the working directory,
#  and un-tar it to reveal your large input file(s) or directories:
mv /staging/groups/dorea_group/maria/$num.zip ./$num.zip
unzip $num.zip -d $num
rm $num.zip
# zip -xzvf $num.zip # this is for unziping (this is untar)
#
# Command for myprogram, which will use files from the working directory
python3 my_script_v2.py $num
#
# Before the script exits, make sure to remove the file(s) from the working directory
tar -czf $num.tar.gz $num
cp -r $num.tar.gz /staging/groups/dorea_group/maria/
rm -r $num
rm $num.tar.gz
#
# END