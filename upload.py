#! /bin/python3
import os
import argparse


pswd = 'ustc.112233'
ip = 'user@192.168.51.24:'

parser = argparse.ArgumentParser('upload')
# parser.add_argument('-f', '--file', help='files to upload to server', type=str)
parser.add_argument('file', help='files to upload to server', type=str)
parser.add_argument('-d', '--dst', help='destination for the file', type=str)
parser.add_argument('-r', '--recursive', help='upload files recursively', action='store_true')
args = parser.parse_args()

file_name = args.file

dst = args.dst if args.dst else '/mnt/share/debug/zzy'
dst = ip + dst

if args.recursive:
    os.system(f'sshpass -p \'{pswd}\' scp -r {file_name} {dst}')
else:
    print(1)
    os.system(f'sshpass -p \'{pswd}\' scp {file_name} {dst}')
