#!/bin/bash
query_cn=`expr $RANDOM % 10 + 1`
query_size=`expr $RANDOM % 128 + 1`
[ -n "$1" ] && query_cn=$1
[ -n "$2" ] && query_size=$2

mkdir -p testdata/query
rm -rf ./testdata/query/*

for((i=1;i<=$query_cn;i++)); do
    bash gen.sh 1 $query_size query/query$i.txt
done
