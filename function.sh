#!/bin/bash
# unix KISS :)

# file line format : 1, 2, 3
# need install gawk: brew install gawk; apt-get install gawk; yum install gawk
function sort_line_with_gawk_asort(){
    gawk '
    {
        split($0,ary,", ");
        # sort ary into ascending order
        asort(ary);
        for(i=1; i<=length(ary); i++){
            if(i==length(ary)){
                printf("%s",ary[i]);
            }else{
                printf("%s, ",ary[i]);
            }
        } 
        printf("\n"); 
    }
    '
}

# file line format : 1, 2, 3
function sort_line_with_awk_bubble(){
    awk '
    # sort ary into ascending order
    function bubble(ary){
        for(i=1; i<=length(ary); i++){
            for(j=i+1; j<=length(ary); j++){
                if(ary[i]>ary[j]){
                    tmp=ary[i];
                    ary[i] = ary[j];
                    ary[j] = tmp;
                }
            }
        }
    }

    {
        split($0, ary, ", ");    
        bubble(ary); 
        for(i=1; i<=length(ary); i++){
            if(i==length(ary)){
                printf("%s",ary[i]);
            }else{
                printf("%s, ",ary[i]);
            }
        }
        printf("\n"); 
    }
    '
}

# file line format : 1, 2, 3
function sort_line_with_bubble(){
    local ary i j
    ary=($(echo $@ | sed 's/,//g'))
    for ((i=0; i<${#ary[@]}; i++)); do
       for ((j=i+1; j<${#ary[@]}; j++)); do
            if (( ary[i] > ary[j] )); then
                  ((key=ary[i]))
                  ((ary[i]=ary[j]))
                  ((ary[j]=key))
            fi
       done
    done
    echo ${ary[@]}
}

function sort_file_with_bubble(){
    while read line; do
        line=`echo $@ | sed 's/,//g'`
        ary_line=($line)
        #echo "原始输入数组为: ${ary_line[*]}"
        arg=$(echo ${ary_line[*]})
        #echo "新组成的输入数组arg为: ${arg[*]}"
        result=($(sort_array_with_bubble $arg))
        #echo "函数返回数组为: ${result[*]}"  
        echo ${result[*]}
    done < ${1}
}