#! /bin/bash

# Count the number of lines from the CSV file
JOBS=$(wc -l $2  | awk '{ print $1 }')

# Parse the parameters from the CSV file

job_id=2
while [ "$job_id" -le $JOBS ]; do

    param_line=`awk 'NR==1' $2`
    value_line=`awk -v var="$job_id" 'NR==var' $2`

    IFS=','
    read -a PARAMS <<< "$param_line"
    read -a VALUES <<< "$value_line"

    unset PARAMS_VALUES
    for (( i=0; i<${#PARAMS[*]}; ++i)); do
        PARAMS_VALUES+=( "-${PARAMS[$i]}" "${VALUES[$i]}" )
    done

    python $1 "${PARAMS_VALUES[@]}"

    unset param_line
    unset value_line
    unset PARAMS
    unset VALUES
    job_id=$(( job_id + 1 ))
done