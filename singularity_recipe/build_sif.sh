#!/bin/bash

if [ -e config.json ]; then
    # load variables from config.json
    echo "Loading variables from config.json"
    echo "Remember to always use full file paths inside config.json file. If not, you may encounter errors."

    simg_file=$(cat config.json | jq -r '.simg_file')
    sif_file=$(cat config.json | jq -r '.sif_file')

    if [ "$(cat config.json | jq 'has("put_date_on_sif_file_name")')" = "true" ]; then
        if [ "$(cat config.json | jq -r '.put_date_on_sif_file_name')" = "yes" ]; then
            sif_file="$(cat config.json | jq -r '.sif_file')_$(date +%F).sif"
        fi
    fi

    if [ "$(cat config.json | jq 'has("use_custom_tmp_dir")')" = "true" ] && [ "$(cat config.json | jq 'has("tmp_dir")')" = "true" ] ; then
        use_tmp_dir=$(cat config.json | jq -r '.use_custom_tmp_dir')
        tmp_dir=$(cat config.json | jq -r '.tmp_dir')
    fi

    if [ "$use_tmp_dir" = "yes" ]; then
        mkdir =p $tmp_dir
        export SINGULARITY_TMPDIR="$tmp_dir"
        export TMPDIR="$tmp_dir"
    fi

    #start 
    echo "Starting to build singularity image $sif_file using $simg_file recipe."
    sudo -E singularity build $sif_file $simg_file

else
    echo "config.json not found."
fi

