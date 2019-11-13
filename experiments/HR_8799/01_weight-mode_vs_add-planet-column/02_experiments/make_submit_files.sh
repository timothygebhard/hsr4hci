#!/usr/bin/env bash

while read -r line ; do \
  printf "Creating submit file for: " ;
  echo "$line" ; \
  python "$line/make_submit_file.py" ;
  printf "\n\n" ;
done < list_of_experiments.txt