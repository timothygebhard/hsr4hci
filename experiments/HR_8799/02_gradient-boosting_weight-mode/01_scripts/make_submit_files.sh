#!/usr/bin/env bash

while read -r line ; do \
  printf "Creating submit file for: " ; \
  echo "$line" ; \
  python "../02_experiments/$line/make_submit_file.py" ; \
  printf "\n\n" ; \
done < ../02_experiments/list_of_experiments.txt