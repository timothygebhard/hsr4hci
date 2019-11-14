#!/usr/bin/env bash

while read -r line ; do \
  printf "Merging residuals for for: " ;
  echo "$line" ; \
  python "$line/merge_residuals.py" ;
  printf "\n\n" ;
done < list_of_experiments.txt