#!/usr/bin/env bash

while read -r line ; do \
  printf "Merging residuals for for: " ; \
  echo "$line" ; \
  python "../02_experiments/$line/merge_residuals.py" ; \
  printf "\n\n" ; \
done < ../02_experiments/list_of_experiments.txt