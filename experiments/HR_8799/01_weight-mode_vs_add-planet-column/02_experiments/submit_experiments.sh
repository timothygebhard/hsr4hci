#!/usr/bin/env bash

while read -r line ; do \
  printf "\nSUBMITTING FILE: " ;
  echo "$line/submit.sub" ; \
  csb 25 "$line/submit.sub" ;
  printf "\nGoing to sleep for 10 minutes... " ;
  sleep 10m ;
  printf "Done!\n" ;
done < list_of_experiments.txt