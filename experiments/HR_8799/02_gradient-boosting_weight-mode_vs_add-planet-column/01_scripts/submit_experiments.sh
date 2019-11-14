#!/usr/bin/env bash

while read -r line ; do \
  printf "Submitting jobs for: " ;
  echo "$line" ; \
  condor_submit_bid 25 "../02_experiments/$line/submit.sub" ; \
  printf "\nGoing to sleep for 30 minutes (%s)... " "$(date +"%T")"; \
  sleep 30m ; \
  printf "Done!\n\n" ;
done < ../02_experiments/list_of_experiments.txt