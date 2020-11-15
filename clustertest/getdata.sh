#!/bin/bash
curl http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz --output kdd10.data.gz
gunzip kdd10.data.gz