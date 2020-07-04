#!/usr/bin/env bash
grep '>>> ' full_log.txt | cut -d ',' -f 2- |  sort -t ',' -h -k 3,3 | sort -t ',' -s -h -k 1,1 | sed 's/%/\\,\\%/g' | sed 's/\./\{,\}/g' | grep -E '^[0-4a-z]' > stripped.txt
