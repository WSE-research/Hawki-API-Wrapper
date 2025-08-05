#!/bin/bash

# Required environment variables:
#   INTERNAL_KEY="internal_value"
#   EXTERNAL_KEYS="key1,key2,key3"

if [[ -z "$INTERNAL_KEY" || -z "$EXTERNAL_KEYS" ]]; then
  echo "INTERNAL_KEY or EXTERNAL_KEYS is empty." >&2
  exit 1
fi

IFS=',' read -ra KEYS <<< "$EXTERNAL_KEYS"

json="{"
for key in "${KEYS[@]}"; do
  json+="\"$key\":\"$INTERNAL_KEY\","
done
json="${json%,}}"

export KEY_MAPPINGS="$json"