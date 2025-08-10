#!/bin/bash

# Inject the environment variables into the env file

cp ./service_config/files/env.template ./service_config/files/env

# Set required environment variables for create_key_mappings.sh
export INTERNAL_KEY="$INTERNAL_KEY"
export EXTERNAL_KEYS="$EXTERNAL_KEYS"

chmod +x ./service_config/create_key_mappings.sh

## Generate and capture key mappings
KEY_MAPPINGS=$(./service_config/create_key_mappings.sh)
if [ $? -ne 0 ]; then
    echo "Failed to generate key mappings"
    exit 3
fi
    
## Escape special characters for sed
ESCAPED_MAPPINGS=$(printf '%s\n' "$KEY_MAPPINGS" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
    
## Replace in env file
sed -i "s/INJECTED_ALLOWED_KEYS/${ESCAPED_MAPPINGS}/g" ./service_config/files/env

if [ `grep -c "INJECTED_ALLOWED_KEYS" ./service_config/files/env` -ne 0 ]
then
    echo "INJECTED_ALLOWED_KEYS was not successfully replaced. Exiting."
    exit 3
fi

# Replace port
if [ -z "$INJECTED_PORT" ]
then
    echo "INJECTED_PORT is not set. Exiting."
    exit 4
else
    sed -i "s/INJECTED_PORT/${INJECTED_PORT}/g" ./service_config/files/env
    
    if [ `grep -c "INJECTED_PORT" ./service_config/files/env` -ne 0 ]
    then
        echo "INJECTED_PORT was not successfully replaced. Exiting."
        exit 4
    fi
fi
