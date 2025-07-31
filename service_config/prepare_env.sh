#!/bin/bash

# Inject the environment variables into the env file

cp ./service_config/files/env.template ./service_config/files/env

# test if the environment variables are set and not empty
if [ -z "$INJECTED_OPENAI_DEFAULT_API_KEY" ]
then
    echo "INJECTED_OPENAI_DEFAULT_API_KEY is not set. Exiting."
    exit 2
else
    sed -i "s/INJECTED_OPENAI_DEFAULT_API_KEY/${INJECTED_OPENAI_DEFAULT_API_KEY}/g" ./service_config/files/env

    if [ `grep -c "INJECTED_OPENAI_DEFAULT_API_KEY" ./service_config/files/env` -ne 0 ]
    then
        echo "INJECTED_OPENAI_DEFAULT_API_KEY was not successfully replaced. Exiting."
        exit 2
    fi
fi

if [ -z "$INJECTED_ALLOWED_KEYS" ]
then
    echo "INJECTED_ALLOWED_KEYS is not set. Exiting."
    exit 3
else
    sed -i "s/INJECTED_ALLOWED_KEYS/${INJECTED_ALLOWED_KEYS}/g" ./service_config/files/env

    if [ `grep -c "INJECTED_ALLOWED_KEYS" ./service_config/files/env` -ne 0 ]
    then
        echo "INJECTED_ALLOWED_KEYS was not successfully replaced. Exiting."
        exit 3
    fi
fi

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
