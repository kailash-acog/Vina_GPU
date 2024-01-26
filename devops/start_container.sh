#!/usr/bin/env bash

if ! docker images | grep -wq ${IMAGE_NAME}; then
    echo -e "Pulling image: \e[1;33m${IMAGE_NAME}\e[0m"
    docker pull ${IMAGE_NAME}
fi


if docker ps --format '{{.Names}}' | grep -wq ${CONTAINER_NAME}; then
    echo -e "Already running container: \e[1;32m${CONTAINER_NAME}\e[0m"
else
    docker run -d \
      --name ${CONTAINER_NAME} \
      --restart unless-stopped \
      -v $(pwd)/../main:/home/vina_gpu \
      --label description="Jupyterlab_for_Vina_GPU_2.0" \
      --label user=$USER \
      --label port=7777 \
      --label security=basic \
      --gpus all \
      ${IMAGE_NAME} &> /dev/null

    if [ $? -eq 0 ]; then
        echo -e "Successfully started container: \e[1;32m${CONTAINER_NAME}\e[0m"
    else
        echo -e "Failed to start container: \e[1;32m${CONTAINER_NAME}\e[0m"
    fi
fi
