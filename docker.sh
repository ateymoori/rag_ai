#!/bin/bash

echo "Enter your action: list-containers, commit-push"
read action

# Repository name
repo_name="amirhosseinteymoori/ai"

# Hardcoded Docker token
docker_token="dckr_pat_xCnLuED1vgkure0aoh8-no4jcRw"

docker_login() {
    echo "Logging in to Docker Hub using the hardcoded token..."
    echo $docker_token | docker login --username amirhosseinteymoori --password-stdin
    if [ $? -ne 0 ]; then
        echo "Docker login failed. Exiting."
        exit 1
    fi
}

list_containers() {
    echo "Currently running Docker Containers:"
    docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}" | nl -v 0
}

commit_and_push() {
    echo "Enter the container ID you wish to commit:"
    read container_id

    echo "Enter a tag for the new image (this will be used as $repo_name:<tag>):"
    read tag

    new_image_name="$repo_name:$tag"

    # Commit the container to a new image
    docker commit "$container_id" "$new_image_name"

    # Pushing the image
    docker push "$new_image_name"
}

case $action in
    list-containers)
        list_containers
        ;;
    commit-push)
        docker_login
        list_containers
        commit_and_push
        ;;
    *)
        echo "Invalid action. Please enter 'list-containers' or 'commit-push'."
        ;;
esac
