set -euxo pipefail

USER_LOWER=$(echo "$USER" | awk '{print tolower($0)}')

container_name=mae_stereo_v4.0_${USER_LOWER}
docker exec -it -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK -u root ${container_name} /bin/bash