BASE_BEAKER_IMAGE="spoc-base-$(date +"%Y%m%d")" \
  && DOCKER_BUILDKIT=1 docker build -t \
   $BASE_BEAKER_IMAGE:latest \
   --file Dockerfile \
   .

echo "Docker image name: ${BASE_BEAKER_IMAGE}"