IMAGE_NAME?="dockerhub.aganitha.ai:4443/dev/vina-gpu-kailash"
CONTAINER_NAME?=$(USER)-vina-gpu

build-image: 
	@cd devops && docker build -t $(IMAGE_NAME) .

start-container: 
	@cd devops && IMAGE_NAME=$(IMAGE_NAME) CONTAINER_NAME=$(CONTAINER_NAME) \
		./start_container.sh; 

enter-container:
	@echo "You are inside the Container: \033[1;33m$(CONTAINER_NAME)\033[0m"; \
	docker exec -u root -it $(CONTAINER_NAME) bash || true

stop-container:
	@if docker ps -a --format '{{.Names}}' | grep -q "^$(CONTAINER_NAME)$$"; then \
		echo "Stopped and Removed container: \033[1;31m$(CONTAINER_NAME)\033[0m"; \
		docker rm -f $(CONTAINER_NAME) > /dev/null 2>&1 || true; \
	else echo "There is no running container: \033[1;31m$(CONTAINER_NAME)\033[0m"; fi
