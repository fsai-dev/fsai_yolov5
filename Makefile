IMAGE_NAME?=fsai-yolov5

export IMAGE_NAME

##### Build Docker Image
build:
	docker build -t $(IMAGE_NAME) .


# ------ EC2 Ubuntu ------
run-ec2:
	docker run --gpus all -it --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /home/ubuntu/data:/home/data -v /home/ubuntu/model-output:/home/model-output -v /home/ubuntu/db-images:/home/db-images -v /home/ubuntu/test_images:/home/test_images -v ${PWD}:/code --rm -it $(IMAGE_NAME)


# ------ FSAI Lambda Machine ------
run-fsai:
	docker run --gpus all -it --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /home/fsai/data:/home/data -v /home/fsai/model-output:/home/model-output -v ${PWD}:/code --rm -it $(IMAGE_NAME)