CWD := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

build:
	docker build -t pose-estimation .

start:
	docker run \
		--gpus all \
		-p "5000:5000" \
		-v "${CWD}Makefile:/home/user/pose-estimation/Makefile" \
		-v "${CWD}server.py:/home/user/pose-estimation/server.py" \
		-v "${CWD}estimate_pose.py:/home/user/pose-estimation/estimate_pose.py" \
		-v "${CWD}classify_pose.py:/home/user/pose-estimation/classify_pose.py" \
		pose-estimation

enter:
	docker run \
		--gpus all \
		-p "5000:5000" \
		-v "${CWD}Makefile:/home/user/pose-estimation/Makefile" \
		-v "${CWD}server.py:/home/user/pose-estimation/server.py" \
		-v "${CWD}estimate_pose.py:/home/user/pose-estimation/estimate_pose.py" \
		-v "${CWD}classify_pose.py:/home/user/pose-estimation/classify_pose.py" \
		-it pose-estimation bash
