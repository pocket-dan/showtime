CWD := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

build:
	docker build -t showtime-ml .

start:
	docker run \
		-d \
		--name showtime \
		-p "5000:5000" \
		showtime-ml

stop:
	docker rm -f showtime

logs:
	docker logs -f showtime

start-dev:
	docker run \
		-d \
		--name showtime \
		-p "5000:5000" \
		-v "${PWD}/classifier:/home/user/showtime/ml/classifier" \
		-v "${PWD}/classify_pose.py:/home/user/showtime/ml/classify_pose.py" \
		-v "${PWD}/estimate_pose.py:/home/user/showtime/ml/estimate_pose.py" \
		-v "${PWD}/Makefile:/home/user/showtime/ml/Makefile" \
		showtime-ml

enter:
	docker run \
		-p "5000:5000" \
		-v "${PWD}/classifier:/home/user/showtime/ml/classifier" \
		-v "${PWD}/classify_pose.py:/home/user/showtime/ml/classify_pose.py" \
		-v "${PWD}/estimate_pose.py:/home/user/showtime/ml/estimate_pose.py" \
		-v "${PWD}/Makefile:/home/user/showtime/ml/Makefile" \
		-it showtime-ml bash
