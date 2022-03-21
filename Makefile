jupyter:
	docker run -it -v ${PWD}:/home/jovyan -p 8090:8090 biocomp jupyter notebook --port 8090
