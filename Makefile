install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt
venv :
	test -d venv || python3 -m venv venv

setup:
	pip install numpy
	pip install matplotlib


clean:
	rm -rf venv
	find -iname "*.pyc" -delete
