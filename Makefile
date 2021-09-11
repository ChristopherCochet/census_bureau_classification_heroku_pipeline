install:
	pip install --upgrade pip &&\
		pip install -r starter/requirements.txt

test:
	pytest -vv --cov-report term-missing --cov=app starter/test_*.py

format:
	black starter/*.py

lint:
	pylint --disable=R,C starter/main.py 

all: install lint test
