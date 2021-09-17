install:
	pip install --upgrade pip &&\
		pip install -r starter/requirements.txt

test:
	pytest -vv --cov-report term-missing --cov=app starter/test_*.py

format:
	black starter/*.py

lint:
	pylint --disable=R,C starter/main.py 

dvc:          
	dvc remote add -d s3remote s3://census-bureau-classification &&\
          dvc pull	

all: install lint test
