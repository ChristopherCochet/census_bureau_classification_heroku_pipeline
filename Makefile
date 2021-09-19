install:
	pip install --upgrade pip &&\
		pip install -r starter/requirements.txt

test:
	pytest -vv

format:
	black starter/*.py

lint:
	flake8 --ignore=E303,E302 starter/main.py 

dvc:          
	dvc remote add -d s3remote s3://census-bureau-classification &&\
          dvc pull	

all: install lint test
