install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	pytest -vv

format:
	black starter/*.py starter/starter/ml/*.py

lint:
	flake8 --ignore=E303,E302  --max-line-length=88 starter/*.py starter/starter/ml/*.py 

dvc:          
	dvc remote add -d s3remote s3://census-bureau-classification &&\
          dvc pull	

all: install lint test
