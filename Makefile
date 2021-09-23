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
	dvc pull -r s3remote

all: install lint test
