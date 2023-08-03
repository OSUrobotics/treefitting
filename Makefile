.PHONY: black
black:
    black ./Image_based --line-length=120
    
.PHONY: flake8
flake8:
	flake8 ./Image_based --count --show-source --statistics

.PHONY: format
format:
	make black
	make flake8