clean:
	find . -name ".ipynb_checkpoints" -print0 | xargs -0 rm -rf

format:
	black **/*.ipynb

convert FMT FILE:
	jupyter nbconvert --to {{FMT}} {{FILE}} --output-dir converted/
