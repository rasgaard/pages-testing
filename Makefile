.PHONY: help serve install build clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies (first time setup)"
	@echo "  make serve    - Start local development server"
	@echo "  make build    - Build the site"
	@echo "  make clean    - Clean build artifacts"

install:
	gem install bundler
	bundle install

serve: install
	bundle exec jekyll serve

build:
	bundle exec jekyll build

clean:
	bundle exec jekyll clean
