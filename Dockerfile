
# pipenv base container
FROM kennethreitz/pipenv

LABEL MAINTAINER="John Naylor <jonaylor89@gmail.com>"

# cd to /app
WORKDIR /app

# copy source code to /app
COPY . /app/

# install dependencies
RUN pipenv install --deploy --system

RUN mkdir -p /app/datasets/Cancerous_cell_smears

# run script
CMD ["python3", "/app/main.py", "preprocess", "test"]
