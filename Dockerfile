# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies Cython requiredfor installation
RUN pip3 install Cython==0.29.14 && pip3 install -r requirements.txt

# copy the content of the local src directory to the working directory

COPY resources/ ./resources
COPY src/ ./src
COPY run.sh .

EXPOSE 5000
ENV PYTHONPATH "${PYTHONPATH}:src"
# command to run on container start
ENTRYPOINT [ "bash", "run.sh" ]
CMD [ "demo" ]
