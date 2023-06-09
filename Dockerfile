FROM python:3.9.16

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

# set work directory
WORKDIR /src/app

# copy requirements.txt
COPY requirements.txt /src/app/requirements.txt

# install system dependencies
RUN apt-get update \
    && apt-get -y --no-install-recommends install gcc make \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# copy project
COPY ./flask_ap.py /src/app
COPY ./utils1.py /src/app
COPY ./my_model_weights.h5 /src/app
COPY ./tokenizer.pickle /src/app

# set app port
EXPOSE 4002

ENTRYPOINT [ "python" ] 

# Run app.py when the container launches
CMD [ "flask_ap.py","run","--host","0.0.0.0"] 
