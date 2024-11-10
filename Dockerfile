FROM python:3.12.5-slim

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

CMD ["/bin/bash"]
