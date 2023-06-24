# this is dockerfile
FROM python:3.11.3
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python download_model.py
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app