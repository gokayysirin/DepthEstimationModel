FROM python:3.10

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean 

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/ 

EXPOSE 8041

CMD ["python", "api.py"]