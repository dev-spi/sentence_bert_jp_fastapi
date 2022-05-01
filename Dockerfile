FROM python:3.8

COPY requirements.txt /
RUN pip install -r requirements.txt

RUN pip install fastapi uvicorn


EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
#CMD ["python","app/sentencebertjapanese.py"]
