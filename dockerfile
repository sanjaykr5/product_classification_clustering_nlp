FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

WORKDIR /product_classification

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888 6006
