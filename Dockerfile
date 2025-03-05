FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN apt-get update && apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/
RUN pip3 install --no-cache-dir jupyterlab polars transformers ipywidgets

WORKDIR /workspace

COPY src ./src
COPY Aspect_based_Sentiment_Analisys.ipynb .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
