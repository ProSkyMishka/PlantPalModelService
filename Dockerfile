FROM pytorch/pytorch:latest

WORKDIR /app

COPY . /app

# RUN pip3 install -r requirements.txt
# RUN pip3 install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu

RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install --yes \
    astropy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image 

RUN pip install torch

RUN pip3 install -r requirements.txt


EXPOSE 8000

CMD ["python3", "app.py"]