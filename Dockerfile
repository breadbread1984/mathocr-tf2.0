FROM weli/tensorflow-base

RUN pip3 install --upgrade numpy tf-nightly-2.0-preview requests

# supervisor 
COPY setup /root/setup
COPY supervisor.ini /etc/supervisor.d/default.ini
RUN /root/setup
# copy code
COPY . /root/matchocr
RUN cp /root/models/matchocr /root/CROHME-png
WORKDIR /root/matchocr
RUN python3 create_datasets.py && python3 train_mathocr.py && python3 save_model.py && python3 convert_model.py

ENTRYPOINT supervisord -c /etc/supervisor.d/default.ini && sh
