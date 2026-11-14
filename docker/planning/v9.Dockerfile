FROM teamedplanning-ad:v1
USER 0
RUN python3 -m pip install --no-cache-dir --no-deps --target /opt/v9-python \
    --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu \
    && python3 -m pip install --no-cache-dir --target /opt/v9-python \
       'typing-extensions>=4.8,<5' 'filelock<4' 'sympy==1.12.1' 'fsspec<2025' 'jinja2<4'
