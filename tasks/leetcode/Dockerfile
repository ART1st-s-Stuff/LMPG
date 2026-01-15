FROM gcc:15-trixie
VOLUME ["/workspace"]

RUN cd /tmp && \
    wget https://github.com/j178/leetgo/releases/latest/download/leetgo_linux_x86_64.tar.gz && \
    mkdir leetgo && \
    tar -xzf leetgo_linux_x86_64.tar.gz -C leetgo && \
    mv leetgo/leetgo /usr/local/bin/leetgo && \
    chmod +x /usr/local/bin/leetgo
RUN mkdir /workspace && cd /workspace && leetgo init -t us -l cpp

WORKDIR /workspace/cpp
CMD ["bash"]