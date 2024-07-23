FROM python:3.9



RUN groupadd -r evaluator && useradd -m --no-log-init -r -g evaluator evaluator

RUN mkdir -p /opt/evaluation /input /output \
    && chown evaluator:evaluator /opt/evaluation /input /output

USER evaluator
WORKDIR /opt/evaluation

ENV PATH="/home/evaluator/.local/bin:${PATH}"

RUN python -m pip install --user -U pip



COPY --chown=evaluator:evaluator ground-truth /opt/evaluation/ground-truth

COPY --chown=evaluator:evaluator requirements.txt /opt/evaluation/
RUN pip install Cython
RUN pip install pandas
RUN pip install numpy
RUN pip install matplotlib
RUN python -m pip install --user -r requirements.txt

COPY --chown=evaluator:evaluator evaluation.py /opt/evaluation/
COPY --chown=evaluator:evaluator metric.py /opt/evaluation/
COPY --chown=evaluator:evaluator gc_to_coco.py /opt/evaluation/
COPY --chown=evaluator:evaluator ground-truth   /opt/evaluation/ground-truth
COPY --chown=evaluator:evaluator inference_output /opt/evaluation/inference_output

ENTRYPOINT "python" "-m" "evaluation"
