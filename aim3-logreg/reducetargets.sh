#!/bin/bash

RCV1_DIR=/home/andre/dev/datasets/RCV1-v2
grep -i CCAT ${RCV1_DIR}/rcv1-v2.topics.qrels > ${RCV1_DIR}/rcv1-v2.topics_ccat.qrels
grep -i ECAT ${RCV1_DIR}/rcv1-v2.topics.qrels > ${RCV1_DIR}/rcv1-v2.topics_ecat.qrels
grep -i GCAT ${RCV1_DIR}/rcv1-v2.topics.qrels > ${RCV1_DIR}/rcv1-v2.topics_gcat.qrels
grep -i MCAT ${RCV1_DIR}/rcv1-v2.topics.qrels > ${RCV1_DIR}/rcv1-v2.topics_mcat.qrels
