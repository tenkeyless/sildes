#!/usr/bin/env bash
# Slidev Docker 실행 스크립트 — 사용법: ./run-slidev.sh <슬라이드파일.md>
# 예: ./run-slidev.sh Day1-3_BERT_Slides.md

SLIDE_FILE="${1:?슬라이드 파일을 지정하세요. 예: ./run-slidev.sh Day1-3_BERT_Slides.md}"

docker run --name slidev-m1 --rm -it \
  -v "$(pwd):/slidev" \
  -p 3030:3030 \
  my-slidev:m1 slidev "$SLIDE_FILE" --remote
