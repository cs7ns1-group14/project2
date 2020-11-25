#!/usr/bin/env bash
# run.sh - run CAPTCHA classification

# Authors:
#   Basil Contovounesios <contovob@tcd.ie>
#   Salil Kulkarni <sakulkar@tcd.ie>

set -o errexit
set -o nounset

usage() {
  echo 'Usage: run.sh [-m MODEL] [-d DIR] [-o OUT] [-S SYMBOLS]'
  echo
  echo 'Run CAPTCHA classification.'
  echo
  echo 'Options:'
  echo
  echo '  -h, ?       Print this help text'
  echo '  -m MODEL    TensorFlow Lite file (.tflite)'
  echo '              (default: models/model1.tflite)'
  echo '  -d DIR      CAPTCHA directory'
  echo '              (default: img/contovob)'
  echo '  -o OUT      Output CSV'
  echo '              (default: results/classified.csv)'
  echo '  -S SYMBOLS  CAPTCHA symbol file'
  echo '              (default: syms/p2.txt)'
  exit 1
}

while getopts m:d:o:S:h f; do
  case "$f" in
    m) model="${OPTARG}";;
    d) directory="${OPTARG}";;
    o) output="${OPTARG}";;
    S) syms="${OPTARG}";;
    h | \?) usage;;
  esac
done

echo "Model    : ${model:=models/model1.tflite}"
echo "CAPTCHAs : ${directory:=img/contovob}"
echo "Symbols  : ${syms:=results/classified.csv}"
echo "Output   : ${output:=syms/p2.txt}"

if [ ! -d .venv ]; then
  echo 'No .venv directory found; creating'
  python3 -m venv .venv
  echo 'Activating .venv'
  . .venv/bin/activate
  echo 'Installing requirements in .venv'
  pip3 install -r requirements.txt
elif [ ! -x "$(command -v deactivate)" ]; then
  echo 'Activating .venv'
  . .venv/bin/activate
fi

echo 'Running classification'
time ./classifylite.py \
     -m "${model}" -d "${directory}" -o "${output}" -S "${syms}"

echo 'Deactivating .venv'
deactivate
