#!/usr/bin/env sh
# run.sh - run CAPTCHA classification

# Authors:
#   Basil Contovounesios <contovob@tcd.ie>
#   Salil Kulkarni <sakulkar@tcd.ie>

set -o errexit
set -o nounset

usage() {
  echo 'Usage: run.sh -m MODEL -d DIR -o OUT -S SYMBOLS'
  echo
  echo 'Run CAPTCHA classification.'
  echo
  echo 'Options:'
  echo
  echo '  -m MODEL    TensorFlow Lite file (.tflite)'
  echo '  -d DIR      CAPTCHA directory'
  echo '  -o OUT      Output CSV'
  echo '  -S SYMBOLS  CAPTCHA symbol file'
  exit 1
}

while getopts m:d:o:S: f; do
  case "$f" in
    m) model="${OPTARG}";;
    d) directory="${OPTARG}";;
    o) output="${OPTARG}";;
    S) syms="${OPTARG}";;
    \?) usage;;
  esac
done

[ -z "${model:-}" -o -z "${directory:-}" -o \
     -z "${output:-}" -o -z "${syms:-}" ] && usage

echo "Model    : ${model}"
echo "CAPTCHAs : ${directory}"
echo "Symbols  : ${syms}"
echo "Output   : ${output}"

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
./classifylite.py -m "${model}" -d "${directory}" -o "${output}" -S "${syms}"

echo 'Deactivating .venv'
deactivate
