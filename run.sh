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

if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip3 install -r requirements.txt
elif [ ! -x "$(command -v deactivate)" ]; then
  . .venv/bin/activate
fi

./classifylite.py -m "${model}" -d "${directory}" -o "${output}" -S "${syms}"
