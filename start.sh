#!/bin/bash
python3.5 -m venv .venv
. .venv/bin/activate
while true
do
  python bot.py
#deactivate
done
