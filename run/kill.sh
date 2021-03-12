#!/usr/bin/env bash

ps | grep python | awk -F ' ' '{print $1}' | xargs kill -9
ps -ef