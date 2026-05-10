#!/bin/bash
# Wrap every SSH session with script(1) for full TTY recording.
# Files are stored in /var/log/sessions (named volume, persists across restarts).
#
# IMPORTANT: util-linux `script` treats positional args after [file] as its
# own options, so `script -q -f LOGFILE /bin/bash -l` blows up with
# "script: invalid option -- 'l'". The shell command MUST be passed via -c.
LOGFILE="/var/log/sessions/$(date +%Y%m%d_%H%M%S)_${USER}_$$.log"
exec script -q -f -c "/bin/bash -l" "$LOGFILE"
