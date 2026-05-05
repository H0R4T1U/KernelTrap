#!/bin/bash
# Wrap every SSH session with script(1) for full TTY recording.
# Files are stored in /var/log/sessions (named volume, persists across restarts).
LOGFILE="/var/log/sessions/$(date +%Y%m%d_%H%M%S)_${USER}_$$.log"
exec script -q -f "$LOGFILE" /bin/bash -l
