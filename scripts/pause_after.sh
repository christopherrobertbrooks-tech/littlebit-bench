#!/usr/bin/env bash
# pause_after.sh <run name> <checkpoint step> <deadline HH:MM> <queue script name>
# Pauses a training run right after it has fully written checkpoint-<step> (the progress bar has moved past it), or at
# the deadline, whichever comes first. Stops the queue script first so it doesn't go on to the evals, then the
# trainer; checks both are really gone. Continue later with: RESUME=1 scripts/<queue script>.
source "$(dirname "$0")/env.sh"
name=$1; step=$2; deadline=$3; queue=$4
dir=$RUNS/$name; status=$RUNS/$name.pause.log
say() { echo "$(date '+%F %T') $*" | tee -a "$status"; }
at=$(tr '\r' '\n' < "$dir/train.log" 2>/dev/null | grep -ao ' [0-9]*/[0-9]* \[' | tail -1 | tr -dc '0-9/' | cut -d/ -f1)
say "waiting for checkpoint-$step (now at step ${at:-?}), deadline $deadline"
while :; do
    at=$(tr '\r' '\n' < "$dir/train.log" | grep -ao ' [0-9]*/[0-9]* \[' | tail -1 | tr -dc '0-9/' | cut -d/ -f1)
    if [ -f "$dir/checkpoint-$step/trainer_state.json" ] && [ "${at:-0}" -gt "$step" ]; then say "checkpoint-$step written, step $at"; break; fi
    if [ "$(date +%H%M)" -ge "${deadline/:/}" ]; then say "deadline $deadline reached at step ${at:-?}"; break; fi
    sleep 20
done
q=$(pgrep -f "[b]ash .*scripts/$queue"); t=$(pgrep -f "[m]ain.py .*--f_name $name ")
say "stopping queue (${q:-none}) and trainer (${t:-none})"
[ -n "$q" ] && kill -TERM $q
[ -n "$t" ] && kill -TERM $t
for i in $(seq 1 30); do alive=""; for p in $q $t; do kill -0 $p 2>/dev/null && alive="$alive $p"; done; [ -z "$alive" ] && break; sleep 1; done
[ -n "$alive" ] && { say "still alive after 30 s:$alive -- SIGKILL"; kill -KILL $alive; sleep 2; }
pgrep -f "[m]ain.py .*--f_name $name " > /dev/null && say "TRAINER STILL RUNNING" || say "paused; saved: $(ls -d "$dir"/checkpoint-* | xargs -n1 basename | tr '\n' ' ')"
