# Shared helper: start a daemon in a new session so IDE/agent shell teardown
# cannot SIGKILL the whole LLM stack via process-group membership.
#
# Usage: source this file, then
#   detach_cmd PIDFILE STDOUT_LOG STDERR_LOG command [args...]
# Prints the new PID on stdout and writes it to PIDFILE.

detach_cmd() {
  if [ "$#" -lt 4 ]; then
    echo "detach_cmd: pidfile stdout stderr command [args...]" >&2
    return 2
  fi
  local pidfile="$1" out_log="$2" err_log="$3"
  shift 3
  /usr/bin/env python3 - "$pidfile" "$out_log" "$err_log" "$@" <<'PY'
import subprocess
import sys

pidfile, out_path, err_path, *cmd = sys.argv[1:]
with open(out_path, "a", buffering=1) as out, open(err_path, "a", buffering=1) as err:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=err,
        start_new_session=True,
        close_fds=True,
    )
with open(pidfile, "w", encoding="utf-8") as handle:
    handle.write(str(proc.pid))
print(proc.pid)
PY
}
