#!/usr/bin/env python3
"""
KernelTrap Syscall Logger (Agent)

Collects syscall events via Tracee/auditd, publishes them to Redis Streams
on the central analysis server, and listens for pivot commands back.

Redis Streams used:
  events.{hostname}   <- agent publishes batches of events here
  commands.{hostname} <- agent subscribes; central server publishes pivot commands

Usage (agent mode, sends to central server):
    tracee --output json | python syscall_logger.py --source tracee \
        --redis-host 10.0.0.5 --redis-port 6379

Usage (standalone, local CSV only):
    tracee --output json | python syscall_logger.py --source tracee --output events.csv
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, List, TextIO

# Syscall name to eventId mapping (based on BETH dataset / x86_64 Linux syscalls)
SYSCALL_TO_ID = {
    "read": 0, "write": 1, "open": 2, "close": 3, "stat": 4, "fstat": 5, "lstat": 6,
    "poll": 7, "lseek": 8, "mmap": 9, "mprotect": 10, "munmap": 11, "brk": 12,
    "ioctl": 16, "access": 21, "pipe": 22, "select": 23, "sched_yield": 24,
    "mremap": 25, "msync": 26, "mincore": 27, "madvise": 28, "shmget": 29,
    "shmat": 30, "shmctl": 31, "dup": 32, "dup2": 33, "pause": 34, "nanosleep": 35,
    "getitimer": 36, "alarm": 37, "setitimer": 38, "getpid": 39, "sendfile": 40,
    "socket": 41, "connect": 42, "accept": 43, "sendto": 44, "recvfrom": 45,
    "sendmsg": 46, "recvmsg": 47, "shutdown": 48, "bind": 49, "listen": 50,
    "getsockname": 51, "getpeername": 52, "socketpair": 53, "setsockopt": 54,
    "getsockopt": 55, "clone": 56, "fork": 57, "vfork": 58, "execve": 59, "exit": 60,
    "wait4": 61, "kill": 62, "uname": 63, "semget": 64, "semop": 65, "semctl": 66,
    "shmdt": 67, "msgget": 68, "msgsnd": 69, "msgrcv": 70, "msgctl": 71, "fcntl": 72,
    "flock": 73, "fsync": 74, "fdatasync": 75, "truncate": 76, "ftruncate": 77,
    "getdents": 78, "getcwd": 79, "chdir": 80, "fchdir": 81, "rename": 82,
    "mkdir": 83, "rmdir": 84, "creat": 85, "link": 86, "unlink": 87, "symlink": 88,
    "readlink": 89, "chmod": 90, "fchmod": 91, "chown": 92, "fchown": 93, "lchown": 94,
    "umask": 95, "gettimeofday": 96, "getrlimit": 97, "getrusage": 98, "sysinfo": 99,
    "times": 100, "ptrace": 101, "getuid": 102, "syslog": 103, "getgid": 104,
    "setuid": 105, "setgid": 106, "geteuid": 107, "getegid": 108, "setpgid": 109,
    "getppid": 110, "getpgrp": 111, "setsid": 112, "setreuid": 113, "setregid": 114,
    "getgroups": 115, "setgroups": 116, "setresuid": 117, "getresuid": 118,
    "setresgid": 119, "getresgid": 120, "getpgid": 121, "setfsuid": 122,
    "setfsgid": 123, "getsid": 124, "capget": 125, "capset": 126, "rt_sigpending": 127,
    "rt_sigtimedwait": 128, "rt_sigqueueinfo": 129, "rt_sigsuspend": 130,
    "sigaltstack": 131, "utime": 132, "mknod": 133, "uselib": 134, "personality": 135,
    "ustat": 136, "statfs": 137, "fstatfs": 138, "sysfs": 139, "getpriority": 140,
    "setpriority": 141, "sched_setparam": 142, "sched_getparam": 143,
    "sched_setscheduler": 144, "sched_getscheduler": 145, "sched_get_priority_max": 146,
    "sched_get_priority_min": 147, "sched_rr_get_interval": 148, "mlock": 149,
    "munlock": 150, "mlockall": 151, "munlockall": 152, "vhangup": 153, "modify_ldt": 154,
    "pivot_root": 155, "prctl": 157, "arch_prctl": 158, "adjtimex": 159,
    "setrlimit": 160, "chroot": 161, "sync": 162, "acct": 163, "settimeofday": 164,
    "mount": 165, "umount2": 166, "swapon": 167, "swapoff": 168, "reboot": 169,
    "sethostname": 170, "setdomainname": 171, "ioperm": 172, "init_module": 175,
    "delete_module": 176, "quotactl": 179, "gettid": 186, "readahead": 187,
    "setxattr": 188, "lsetxattr": 189, "fsetxattr": 190, "getxattr": 191,
    "lgetxattr": 192, "fgetxattr": 193, "listxattr": 194, "llistxattr": 195,
    "flistxattr": 196, "removexattr": 197, "lremovexattr": 198, "fremovexattr": 199,
    "tkill": 200, "time": 201, "futex": 202, "sched_setaffinity": 203,
    "sched_getaffinity": 204, "io_setup": 206, "io_destroy": 207, "io_getevents": 208,
    "io_submit": 209, "io_cancel": 210, "lookup_dcookie": 212, "epoll_create": 213,
    "remap_file_pages": 216, "getdents64": 217, "set_tid_address": 218,
    "restart_syscall": 219, "semtimedop": 220, "fadvise64": 221, "timer_create": 222,
    "timer_settime": 223, "timer_gettime": 224, "timer_getoverrun": 225,
    "timer_delete": 226, "clock_settime": 227, "clock_gettime": 228,
    "clock_getres": 229, "clock_nanosleep": 230, "exit_group": 231, "epoll_wait": 232,
    "epoll_ctl": 233, "tgkill": 234, "utimes": 235, "mbind": 237, "set_mempolicy": 238,
    "get_mempolicy": 239, "mq_open": 240, "mq_unlink": 241, "mq_timedsend": 242,
    "mq_timedreceive": 243, "mq_notify": 244, "mq_getsetattr": 245, "kexec_load": 246,
    "waitid": 247, "add_key": 248, "request_key": 249, "keyctl": 250, "ioprio_set": 251,
    "ioprio_get": 252, "inotify_init": 253, "inotify_add_watch": 254,
    "inotify_rm_watch": 255, "migrate_pages": 256, "openat": 257, "mkdirat": 258,
    "mknodat": 259, "fchownat": 260, "futimesat": 261, "newfstatat": 262, "unlinkat": 263,
    "renameat": 264, "linkat": 265, "symlinkat": 266, "readlinkat": 267, "fchmodat": 268,
    "faccessat": 269, "pselect6": 270, "ppoll": 271, "unshare": 272,
    "set_robust_list": 273, "get_robust_list": 274, "splice": 275, "tee": 276,
    "sync_file_range": 277, "vmsplice": 278, "move_pages": 279, "utimensat": 280,
    "epoll_pwait": 281, "signalfd": 282, "timerfd_create": 283, "eventfd": 284,
    "fallocate": 285, "timerfd_settime": 286, "timerfd_gettime": 287, "accept4": 288,
    "signalfd4": 289, "eventfd2": 290, "epoll_create1": 291, "dup3": 292,
    "pipe2": 293, "inotify_init1": 294, "preadv": 295, "pwritev": 296,
    "rt_tgsigqueueinfo": 297, "perf_event_open": 298, "recvmmsg": 299,
    "fanotify_init": 300, "fanotify_mark": 301, "prlimit64": 302, "name_to_handle_at": 303,
    "open_by_handle_at": 304, "clock_adjtime": 305, "syncfs": 306, "sendmmsg": 307,
    "setns": 308, "getcpu": 309, "process_vm_readv": 310, "process_vm_writev": 311,
    "kcmp": 312, "finit_module": 313, "sched_setattr": 314, "sched_getattr": 315,
    "renameat2": 316, "seccomp": 317, "getrandom": 318, "memfd_create": 319,
    "kexec_file_load": 320, "bpf": 321, "execveat": 322, "userfaultfd": 323,
    "membarrier": 324, "mlock2": 325, "copy_file_range": 326, "preadv2": 327,
    "pwritev2": 328, "pkey_mprotect": 329, "pkey_alloc": 330, "pkey_free": 331,
    "statx": 332, "io_pgetevents": 333, "rseq": 334,
    "security_file_open": 1000, "security_inode_unlink": 1001,
    "security_socket_create": 1002, "security_socket_listen": 1003,
    "security_socket_connect": 1004, "security_socket_accept": 1005,
    "security_socket_bind": 1006, "security_socket_setsockopt": 1007,
    "security_sb_mount": 1008, "security_bpf": 1009, "security_bpf_map": 1010,
    "security_kernel_read_file": 1011, "security_post_read_file": 1012,
    "security_inode_mknod": 1013, "security_inode_symlink": 1014,
    "security_mmap_file": 1015, "security_file_mprotect": 1016,
    "cap_capable": 1100, "cgroup_attach_task": 1101, "cgroup_mkdir": 1102,
    "cgroup_rmdir": 1103, "security_bprm_check": 1104,
    "sched_process_exec": 1200, "sched_process_exit": 1201, "sched_process_fork": 1202,
    "do_exit": 1203, "commit_creds": 1204, "switch_task_ns": 1205,
}

ID_TO_SYSCALL = {v: k for k, v in SYSCALL_TO_ID.items()}


@dataclass
class SyscallEvent:
    """Syscall event in the format expected by the Isolation Forest model."""
    timestamp: float
    processId: int
    parentProcessId: int
    userId: int
    mountNamespace: int
    threadId: int = 0
    processName: str = ""
    hostName: str = ""
    eventId: int = 0
    eventName: str = ""
    stackAddresses: str = "[]"
    argsNum: int = 0
    returnValue: int = 0
    args: str = "[]"
    sus: int = 0
    evil: int = 0

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "processId": self.processId,
            "threadId": self.threadId,
            "parentProcessId": self.parentProcessId,
            "userId": self.userId,
            "mountNamespace": self.mountNamespace,
            "processName": self.processName,
            "hostName": self.hostName,
            "eventId": self.eventId,
            "eventName": self.eventName,
            "stackAddresses": self.stackAddresses,
            "argsNum": self.argsNum,
            "returnValue": self.returnValue,
            "args": self.args,
            "sus": self.sus,
            "evil": self.evil,
        }

    def to_feature_array(self) -> List[float]:
        return [
            float(self.processId),
            float(self.parentProcessId),
            float(self.userId),
            float(self.mountNamespace),
            float(self.eventId),
            float(self.argsNum),
            float(self.returnValue),
        ]


class TraceeParser:
    """Parse Tracee eBPF JSON output into SyscallEvents."""

    def __init__(self, hostname: str = ""):
        self.hostname = hostname or os.uname().nodename

    def parse_line(self, line: str) -> Optional[SyscallEvent]:
        line = line.strip()
        if not line:
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        if "eventName" in data or "eventId" in data:
            return self._parse_tracee_event(data)
        elif "event" in data:
            return self._parse_tracee_event(data["event"])
        elif "syscall" in data:
            return self._parse_legacy_event(data)
        return None

    def _parse_tracee_event(self, data: Dict[str, Any]) -> SyscallEvent:
        event_name = data.get("eventName", data.get("event_name", "unknown"))
        event_id = data.get("eventId", data.get("event_id", 0))
        if not event_id and event_name:
            event_id = SYSCALL_TO_ID.get(event_name.lower(), 0)

        args_raw = data.get("args", data.get("arguments", []))
        if isinstance(args_raw, list):
            args_json = json.dumps([
                {"name": a.get("name", ""), "type": a.get("type", ""), "value": str(a.get("value", ""))}
                for a in args_raw if isinstance(a, dict)
            ])
            args_num = len(args_raw)
        else:
            args_json = "[]"
            args_num = 0

        return_value = data.get("returnValue", data.get("return_value", data.get("retval", 0)))
        if return_value is None:
            return_value = 0

        timestamp = data.get("timestamp", data.get("ts", time.time()))
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
            except ValueError:
                timestamp = time.time()

        context = data.get("context", data.get("processContext", {}))
        if isinstance(context, dict):
            pid = context.get("pid", context.get("processId", data.get("processId", 0)))
            tid = context.get("tid", context.get("threadId", data.get("threadId", 0)))
            ppid = context.get("ppid", context.get("parentProcessId", data.get("parentProcessId", 0)))
            uid = context.get("uid", context.get("userId", data.get("userId", 0)))
            mnt_ns = context.get("mntNS", context.get("mountNamespace", data.get("mountNamespace", 0)))
            proc_name = context.get("processName", context.get("comm", data.get("processName", "")))
            host_name = context.get("hostName", context.get("hostname", self.hostname))
        else:
            pid = data.get("processId", data.get("pid", 0))
            tid = data.get("threadId", data.get("tid", 0))
            ppid = data.get("parentProcessId", data.get("ppid", 0))
            uid = data.get("userId", data.get("uid", 0))
            mnt_ns = data.get("mountNamespace", data.get("mntNS", 0))
            proc_name = data.get("processName", data.get("comm", ""))
            host_name = data.get("hostName", self.hostname)

        stack_raw = data.get("stackAddresses", data.get("stack", []))
        stack_json = json.dumps(stack_raw) if isinstance(stack_raw, list) else "[]"

        return SyscallEvent(
            timestamp=float(timestamp),
            processId=int(pid) if pid else 0,
            threadId=int(tid) if tid else 0,
            parentProcessId=int(ppid) if ppid else 0,
            userId=int(uid) if uid else 0,
            mountNamespace=int(mnt_ns) if mnt_ns else 0,
            processName=str(proc_name),
            hostName=str(host_name),
            eventId=int(event_id),
            eventName=str(event_name),
            stackAddresses=stack_json,
            argsNum=args_num,
            returnValue=int(return_value) if isinstance(return_value, (int, float)) else 0,
            args=args_json,
        )

    def _parse_legacy_event(self, data: Dict[str, Any]) -> SyscallEvent:
        syscall_name = data.get("syscall", "unknown")
        event_id = SYSCALL_TO_ID.get(syscall_name.lower(), 0)
        return SyscallEvent(
            timestamp=float(data.get("timestamp", time.time())),
            processId=int(data.get("pid", 0)),
            parentProcessId=int(data.get("ppid", 0)),
            userId=int(data.get("uid", 0)),
            mountNamespace=int(data.get("mnt_ns", 0)),
            processName=str(data.get("comm", "")),
            hostName=self.hostname,
            eventId=event_id,
            eventName=syscall_name,
            argsNum=len(data.get("args", [])),
            returnValue=int(data.get("ret", 0)),
            args=json.dumps(data.get("args", [])),
        )


class AuditdParser:
    """Parse auditd log files into SyscallEvents."""

    SYSCALL_PATTERN = re.compile(
        r'type=SYSCALL.*?syscall=(\d+).*?success=(\w+).*?exit=(-?\d+).*?'
        r'a0=([0-9a-f]+).*?ppid=(\d+).*?pid=(\d+).*?uid=(\d+).*?comm="([^"]*)"',
        re.IGNORECASE
    )
    TIMESTAMP_PATTERN = re.compile(r'msg=audit\((\d+\.\d+):\d+\)')

    def __init__(self, hostname: str = ""):
        self.hostname = hostname or os.uname().nodename
        self._mount_ns_cache: Dict[int, int] = {}

    def parse_line(self, line: str) -> Optional[SyscallEvent]:
        line = line.strip()
        if not line or "type=SYSCALL" not in line:
            return None
        match = self.SYSCALL_PATTERN.search(line)
        if not match:
            return None

        syscall_num = int(match.group(1))
        success = match.group(2).lower() == "yes"
        exit_code = int(match.group(3))
        ppid = int(match.group(5))
        pid = int(match.group(6))
        uid = int(match.group(7))
        comm = match.group(8)

        ts_match = self.TIMESTAMP_PATTERN.search(line)
        timestamp = float(ts_match.group(1)) if ts_match else time.time()
        args_num = sum(1 for i in range(4) if f"a{i}=" in line)
        mnt_ns = self._get_mount_ns(pid)
        event_name = ID_TO_SYSCALL.get(syscall_num, f"syscall_{syscall_num}")

        return SyscallEvent(
            timestamp=timestamp,
            processId=pid,
            parentProcessId=ppid,
            userId=uid,
            mountNamespace=mnt_ns,
            processName=comm,
            hostName=self.hostname,
            eventId=syscall_num,
            eventName=event_name,
            argsNum=args_num,
            returnValue=exit_code if success else -abs(exit_code),
        )

    def _get_mount_ns(self, pid: int) -> int:
        if pid in self._mount_ns_cache:
            return self._mount_ns_cache[pid]
        try:
            ns_link = os.readlink(f"/proc/{pid}/ns/mnt")
            mnt_ns = int(ns_link.split("[")[1].rstrip("]"))
            self._mount_ns_cache[pid] = mnt_ns
            return mnt_ns
        except (OSError, IndexError, ValueError):
            return 0

    def parse_file(self, log_path: str) -> Iterator[SyscallEvent]:
        with open(log_path, "r") as f:
            for line in f:
                event = self.parse_line(line)
                if event:
                    yield event


class CSVWriter:
    """Write SyscallEvents to CSV file in BETH format."""

    FIELDNAMES = [
        "timestamp", "processId", "threadId", "parentProcessId", "userId",
        "mountNamespace", "processName", "hostName", "eventId", "eventName",
        "stackAddresses", "argsNum", "returnValue", "args", "sus", "evil"
    ]

    def __init__(self, output_path: str):
        self.output_path = output_path
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None

    def __enter__(self):
        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def write(self, event: SyscallEvent):
        if self._writer:
            self._writer.writerow(event.to_csv_row())

    def flush(self):
        if self._file:
            self._file.flush()


class RedisPublisher:
    """Publish batches of SyscallEvents to a Redis Stream."""

    def __init__(self, host: str, port: int, hostname: str, stream_maxlen: int = 10000):
        try:
            import redis as redis_lib
        except ImportError:
            print("Error: redis-py required. Install with: pip install redis>=5.0", file=sys.stderr)
            sys.exit(1)
        self._r = redis_lib.Redis(host=host, port=port, decode_responses=True)
        self._stream_key = f"events.{hostname}"
        self._maxlen = stream_maxlen
        # Verify connectivity on startup
        self._r.ping()
        print(f"[Redis] Connected to {host}:{port}, publishing to '{self._stream_key}'", file=sys.stderr)

    def publish(self, events: List[SyscallEvent]):
        if not events:
            return
        batch = json.dumps([e.to_csv_row() for e in events])
        self._r.xadd(self._stream_key, {"data": batch}, maxlen=self._maxlen, approximate=True)


class CommandListener(threading.Thread):
    """
    Background thread that reads pivot commands from the central server.

    The central server publishes to commands.{hostname} when it decides a user
    should be pivoted into the honeypot. This thread calls the pivot script
    immediately when a command arrives, regardless of what batch is being processed.
    """

    def __init__(self, host: str, port: int, hostname: str, pivot_script: str):
        super().__init__(daemon=True, name="CommandListener")
        import redis as redis_lib
        self._r = redis_lib.Redis(host=host, port=port, decode_responses=True)
        self._stream_key = f"commands.{hostname}"
        self._pivot_script = pivot_script
        # "$" means: only messages arriving after this thread starts
        self._last_id = "$"

    def run(self):
        print(f"[CommandListener] Listening on '{self._stream_key}'", file=sys.stderr)
        while True:
            try:
                result = self._r.xread({self._stream_key: self._last_id}, block=1000, count=10)
                if not result:
                    continue
                for _stream, messages in result:
                    for msg_id, fields in messages:
                        self._last_id = msg_id
                        try:
                            cmd = json.loads(fields.get("data", "{}"))
                        except json.JSONDecodeError:
                            continue
                        if cmd.get("action") == "pivot":
                            _trigger_pivot(cmd.get("user", ""), self._pivot_script)
            except Exception as e:
                print(f"[CommandListener] error: {e} — retrying in 1s", file=sys.stderr)
                time.sleep(1)


def _trigger_pivot(username: str, pivot_script: str):
    """Execute the honeypot pivot script for the given user."""
    if not username:
        return
    print(f"[PIVOT] Pivoting user '{username}' to honeypot", file=sys.stderr)
    try:
        if not os.path.exists(pivot_script):
            print(f"[PIVOT] Script not found: {pivot_script}", file=sys.stderr)
            return
        subprocess.run(["sudo", pivot_script, username], timeout=10, check=False)
    except Exception as e:
        print(f"[PIVOT] Error executing pivot for '{username}': {e}", file=sys.stderr)


class SyscallLogger:
    """Orchestrates syscall collection, optional CSV output, and Redis publishing."""

    def __init__(
        self,
        source: str = "tracee",
        output_path: Optional[str] = None,
        hostname: str = "",
        buffer_size: int = 100,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
        redis_maxlen: int = 10000,
        pivot_script: str = "/usr/local/sbin/hp_pivot_user",
    ):
        self.source = source
        self.output_path = output_path
        self.hostname = hostname or os.uname().nodename
        self.buffer_size = buffer_size

        if source == "tracee":
            self.parser = TraceeParser(hostname=self.hostname)
        elif source == "auditd":
            self.parser = AuditdParser(hostname=self.hostname)
        else:
            raise ValueError(f"Unknown source: {source}")

        # Redis publisher (optional)
        self._redis_publisher: Optional[RedisPublisher] = None
        self._redis_batch: List[SyscallEvent] = []
        if redis_host:
            self._redis_publisher = RedisPublisher(
                host=redis_host,
                port=redis_port,
                hostname=self.hostname,
                stream_maxlen=redis_maxlen,
            )
            # Start command listener thread
            listener = CommandListener(
                host=redis_host,
                port=redis_port,
                hostname=self.hostname,
                pivot_script=pivot_script,
            )
            listener.start()

        self._event_count = 0
        self._running = True
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down...", file=sys.stderr)
        self._running = False

    def _flush_redis(self):
        if self._redis_publisher and self._redis_batch:
            self._redis_publisher.publish(self._redis_batch)
            self._redis_batch = []

    def process_stdin(self):
        """Process events from stdin (piped from Tracee)."""
        csv_writer = None
        if self.output_path:
            csv_writer = CSVWriter(self.output_path)
            csv_writer.__enter__()

        try:
            for line in sys.stdin:
                if not self._running:
                    break

                event = self.parser.parse_line(line)
                if not event:
                    continue

                self._event_count += 1
                self._process_event(event, csv_writer)

                if self._event_count % self.buffer_size == 0:
                    if csv_writer:
                        csv_writer.flush()
                    self._flush_redis()
                    self._print_stats()

        finally:
            self._flush_redis()
            if csv_writer:
                csv_writer.__exit__(None, None, None)
            self._print_final_stats()

    def process_file(self, log_path: str):
        """Process events from a log file (auditd)."""
        csv_writer = None
        if self.output_path:
            csv_writer = CSVWriter(self.output_path)
            csv_writer.__enter__()

        try:
            for event in self.parser.parse_file(log_path):
                if not self._running:
                    break
                self._event_count += 1
                self._process_event(event, csv_writer)
                if self._event_count % self.buffer_size == 0:
                    if csv_writer:
                        csv_writer.flush()
                    self._flush_redis()
        finally:
            self._flush_redis()
            if csv_writer:
                csv_writer.__exit__(None, None, None)
            self._print_final_stats()

    def _process_event(self, event: SyscallEvent, csv_writer: Optional[CSVWriter]):
        if csv_writer:
            csv_writer.write(event)
        if self._redis_publisher is not None:
            self._redis_batch.append(event)

    def _print_stats(self):
        print(f"[STATS] Events forwarded to central server: {self._event_count}", file=sys.stderr)

    def _print_final_stats(self):
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"KernelTrap Agent — Final Statistics", file=sys.stderr)
        print(f"Total events processed: {self._event_count}", file=sys.stderr)
        if self.output_path:
            print(f"CSV written to: {self.output_path}", file=sys.stderr)
        if self._redis_publisher:
            print(f"Events streamed to Redis stream: events.{self.hostname}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="KernelTrap Agent — collect syscall events and stream to central server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream to central server at 10.0.0.5:
  sudo tracee --output json | python syscall_logger.py --source tracee --redis-host 10.0.0.5

  # Standalone CSV only (no central server):
  sudo tracee --output json | python syscall_logger.py --source tracee --output events.csv

  # Both CSV backup and Redis streaming:
  sudo tracee --output json | python syscall_logger.py --source tracee \\
      --redis-host 10.0.0.5 --output /var/log/kerneltrap/events.csv
        """
    )

    parser.add_argument("--source", "-s", choices=["tracee", "auditd"], default="tracee")
    parser.add_argument("--output", "-o", help="Local CSV backup path (optional)")
    parser.add_argument("--auditd-log", default="/var/log/audit/audit.log")
    parser.add_argument("--hostname", default="", help="Override system hostname")
    parser.add_argument("--buffer-size", "-b", type=int, default=100,
                        help="Events to buffer before flushing to Redis (default: 100)")

    # Redis / central server options
    redis_group = parser.add_argument_group("Redis / central server")
    redis_group.add_argument("--redis-host", default=None,
                             help="Central server Redis host (enables streaming mode)")
    redis_group.add_argument("--redis-port", type=int, default=6379)
    redis_group.add_argument("--redis-maxlen", type=int, default=10000,
                             help="Max stream length per host (default: 10000)")
    redis_group.add_argument("--pivot-script", default="/usr/local/sbin/hp_pivot_user",
                             help="Path to honeypot pivot script")

    args = parser.parse_args()

    if not args.redis_host and not args.output:
        parser.error("At least one of --redis-host or --output must be specified")

    logger = SyscallLogger(
        source=args.source,
        output_path=args.output,
        hostname=args.hostname,
        buffer_size=args.buffer_size,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_maxlen=args.redis_maxlen,
        pivot_script=args.pivot_script,
    )

    if args.source == "auditd":
        logger.process_file(args.auditd_log)
    else:
        logger.process_stdin()


if __name__ == "__main__":
    main()
