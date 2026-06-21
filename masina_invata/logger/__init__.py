"""
KernelTrap Logger Module

Syscall collection and transformation for Isolation Forest anomaly detection.
"""

from .syscall_logger import (
    SyscallEvent,
    TraceeParser,
    AuditdParser,
    CSVWriter,
    SyscallLogger,
    SYSCALL_TO_ID,
    ID_TO_SYSCALL,
)

__all__ = [
    "SyscallEvent",
    "TraceeParser",
    "AuditdParser",
    "CSVWriter",
    "SyscallLogger",
    "SYSCALL_TO_ID",
    "ID_TO_SYSCALL",
]

__version__ = "1.0.0"
