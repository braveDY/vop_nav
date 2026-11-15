"""Track local TensorBoard SSH tunnels across this project's server profiles."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import tempfile
import time

from tasks import PROJECT, pidfd_open, pidfd_signal


def identity(pid, host, port, project, start=None):
    proc = Path('/proc') / str(pid)
    try:
        if proc.stat().st_uid != os.getuid() or (proc / 'exe').resolve(strict=True).name != 'ssh':
            return None
        if (proc / 'cwd').resolve(strict=True) != project:
            return None
        argv = [os.fsdecode(x) for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        birth = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        if start is not None and str(start) != birth:
            return None
        if (not argv or argv[-1] != host or '-N' not in argv
                or not any(argv[i:i + 2] == ['-L', f'{port}:localhost:{port}'] for i in range(len(argv) - 1))):
            return None
        return birth
    except (OSError, ValueError):
        return None


def record_path(project, port):
    return project / '.ops' / f'tensorboard_tunnel_{port}.json'


def register(pid, host, port, project=PROJECT):
    project = Path(project).resolve()
    birth = identity(pid, host, port, project)
    if birth is None:
        raise RuntimeError('SSH tunnel check failed. Process not recorded')
    path = record_path(project, port)
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(dict(pid=pid, host=host, port=port, start=birth)))
    temporary.replace(path)


def launch(host, port, project=PROJECT):
    """Create the session without a shell/setsid fork changing the tracked PID."""
    project = Path(project).resolve()
    worker = subprocess.Popen(['ssh', '-o', 'ControlPath=none', '-o', 'ExitOnForwardFailure=yes',
        '-o', 'ConnectTimeout=10', '-N', '-L', f'{port}:localhost:{port}', host], cwd=project,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True, close_fds=True)
    try:
        deadline = time.monotonic() + 5
        while identity(worker.pid, host, port, project) is None:
            if worker.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('SSH tunnel failed to start')
            time.sleep(.02)
        register(worker.pid, host, port, project)
    except BaseException:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)
        raise
    return worker


def managed_tunnels(port, project):
    options = ['-o', 'ControlPath=none', '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
               '-N', '-L', f'{port}:localhost:{port}']
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [os.fsdecode(x) for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            if len(argv) != len(options) + 2 or argv[1:-1] != options:
                continue
            pid, host = int(proc.name), argv[-1]
            birth = identity(pid, host, port, project)
            if birth is not None:
                yield pid, host, birth
        except (OSError, ValueError):
            continue


def port_free(port):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(('127.0.0.1', port))
        except OSError:
            return False
    return True


def stop(port, project=PROJECT, legacy_dir=None):
    project = Path(project).resolve()
    path = record_path(project, port)
    records = []
    if path.exists():
        data = json.loads(path.read_text())
        records.append((path, int(data['pid']), data['host'], data.get('start')))
    # Migrate the previous host-keyed PID records, including a different active profile.
    directory = Path(legacy_dir) if legacy_dir is not None else Path(tempfile.gettempdir())
    for legacy in directory.glob(f'.abs_tb_*_{port}.pid'):
        if legacy.stat().st_uid != os.getuid():
            continue
        try:
            pid = int(legacy.read_text().strip())
            argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
            host = os.fsdecode(next(x for x in reversed(argv) if x))
            key = ''.join(c if c.isascii() and (c.isalnum() or c == '_') else '_' for c in f'{host}_{port}')
            if legacy.name == f'.abs_tb_{key}.pid':
                records.append((legacy, pid, host, None))
        except (OSError, ValueError, StopIteration):
            continue
    known = {record[1] for record in records}
    for pid, host, birth in managed_tunnels(port, project):
        if pid not in known:
            records.append((None, pid, host, birth))
    for marker, pid, host, birth in records:
        if identity(pid, host, port, project, birth) is None:
            if marker == path:
                marker.unlink(missing_ok=True)
            continue
        try:
            handle = pidfd_open(pid)
        except ProcessLookupError:
            if marker:
                marker.unlink(missing_ok=True)
            continue
        try:
            if identity(pid, host, port, project, birth) is None:
                continue
            pidfd_signal(handle, signal.SIGTERM)
            poller = select.poll()
            poller.register(handle, select.POLLIN)
            if not poller.poll(5000):
                raise RuntimeError(f'Old SSH tunnel for this project did not stop. PID: {pid}')
            print(f'Stopped the local TensorBoard tunnel for {host} in this project (PID: {pid})', flush=True)
            if marker:
                marker.unlink(missing_ok=True)
        finally:
            os.close(handle)


def tunnel_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stop', 'register', 'launch', 'port-free'))
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--host')
    args = parser.parse_args()
    try:
        if args.action == 'stop':
            stop(args.port)
        elif args.action == 'port-free':
            if not port_free(args.port):
                parser.exit(1)
        elif args.action == 'launch':
            if not args.host:
                parser.error('launch requires --host')
            worker = launch(args.host, args.port)
            print(worker.pid, flush=True)
        else:
            if args.pid is None or not args.host:
                parser.error('register requires --pid and --host')
            register(args.pid, args.host, args.port)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'TensorBoard tunnel command failed: {error}\n')



import struct
import sys
import threading

def run_remote_bridge(tcp_port: int = 47997, target_udp_port: int = 47998) -> None:
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", tcp_port))
    server_sock.listen(5)
    print(f"[Bridge-Remote] Listening on TCP 127.0.0.1:{tcp_port} (forwarding to UDP :{target_udp_port})...", flush=True)

    while True:
        try:
            conn, _ = server_sock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("[Bridge-Remote] Local client connected via TCP tunnel.", flush=True)
            _handle_remote_connection(conn, target_udp_port)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Bridge-Remote] Error accepting connection: {e}", flush=True)
            time.sleep(0.5)


def _handle_remote_connection(conn: socket.socket, target_udp_port: int) -> None:
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_target = ("127.0.0.1", target_udp_port)
    stop_event = threading.Event()

    def udp_to_tcp():
        while not stop_event.is_set():
            try:
                udp_sock.settimeout(0.5)
                data, _ = udp_sock.recvfrom(65535)
                header = struct.pack("!H", len(data))
                conn.sendall(header + data)
            except socket.timeout:
                continue
            except Exception:
                break
        stop_event.set()

    t = threading.Thread(target=udp_to_tcp, daemon=True)
    t.start()

    try:
        while not stop_event.is_set():
            hdr = _recv_all(conn, 2)
            if not hdr:
                break
            length = struct.unpack("!H", hdr)[0]
            data = _recv_all(conn, length)
            if not data:
                break
            udp_sock.sendto(data, udp_target)
    except Exception as e:
        print(f"[Bridge-Remote] TCP link closed: {e}", flush=True)
    finally:
        stop_event.set()
        try:
            conn.close()
        except Exception:
            pass
        try:
            udp_sock.close()
        except Exception:
            pass
        print("[Bridge-Remote] Session ended, ready for next connection.", flush=True)


def run_local_bridge(listen_udp_port: int = 47998, tcp_target_port: int = 47997) -> None:
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_sock.bind(("127.0.0.1", listen_udp_port))
    print(f"[Bridge-Local] Bound local UDP 127.0.0.1:{listen_udp_port}. Target TCP tunnel :{tcp_target_port}...", flush=True)

    client_addr = None

    try:
        while True:
            tcp_sock = None
            while True:
                try:
                    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    tcp_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    tcp_sock.connect(("127.0.0.1", tcp_target_port))
                    print("[Bridge-Local] TCP tunnel connected successfully!", flush=True)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception:
                    if tcp_sock:
                        try:
                            tcp_sock.close()
                        except Exception:
                            pass
                    time.sleep(1)

            stop_event = threading.Event()

            def tcp_to_udp():
                nonlocal client_addr
                while not stop_event.is_set():
                    try:
                        hdr = _recv_all(tcp_sock, 2)
                        if not hdr:
                            break
                        length = struct.unpack("!H", hdr)[0]
                        data = _recv_all(tcp_sock, length)
                        if not data:
                            break
                        if client_addr:
                            udp_sock.sendto(data, client_addr)
                    except Exception:
                        break
                stop_event.set()

            t = threading.Thread(target=tcp_to_udp, daemon=True)
            t.start()

            try:
                while not stop_event.is_set():
                    udp_sock.settimeout(0.5)
                    try:
                        data, addr = udp_sock.recvfrom(65535)
                        client_addr = addr
                        header = struct.pack("!H", len(data))
                        tcp_sock.sendall(header + data)
                    except socket.timeout:
                        continue
                    except (BrokenPipeError, ConnectionResetError, OSError) as e:
                        print(f"[Bridge-Local] TCP tunnel disconnected: {e}. Reconnecting...", flush=True)
                        break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[Bridge-Local] Session exception: {e}. Reconnecting...", flush=True)
            finally:
                stop_event.set()
                try:
                    tcp_sock.close()
                except Exception:
                    pass
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[Bridge-Local] Stopped by user.", flush=True)
    finally:
        udp_sock.close()


def _recv_all(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def bridge_main() -> None:
    parser = argparse.ArgumentParser(description="WebRTC UDP-over-TCP tunnel bridge")
    parser.add_argument("mode", choices=["local", "remote"], help="Bridge run mode")
    parser.add_argument("--udp_port", type=int, default=47998, help="UDP media stream port")
    parser.add_argument("--tcp_port", type=int, default=47997, help="TCP tunnel bridge port")
    args = parser.parse_args()

    if args.mode == "remote":
        run_remote_bridge(tcp_port=args.tcp_port, target_udp_port=args.udp_port)
    else:
        run_local_bridge(listen_udp_port=args.udp_port, tcp_target_port=args.tcp_port)



if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('tb', 'bridge'):
        raise SystemExit('Usage: streaming.py tb <action> | bridge <local|remote>')
    mode = sys.argv.pop(1)
    tunnel_main() if mode == 'tb' else bridge_main()
