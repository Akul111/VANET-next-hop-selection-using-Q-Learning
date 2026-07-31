import argparse
import socket
import struct
import threading
import time
import random
import os
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ───────── hyper-parameters ─────────

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
MEM_CAP = 50_000
TARGET_SYNC = 2.0  # seconds
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 600
SAVE_INTERVAL = 300  # seconds

GLOBAL_FEATS = 3  # x, y, dstDist
NEIGH_FEATS = 3   # dx, dy, ewma
INPUT_DIM = GLOBAL_FEATS + NEIGH_FEATS

# ────────────────────────────────────

Transition = namedtuple(
    "Transition",
    "g n a_idx a_feat r g2 n2 done"
)


# ───────── network: score one neighbour ─────────

class Scorer(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [*, 6]
        return self.net(x).squeeze(-1)  # -> [*]


# ───────── running normaliser ─────────

class Normaliser:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)

    def observe(self, v):
        self.n += 1
        delta = v - self.mean
        self.mean += delta / self.n
        self.var += delta * (v - self.mean)

    def norm(self, v):
        std = np.sqrt(self.var / max(self.n, 1))
        return (v - self.mean) / (std + 1e-6)


# ───────── agent wrapper ─────────

class Agent:
    def __init__(self, save_dir="./models"):
        self.q = Scorer()

        self.targ = Scorer()
        self.targ.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=LR)

        self.buf = deque(maxlen=MEM_CAP)

        self.gnorm = Normaliser(GLOBAL_FEATS)
        self.nnorm = Normaliser(NEIGH_FEATS)

        self.t0 = time.time()
        self.last_sync = self.last_save = self.t0

        self.step = 0
        self.loss_hist = deque(maxlen=100)

        self.lock = threading.Lock()

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self._load()

    # ── convenience wrappers around the buffer ──

    def push(self, *t):
        self.buf.append(Transition(*t))

    def __len__(self):
        return len(self.buf)

    def sample(self, k):
        tup = zip(*random.sample(self.buf, k))
        return Transition(*[list(x) for x in tup])

    # ── ε-greedy ──

    def eps(self):
        dt = time.time() - self.t0
        return EPS_END + (EPS_START - EPS_END) * np.exp(-dt / EPS_DECAY)

    # ── choose action ──

    def act(self, g, neigh):
        # g: [3], neigh: [N, 3]

        n = len(neigh)

        if n == 0:
            return -1

        self.gnorm.observe(g)

        if n:
            self.nnorm.observe(neigh.mean(axis=0))

        g_ = self.gnorm.norm(g)
        n_ = self.nnorm.norm(neigh)

        if random.random() < self.eps():
            return random.randrange(n)

        with torch.no_grad():
            g_rep = np.repeat(g_[None, :], n, axis=0)
            inp = torch.tensor(
                np.hstack([g_rep, n_]),
                dtype=torch.float32
            )
            q = self.q(inp)

        return int(torch.argmax(q).item())

    def remember(self, *args):
        self.push(*args)

        if self.lock.acquire(False):
            try:
                self._learn()
            finally:
                self.lock.release()

    def _learn(self):
        if len(self.buf) < BATCH_SIZE:
            return

        batch = self.sample(BATCH_SIZE)

        g = torch.tensor(np.asarray(batch.g), dtype=torch.float32)
        n_a = torch.tensor(np.asarray(batch.a_feat), dtype=torch.float32)
        g2 = torch.tensor(np.asarray(batch.g2), dtype=torch.float32)

        r = torch.tensor(
            np.asarray(batch.r),
            dtype=torch.float32
        )[:, None]

        d = torch.tensor(
            np.asarray(batch.done),
            dtype=torch.float32
        )[:, None]

        q_sa = self.q(torch.cat([g, n_a], dim=1))[:, None]

        # compute target

        max_q_next = []

        for gi, neigh_i in zip(batch.g2, batch.n2):
            if len(neigh_i) == 0:
                max_q_next.append(0.0)
                continue

            g_rep = np.repeat(
                self.gnorm.norm(gi)[None, :],
                len(neigh_i),
                axis=0
            )

            ni_ = self.nnorm.norm(neigh_i)

            qs = self.targ(
                torch.tensor(
                    np.hstack([g_rep, ni_]),
                    dtype=torch.float32
                )
            )

            max_q_next.append(float(qs.max().item()))

        max_q_next = torch.tensor(max_q_next)[:, None]

        tgt = r + (1 - d) * GAMMA * max_q_next

        loss = nn.functional.mse_loss(q_sa, tgt)

        self.opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.q.parameters(),
            1.0
        )

        self.opt.step()

        self.loss_hist.append(loss.item())
        self.step += 1

        now = time.time()

        if now - self.last_sync > TARGET_SYNC:
            self.targ.load_state_dict(self.q.state_dict())
            self.last_sync = now

        if now - self.last_save > SAVE_INTERVAL:
            self._save()
            self.last_save = now

        if self.step % 1000 == 0:
            print(
                f"steps {self.step:7d} "
                f"loss {np.mean(self.loss_hist):.4f} "
                f"eps {self.eps():.3f}"
            )

    # ── persistence ──

    def _save(self):
        path = f"{self.save_dir}/dqn_latest.pt"

        torch.save(
            {
                "q": self.q.state_dict(),
                "targ": self.targ.state_dict(),
                "opt": self.opt.state_dict(),
                "gnm": (
                    self.gnorm.mean,
                    self.gnorm.var,
                    self.gnorm.n,
                ),
                "nnm": (
                    self.nnorm.mean,
                    self.nnorm.var,
                    self.nnorm.n,
                ),
                "step": self.step,
            },
            path,
        )

        print(f"[✓] model saved → {path}")

    def _load(self):
        path = f"{self.save_dir}/dqn_latest.pt"

        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu")

            self.q.load_state_dict(ck["q"])
            self.targ.load_state_dict(ck["targ"])
            self.opt.load_state_dict(ck["opt"])

            (
                self.gnorm.mean,
                self.gnorm.var,
                self.gnorm.n,
            ) = ck["gnm"]

            (
                self.nnorm.mean,
                self.nnorm.var,
                self.nnorm.n,
            ) = ck["nnm"]

            self.step = ck["step"]

            print(f"[✓] model restored ({self.step} steps)")


# ───────── protocol helpers ─────────

HDR = struct.Struct("<BI")  # type u8 | len u32

HEAD_FMT = "<BdddI"   # ver x y dist N
NEI_FMT = "<Ifff"     # ip dx dy ewma
REW_HEAD = "<BddddI"  # ver R x y dist N


def recv_all(sock, n):
    data = b""

    while len(data) < n:
        chunk = sock.recv(n - len(data))

        if not chunk:
            raise ConnectionResetError

        data += chunk

    return data


# ───────── per-client thread ─────────

class Client(threading.Thread):
    def __init__(self, sock, addr, agent):
        super().__init__(daemon=True)

        self.s = sock
        self.addr = addr
        self.agent = agent

        self.prev_g = None
        self.prev_nei = None
        self.prev_a_idx = None
        self.prev_a_feat = None

    def run(self):
        print(f"[+] {self.addr} connected")

        try:
            while True:
                typ, ln = HDR.unpack(
                    recv_all(self.s, HDR.size)
                )

                pl = recv_all(self.s, ln)

                if typ == 1:
                    # STATE

                    ver, x, y, dist, N = struct.unpack_from(
                        HEAD_FMT,
                        pl,
                        0
                    )

                    off = struct.calcsize(HEAD_FMT)

                    nei = []

                    for _ in range(N):
                        ip, dx, dy, ew = struct.unpack_from(
                            NEI_FMT,
                            pl,
                            off
                        )

                        nei.append((dx, dy, ew))
                        off += struct.calcsize(NEI_FMT)

                    g = np.array(
                        [x, y, dist],
                        dtype=np.float32
                    )

                    nei_arr = np.array(
                        nei,
                        dtype=np.float32
                    )

                    a_idx = self.agent.act(g, nei_arr)

                    self.s.sendall(
                        struct.pack("<i", a_idx)
                    )

                    if 0 <= a_idx < len(nei_arr):
                        self.prev_g = g
                        self.prev_nei = nei_arr
                        self.prev_a_idx = a_idx
                        self.prev_a_feat = nei_arr[a_idx]
                    else:
                        self.prev_g = None
                        self.prev_nei = None
                        self.prev_a_idx = None
                        self.prev_a_feat = None

                elif typ == 2:
                    # REWARD

                    ver, R, x, y, dist, N = struct.unpack_from(
                        REW_HEAD,
                        pl,
                        0
                    )

                    off = struct.calcsize(REW_HEAD)

                    nei = []

                    for _ in range(N):
                        _, dx, dy, ew = struct.unpack_from(
                            NEI_FMT,
                            pl,
                            off
                        )

                        nei.append((dx, dy, ew))
                        off += struct.calcsize(NEI_FMT)

                    g2 = np.array(
                        [x, y, dist],
                        dtype=np.float32
                    )

                    nei2 = np.array(
                        nei,
                        dtype=np.float32
                    )

                    if self.prev_g is not None:
                        self.agent.remember(
                            self.prev_g,
                            self.prev_nei,
                            self.prev_a_idx,
                            self.prev_a_feat,
                            R,
                            g2,
                            nei2,
                            0.0,
                        )

                else:
                    print(f"[?] unknown msg type {typ}")

        except (ConnectionResetError, BrokenPipeError):
            print(f"[-] {self.addr} disconnected")

        finally:
            self.s.close()


# ─────────── entry point ───────────

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--host",
        default="0.0.0.0"
    )

    ap.add_argument(
        "--port",
        type=int,
        required=True
    )

    ap.add_argument(
        "--model-dir",
        default="./models"
    )

    args = ap.parse_args()

    agent = Agent(args.model_dir)

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    ) as srv:

        srv.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        srv.bind((args.host, args.port))
        srv.listen()

        print(
            f"[✓] server listening on "
            f"{args.host}:{args.port}"
        )

        try:
            while True:
                c, addr = srv.accept()
                Client(c, addr, agent).start()

        except KeyboardInterrupt:
            print("\n[!] shutting down – saving model")
            agent._save()


if __name__ == "__main__":
    main()