# VANET-next-hop-selection-using-Q-Learning



# VANET Next-Hop Selection using Deep Q-Learning

A Deep Q-Network (DQN) that learns to pick the next hop for routing in Vehicular Ad-hoc Networks (VANETs), built on top of **INET's AODV** and simulated in **OMNeT++ / INET / Veins** with **SUMO**-generated mobility. A custom OMNeT++ module overrides AODV's routing hooks and streams state and reward over a TCP socket to a standalone Python learner, which returns the neighbour to forward to. When the learner is unavailable or returns an invalid choice, the module falls back to vanilla AODV.


> **Scope of this repository.** This repo contains only the two custom components — the modified routing module (`QLearningAODV.cc`) and the learning server (`dqn.py`). `QLearningAODV` is a subclass of INET's stock `Aodv`; the base AODV sources, the module header/`.ned`, `omnetpp.ini`, and the Veins/SUMO scenario are **not** included. A working INET build is required to compile and run it.

---

## Results summary

Evaluated on a hybrid suburban scenario (two intersecting highways plus local roads, ~2 km², extracted from OpenStreetMap, IEEE 802.11p radios, 220 s runs) at three densities. Metrics: end-to-end delay (CDF), throughput (CDF), and packet delivery ratio (PDR).

| Density | End-to-end delay | PDR (DQN vs AODV) | Throughput |
|---------|------------------|-------------------|------------|
| **Small (5 nodes)** | Effectively identical | 38.03% vs 38.3% | Identical |
| **Medium (17 nodes)** | DQN lower: median 0.5→0.4 ms, p90 1.4→1.0 ms, p99 7.3→6.8 ms | 23.57% vs 23.34% | Near-identical |
| **Large (50 nodes)** | Effectively identical | **39.10% vs 31.73% (~8% gain)** | AODV higher (median ~4.1 vs 2.7 Mb/s) |

The DQN performs best at **medium density**, cutting tail latency while holding PDR. At large density it lifts PDR by roughly 8 points at a throughput cost, and at small density behaves much like AODV.

---

## Architecture

```
        OMNeT++ / INET / Veins / SUMO                 Python learner
   ┌──────────────────────────────────┐          ┌────────────────────┐
   │  node[i]                         │  STATE   │                    │
   │  ┌────────────────────────────┐  │ ───────► │   DQN Agent        │
   │  │ QLearningAODV : Aodv       │  │          │   (Scorer network) │
   │  │  • Netfilter hooks         │  │ ◄─────── │                    │
   │  │  • builds state vector     │  │  ACTION  │   replay buffer    │
   │  │  • non-blocking TCP client │  │          │   + online learning│
   │  │  • per-link EWMA stats     │  │  REWARD  │                    │
   │  └────────────────────────────┘  │ ───────► │                    │
   └──────────────────────────────────┘          └────────────────────┘
```

Each node opens its own TCP connection. The server runs **one shared agent** (single Q-network, target network, replay buffer and normalisers) and spawns one daemon thread per connection, so all nodes contribute experience to the same policy.

## How it works

### C++ side (`QLearningAODV`)

`QLearningAODV` derives from INET's `Aodv` and overrides the Netfilter hooks. At `INITSTAGE_LOCAL` it subscribes to packet signals; at `INITSTAGE_ROUTING_PROTOCOLS` it reads parameters, builds a lookup table of every other `node[i]`'s position (via the mobility module and the INET address resolver), and opens a non-blocking TCP socket to the learner.

Routing decisions are intercepted at:

- **`datagramLocalOutHook`** — locally originated packets
- **`datagramPreRoutingHook`** — packets to be forwarded

For unicast data packets it builds a **state**, sends it, polls for an **action** (an index into the neighbour list), and tags the packet with the chosen next hop via `NextHopAddressReq`. Broadcast/multicast, packets already at their destination, a disabled/disconnected learner, an empty neighbour list, or an out-of-range action index all cause it to **delegate to the base `Aodv` hook** (vanilla behaviour).

**Neighbour discovery** inspects the routing table, keeping routes learned by AODV whose next hop is neither the node itself nor already listed. This filtered, unique list both sizes the state vector and maps the returned action index back to an address.

### State representation

Global features (3): node `x`, `y`, and Euclidean `distance` to the final destination (sentinel `1e3` if the destination position is unknown). Per neighbour: relative displacement `dx, dy` and a link-reliability `EWMA`. Each neighbour's IP is also serialized but is used only C++-side to resolve the chosen index to an address — it is not a learning feature.

### Link quality (EWMA)

Each node keeps a per-neighbour exponentially-weighted moving average of delivery success, updated on every successful forward (`true`) and every drop (`false`), the latter driven by a subscription to `packetDroppedSignal`. This gives the learner a live reliability signal rather than pure geometry.

### Rewards

Computed C++-side and sent with the resulting state:

| Event | Reward |
|-------|--------|
| Successful forward (intermediate hop) | `hopRewardBase / 2` = **+2.5** |
| Delivery to final destination | `successReward` + `hopRewardBase / hopCount` = **+10 + 5/hops** |
| Drop | **−20** (`−dropPenalty`) |

Hop count on delivery is derived from the TTL decrement (`initialTTL − currentTTL`).

### Python side (`dqn.py`)

- **Scorer network** — an MLP (`6 → 128 → 128 → 1`, ReLU, Xavier-uniform init) that scores **one** neighbour at a time from `[global(3), neighbour(3)]`. To choose, every neighbour is scored and the argmax taken — keeping the action space variable-length instead of fixing a maximum neighbour count.
- **ε-greedy exploration** with time-based decay (`ε_end + (ε_start−ε_end)·exp(−t/τ)`), never reaching 0 so the agent keeps adapting to the changing topology.
- **Experience replay** (deque, capacity 50,000) and a **target network** synced every 2 s.
- **Running normalisers** (Welford-style) for global and neighbour features, so position (hundreds of metres) and EWMA (0–1) don't imbalance gradients.
- **Online learning** — a training step (`BATCH_SIZE = 128`, Adam, γ = 0.99, grad-clip 1.0) runs whenever a reward arrives and the learner lock is free.
- **Checkpointing** to `./models/dqn_latest.pt` on an interval and on Ctrl-C; restored on startup.

### Transition assembly

The learner is stateful per connection. On a `STATE` message it returns an action and caches `(g, neigh, a_idx, a_feat)`. On the next `REWARD` message it forms the transition `(s, N, a, N[a], r, s′, N′)` and pushes it to the replay buffer.

## Wire protocol (v2)

Little-endian throughout. Every message is a 5-byte frame header — `type (u8)` + `length (u32)` — followed by a payload.

- **STATE** (`type = 0x01`, simulator → agent): `ver(u8) x(f64) y(f64) dist(f64) N(u32)` then `N` × `ip(u32) dx(f32) dy(f32) ewma(f32)`
- **REWARD** (`type = 0x02`, simulator → agent): `ver(u8) R(f64) x(f64) y(f64) dist(f64) N(u32)` then `N` × neighbour records
- **ACTION** (agent → simulator): a single `int32` — chosen neighbour index, or `-1`

Per-neighbour records are 16 bytes. Protocol version is always `2`.

## Module parameters

Read in `initialize` from `omnetpp.ini` / `.ned`:

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `useQlearning` | Master on/off; off ⇒ pure AODV | `true` |
| `serverHost` | Learner IP (dotted-quad; parsed with `inet_pton`) | `"127.0.0.1"` |
| `serverPort` | Learner TCP port | `50000` |
| `successReward` | Reward for reaching the destination | `+10.0` |
| `hopRewardBase` | Hop-reward scaling factor | `+5.0` |
| `dropPenalty` | Penalty magnitude on drop | `−20.0` |


## Running

**1. Start the learner first** (the module connects at init):

```bash
python dqn.py --port 50000 [--host 0.0.0.0] [--model-dir ./models]
```

Requires Python 3, `numpy`, `torch`.

**2. Build INET** with `QLearningAODV` added to the AODV sources, register it in a `.ned`, point your nodes at it, and set the parameters above in `omnetpp.ini` (e.g. `serverHost = "127.0.0.1"`, `serverPort = 50000`, `useQlearning = true`).

**3. Run the simulation.** Each node connects and training proceeds online.
