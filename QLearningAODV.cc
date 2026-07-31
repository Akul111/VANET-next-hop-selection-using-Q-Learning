//
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Based on the AODV implementation of the INET Framework.
//



/*
 * QLearningAODV.cc – AODV with Deep-Q-Learning next-hop selection (v2)
 * --------------------------------------------------------------------
 *  Adds a per-neighbour delivery-ratio EWMA to the state sent to Python
 *  Sends protocol-version byte and 16-byte neighbour records
 *  Updates EWMA after every successful forward / drop
 *
 *  NOTE: (PROTO_VERSION = 2).
 */
#include "QLearningAODV.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/packet/Packet.h"
#include "inet/mobility/contract/IMobility.h"
#include "inet/networklayer/common/L3Tools.h"
#include "inet/networklayer/common/NextHopAddressTag_m.h"
#include "inet/networklayer/ipv4/Ipv4Header_m.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "inet/linklayer/common/InterfaceTag_m.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/Simsignals.h"


namespace inet {
namespace aodv {
Define_Module(QLearningAODV);
/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */
namespace {
inline void updateEwma(QLearningAODV::LinkStat& s, bool success)
{
    double oldEwma = s.ewma;
    s.ewma = QLearningAODV::LinkStat::alpha * s.ewma +
             (1.0 - QLearningAODV::LinkStat::alpha) * (success ? 1.0 : 0.0);
    EV_WARN << "[DQN] Updated EWMA: " << oldEwma << " -> " << s.ewma
            << " (success=" << (success ? "true" : "false") << ")" << '\n';
}
} // anonymous namespace
/* ------------------------------------------------------------------ */
/*  Raw socket framing (5-byte header supplied by caller)             */
/* ------------------------------------------------------------------ */
bool QLearningAODV::sendBufferToServer(MsgType t,
                                       const char *buf,
                                       uint32_t    len)
{
    if (!socketConnected) {
        EV_ERROR << "[DQN] Cannot send buffer: socket not connected" << '\n';
        return false;
    }

    char hdr[5];
    hdr[0] = static_cast<uint8_t>(t);
    std::memcpy(hdr + 1, &len, sizeof(uint32_t));

    EV_WARN << "[DQN] Preparing to send " << len << " bytes to server, message type: "
            << (t == STATE ? "STATE" : "REWARD") << '\n';

    auto sendAll = [&](const char *data, size_t n) -> bool {
        size_t done = 0;
        while (done < n) {
            ssize_t s = ::send(socketFd, data + done, n - done, 0);
            if (s <= 0) {
                EV_ERROR << "[DQN] send() failed: " << strerror(errno) << ", bytes sent: " << done << " of " << n << '\n';
                socketConnected = false;
                return false;
            }
            done += s;
            EV_WARN << "[DQN] Send progress: " << done << "/" << n << " bytes" << '\n';
        }
        return true;
    };

    bool headerOk = sendAll(hdr, sizeof hdr);
    if (!headerOk) {
        EV_ERROR << "[DQN] Failed to send header" << '\n';
        return false;
    }

    bool dataOk = sendAll(buf, len);
    if (!dataOk) {
        EV_ERROR << "[DQN] Failed to send data payload" << '\n';
        return false;
    }

    bool ok = headerOk && dataOk;
    EV_ERROR << "[DQN] >> "
            << (t == STATE ? "STATE  " : "REWARD ")
            << len << " B   ok=" << ok << '\n';
    return ok;
}
/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */
QLearningAODV::QLearningAODV() {
    EV_WARN << "[DQN] QLearningAODV constructor called" << '\n';
}

QLearningAODV::~QLearningAODV() {
    EV_WARN << "[DQN] QLearningAODV destructor called, closing socket" << '\n';
    closeSocket();
}

void QLearningAODV::initialize(int stage)
{
    EV_WARN << "[DQN] Initializing QLearningAODV, stage: " << stage << '\n';
    Aodv::initialize(stage);
    /* ---------------- INITSTAGE_LOCAL ---------------- */
    if (stage == INITSTAGE_LOCAL) {
        EV_WARN << "[DQN] Initialize LOCAL stage" << '\n';

        cModule *hostModule = getContainingNode(this);
        hostModule->subscribe(packetDroppedSignal, this);
        hostModule->subscribe(packetSentToLowerSignal, this);
    }
    /* ------------ INITSTAGE_ROUTING_PROTOCOLS -------- */
    if (stage == INITSTAGE_ROUTING_PROTOCOLS) {
        EV_WARN << "[DQN] Initialize ROUTING_PROTOCOLS stage" << '\n';
        /* 1. read parameters */
        serverHost    = par("serverHost").stdstringValue();
        serverPort    = par("serverPort");
        useQlearning  = par("useQlearning");
        successReward = par("successReward");
        hopRewardBase = par("hopRewardBase");
        dropPenalty   = par("dropPenalty");

        EV_WARN << "[DQN] Parameters: serverHost=" << serverHost
                << ", serverPort=" << serverPort
                << ", useQlearning=" << (useQlearning ? "true" : "false")
                << ", successReward=" << successReward
                << ", hopRewardBase=" << hopRewardBase
                << ", dropPenalty=" << dropPenalty << '\n';

        /* 2. record every other node's position (once) */
        cModule *root = getSimulation()->getSystemModule();
        cModule *self = getContainingNode(this);
        int nodeCount = 0;

        EV_WARN << "[DQN] Mapping node positions..." << '\n';
        for (int i = 0;; ++i) {
            cModule *n = root->getSubmodule("host", i);
            if (!n) {
                EV_WARN << "[DQN] No more hosts found, processed " << nodeCount << " nodes" << '\n';
                break;
            }
            if (n == self) {
                EV_WARN << "[DQN] Skipping self node at index " << i << '\n';
                continue;
            }
            auto mob = check_and_cast<IMobility*>(n->getSubmodule("mobility"));
            Coord pos = mob->getCurrentPosition();
            L3Address ip = L3AddressResolver().addressOf(n);
            if (!ip.isUnspecified()) {
                nodePositions[ip] = {pos.x, pos.y};
                EV_ERROR << "[DQN] pos " << ip << " → (" << pos.x << ',' << pos.y << ")\n";
                nodeCount++;
            } else {
                EV_ERROR << "[DQN] Warning: Unspecified IP address for node at index " << i << '\n';
            }
        }
        EV_WARN << "[DQN] Total node positions mapped: " << nodeCount << '\n';

        /* 3. open TCP link to the learner */
        if (useQlearning) {
            EV_WARN << "[DQN] Q-learning enabled, connecting to server..." << '\n';
            if (initializeSocket() && connectToServer()) {
                socketConnected = true;
                EV_ERROR << "[DQN] socket connected to "
                        << serverHost << ':' << serverPort << '\n';
            } else {
                useQlearning = false;
                EV_WARN << "[DQN] disabled – socket error\n";
            }
        } else {
            EV_WARN << "[DQN] Q-learning disabled by configuration" << '\n';
        }
    }
}
/* ------------------------------------------------------------------ */
/*  BSD socket helpers                                                */
/* ------------------------------------------------------------------ */
bool QLearningAODV::initializeSocket()
{
    EV_WARN << "[DQN] Initializing socket..." << '\n';
    socketFd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd < 0) {
        EV_ERROR << "[DQN] socket(): " << strerror(errno) << '\n';
        return false;
    }
    EV_WARN << "[DQN] Socket created successfully with descriptor: " << socketFd << '\n';

    std::memset(&serverAddr, 0, sizeof serverAddr);
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port   = htons(serverPort);

    EV_WARN << "[DQN] Resolving server address: " << serverHost << '\n';
    if (inet_pton(AF_INET, serverHost.c_str(), &serverAddr.sin_addr) <= 0) {
        EV_ERROR << "[DQN] inet_pton(" << serverHost << "): "
                 << strerror(errno) << '\n';
        ::close(socketFd); socketFd = -1;
        return false;
    }
    EV_WARN << "[DQN] Server address resolved successfully" << '\n';
    return true;
}
bool QLearningAODV::connectToServer()
{
    EV_WARN << "[DQN] Connecting to server at " << serverHost << ":" << serverPort << "..." << '\n';
    if (::connect(socketFd, (sockaddr*)&serverAddr, sizeof serverAddr) < 0) {
        EV_ERROR << "[DQN] connect(): " << strerror(errno) << '\n';
        ::close(socketFd); socketFd = -1;
        return false;
    }

    EV_WARN << "[DQN] Connected successfully, setting socket to non-blocking mode" << '\n';
    int fl = fcntl(socketFd, F_GETFL, 0);
    fcntl(socketFd, F_SETFL, fl | O_NONBLOCK);
    EV_WARN << "[DQN] Socket set to non-blocking mode" << '\n';
    return true;
}
void QLearningAODV::closeSocket()
{
    if (socketFd >= 0) {
        EV_WARN << "[DQN] Closing socket with descriptor: " << socketFd << '\n';
        ::close(socketFd);
        socketFd = -1;
        socketConnected = false;
        EV_WARN << "[DQN] Socket closed successfully" << '\n';
    } else {
        EV_WARN << "[DQN] Socket already closed or invalid" << '\n';
    }
}
/* ------------------------------------------------------------------ */
/*  Neighbour discovery                                               */
/* ------------------------------------------------------------------ */
std::vector<L3Address> QLearningAODV::getAvailableNeighbors()
{
    EV_WARN << "[DQN] Looking for available neighbors..." << '\n';
    std::vector<L3Address> v;
    for (int i = 0; i < routingTable->getNumRoutes(); ++i) {
        IRoute *r = routingTable->getRoute(i);
        if (r->getSourceType() != IRoute::AODV) {
            EV_WARN << "[DQN] Route " << i << " is not AODV source type, skipping" << '\n';
            continue;
        }

        L3Address nh = r->getNextHopAsGeneric();
        if (!nh.isUnspecified() &&
            nh != getSelfIPAddress() &&
            std::find(v.begin(), v.end(), nh) == v.end()) {
            v.push_back(nh);
            EV_WARN << "[DQN] Found neighbor: " << nh << '\n';
        }
    }
    EV_ERROR << "[DQN] neighbours = " << v.size() << '\n';
    return v;
}
/* ------------------------------------------------------------------ */
/*  Build state                                                       */
/* ------------------------------------------------------------------ */
QLearningAODV::State
QLearningAODV::getCurrentState(const L3Address& dest)
{
    EV_WARN << "[DQN] Building current state for destination: " << dest << '\n';
    State s;
    /* own position ------------------------------------------------- */
    Coord selfPos = check_and_cast<IMobility*>
        (getContainingNode(this)->getSubmodule("mobility"))
        ->getCurrentPosition();
    s.x = selfPos.x;
    s.y = selfPos.y;
    EV_WARN << "[DQN] Current position: (" << s.x << ", " << s.y << ")" << '\n';

    /* distance to final destination -------------------------------- */
    auto it = nodePositions.find(dest);
    if (it == nodePositions.end()) {
        s.destinationDist = 1e3;   // sentinel "far away"
        EV_WARN << "[DQN] Destination position unknown, using sentinel distance: " << s.destinationDist << '\n';
    } else {
        s.destinationDist = std::hypot(selfPos.x - it->second.first,
                                       selfPos.y - it->second.second);
        EV_WARN << "[DQN] Distance to destination: " << s.destinationDist << '\n';
    }

    /* neighbour feature list --------------------------------------- */
    EV_WARN << "[DQN] Building neighbor feature list..." << '\n';
    for (const L3Address& n : getAvailableNeighbors()) {
        NeighborFeat f{};
        f.ip  = n.toIpv4().getInt();
        auto pIt = nodePositions.find(n);
        if (pIt != nodePositions.end()) {
            f.dx = pIt->second.first  - selfPos.x;
            f.dy = pIt->second.second - selfPos.y;
            EV_WARN << "[DQN] Neighbor " << n << " at relative position: (" << f.dx << ", " << f.dy << ")" << '\n';
        } else {
            f.dx = f.dy = 0.0f;
            EV_ERROR << "[DQN] Warning: Position of neighbor " << n << " unknown, using (0,0)" << '\n';
        }
        f.linkEwma = static_cast<float>(linkStats[n].ewma);
        EV_WARN << "[DQN] Neighbor " << n << " link EWMA: " << f.linkEwma << '\n';
        s.neighbors.push_back(f);
    }
    EV_WARN << "[DQN] State built with " << s.neighbors.size() << " neighbors" << '\n';
    return s;
}
/* ------------------------------------------------------------------ */
/*  Serialise state / reward (protocol v2)                            */
/* ------------------------------------------------------------------ */
bool QLearningAODV::sendStateToServer(const State& st)
{
    EV_WARN << "[DQN] Serializing state to send to server..." << '\n';
    const uint32_t n = st.neighbors.size();
    const std::size_t len = 1 + 24 + 4 + n * NEIGH_REC_BYTES; // 1(version)+x,y,dist+count
    EV_WARN << "[DQN] State buffer size: " << len << " bytes for " << n << " neighbors" << '\n';

    std::vector<char> buf(len);
    std::size_t o = 0;

    /* fixed header ------------------------------------------------- */
    buf[o++] = PROTO_VERSION;
    EV_WARN << "[DQN] Protocol version: " << static_cast<int>(PROTO_VERSION) << '\n';

    std::memcpy(&buf[o], &st.x, 8); o += 8;
    std::memcpy(&buf[o], &st.y, 8); o += 8;
    std::memcpy(&buf[o], &st.destinationDist, 8); o += 8;
    std::memcpy(&buf[o], &n, 4); o += 4;

    /* neighbour list ---------------------------------------------- */
    EV_WARN << "[DQN] Adding " << n << " neighbors to state buffer" << '\n';
    for (const auto& nf : st.neighbors) {
        std::memcpy(&buf[o], &nf.ip, 4); o += 4;
        std::memcpy(&buf[o], &nf.dx, 4); o += 4;
        std::memcpy(&buf[o], &nf.dy, 4); o += 4;
        std::memcpy(&buf[o], &nf.linkEwma, 4); o += 4;

        Ipv4Address ip(nf.ip);
        EV_WARN << "[DQN] Added neighbor: " << ip
                << " dx=" << nf.dx << " dy=" << nf.dy
                << " ewma=" << nf.linkEwma << '\n';
    }

    EV_WARN << "[DQN] State buffer prepared, sending to server" << '\n';
    return sendBufferToServer(STATE, buf.data(), len);
}
bool QLearningAODV::sendRewardToServer(double reward, const State& st)
{
    EV_WARN << "[DQN] Serializing reward " << reward << " to send to server..." << '\n';
    const uint32_t n = st.neighbors.size();
    const std::size_t len = 1 + 8 + 24 + 4 + n * NEIGH_REC_BYTES;
    EV_WARN << "[DQN] Reward buffer size: " << len << " bytes for " << n << " neighbors" << '\n';

    std::vector<char> buf(len);
    std::size_t o = 0;
    buf[o++] = PROTO_VERSION;
    std::memcpy(&buf[o], &reward, 8); o += 8;
    std::memcpy(&buf[o], &st.x, 8); o += 8;
    std::memcpy(&buf[o], &st.y, 8); o += 8;
    std::memcpy(&buf[o], &st.destinationDist, 8); o += 8;
    std::memcpy(&buf[o], &n, 4); o += 4;

    EV_WARN << "[DQN] Adding " << n << " neighbors to reward buffer" << '\n';
    for (const auto& nf : st.neighbors) {
        std::memcpy(&buf[o], &nf.ip, 4); o += 4;
        std::memcpy(&buf[o], &nf.dx, 4); o += 4;
        std::memcpy(&buf[o], &nf.dy, 4); o += 4;
        std::memcpy(&buf[o], &nf.linkEwma, 4); o += 4;

        Ipv4Address ip(nf.ip);
        EV_WARN << "[DQN] Added neighbor to reward: " << ip << '\n';
    }

    EV_WARN << "[DQN] Reward buffer prepared, sending to server" << '\n';
    return sendBufferToServer(REWARD, buf.data(), len);
}
/* ------------------------------------------------------------------ */
/*  Receive one int32 action (non-blocking)                           */
/* ------------------------------------------------------------------ */
int QLearningAODV::receiveActionFromServer()
{
    EV_WARN << "[DQN] Attempting to receive action from server..." << '\n';
    int32_t act;
    ssize_t r = ::recv(socketFd, &act, sizeof act, 0);
    if (r == static_cast<ssize_t>(sizeof act)) {
        EV_ERROR << "[DQN] << action = " << act << '\n';
        return act;
    }
    if (r == 0) {
        EV_WARN << "[DQN] server closed connection\n";
        socketConnected = false;
    } else if (r < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            EV_WARN << "[DQN] No data available yet (non-blocking mode)" << '\n';
        } else {
            EV_WARN << "[DQN] recv(): " << strerror(errno) << '\n';
            socketConnected = false;
        }
    } else {
        EV_ERROR << "[DQN] Received incomplete data: " << r << " bytes" << '\n';
    }
    return -1;
}
/* ------------------------------------------------------------------ */
/*  Ask learner for an action                                         */
/* ------------------------------------------------------------------ */
int QLearningAODV::getActionFromDQN(const State& st)
{
    EV_WARN << "[DQN] Requesting action from DQN for state with "
            << st.neighbors.size() << " neighbors" << '\n';

    if (st.neighbors.empty()) {
        EV_ERROR << "[DQN] no neighbours – cannot query learner\n";
        return -1;
    }

    if (!sendStateToServer(st)) {
        EV_WARN << "[DQN] state send failed\n";
        return -1;
    }

    /* Poll (back-off) --------------------------------------------- */
    EV_WARN << "[DQN] Starting polling with exponential backoff..." << '\n';
    for (int i = 0; i < 10; ++i) {
        int delay = 10'000 * (i + 1);
        EV_WARN << "[DQN] Poll attempt " << (i+1) << "/10, waiting " << delay << " microseconds" << '\n';

        usleep(delay);
        int a = receiveActionFromServer();

        if (a >= 0) {
            EV_WARN << "[DQN] Received valid action: " << a << " after " << (i+1) << " attempts" << '\n';
            return a;
        }

        if (!socketConnected) {
            EV_ERROR << "[DQN] Socket disconnected during polling" << '\n';
            break;
        }
    }

    EV_WARN << "[DQN] timeout waiting for action\n";
    return -1;
}
/* ------------------------------------------------------------------ */
/*  Select next hop                                                   */
/* ------------------------------------------------------------------ */
L3Address QLearningAODV::selectNextHop(const L3Address& dest)
{
    EV_WARN << "[DQN] Selecting next hop for destination: " << dest << '\n';

    if (!(useQlearning && socketConnected)) {
        EV_WARN << "[DQN] Q-learning disabled or socket not connected, falling back to vanilla AODV" << '\n';
        return L3Address();   // fall back to vanilla AODV
    }

    State st = getCurrentState(dest);
    if (st.neighbors.empty()) {
        EV_WARN << "[DQN] No neighbors available, falling back to vanilla AODV" << '\n';
        return L3Address();
    }

    EV_WARN << "[DQN] Requesting action from DQN for " << st.neighbors.size() << " neighbors" << '\n';
    int idx = getActionFromDQN(st);

    if (idx >= 0 && idx < static_cast<int>(st.neighbors.size())) {
        Ipv4Address ip(st.neighbors[idx].ip);
        EV_ERROR << "[DQN] chose neighbour[" << idx << "] = " << ip << '\n';
        EV_WARN << "[DQN] Selected neighbor EWMA: " << st.neighbors[idx].linkEwma << '\n';
        return L3Address(ip);
    }

    EV_ERROR << "[DQN] learner returned invalid idx=" << idx << '\n';
    EV_WARN << "[DQN] Falling back to vanilla AODV due to invalid DQN action" << '\n';
    return L3Address();
}
/* ------------------------------------------------------------------ */
/*  Packet bookkeeping & rewards                                      */
/* ------------------------------------------------------------------ */
QLearningAODV::PacketId QLearningAODV::getPacketId(Packet *p)
{
    EV_WARN << "[DQN] Generating packet ID for packet " << p->getId() << '\n';

    const auto& ipHdr = getNetworkProtocolHeader(p);
    if (!ipHdr) {
        EV_WARN << "[DQN] No IP header found, using generic ID" << '\n';
        return "unknown-" + std::to_string(p->getId());
    }

    std::ostringstream ss;
    ss << ipHdr->getSourceAddress() << '-'
       << ipHdr->getDestinationAddress() << '-'
       << p->getId();

    EV_WARN << "[DQN] Generated packet ID: " << ss.str() << '\n';
    return ss.str();
}
void QLearningAODV::recordPacket(Packet *p, const L3Address& nhop)
{
    EV_WARN << "[DQN] Recording packet " << p->getId() << " with next hop " << nhop << '\n';

    const auto ipv4 = dynamicPtrCast<const Ipv4Header>(getNetworkProtocolHeader(p));
    if (!ipv4) {
        EV_ERROR << "[DQN] Cannot record packet: no IPv4 header found" << '\n';
        return;
    }

    PacketRecord rec;
    rec.initialTTL      = ipv4->getTimeToLive();
    rec.currentHopCount = 0;
    rec.creationTime    = simTime();
    rec.lastHop         = nhop;
    rec.destination     = ipv4->getDestinationAddress();

    PacketId id = getPacketId(p);
    packetRecords[id] = rec;

    EV_WARN << "[DQN] Recorded packet " << id
            << " with hop count " << static_cast<int>(rec.currentHopCount)
            << " at time " << rec.creationTime
            << " to destination " << rec.destination << '\n';
}
/* --------------------- reward helpers ----------------------------- */
void QLearningAODV::rewardForwarding(Packet *p)
{
    PacketId id = getPacketId(p);
    EV_WARN << "[DQN] Processing forward reward for packet " << id << '\n';

    auto it = packetRecords.find(id);
    if (it == packetRecords.end()) {
        EV_ERROR << "[DQN] Cannot reward forwarding: packet record not found" << '\n';
        return;
    }

    double R = hopRewardBase / 2.0;
    EV_WARN << "[DQN] Forwarding reward: " << R
            << " for destination " << it->second.destination << '\n';

    State currentState = getCurrentState(it->second.destination);
    sendRewardToServer(R, currentState);

    /* EWMA success update */
    EV_WARN << "[DQN] Updating EWMA for successful forward via " << it->second.lastHop << '\n';
    updateEwma(linkStats[it->second.lastHop], true);
}
void QLearningAODV::rewardDelivery(Packet *p)
{
    PacketId id = getPacketId(p);
    EV_WARN << "[DQN] Processing delivery reward for packet " << id << '\n';

    auto it = packetRecords.find(id);
    if (it == packetRecords.end()) {
        EV_ERROR << "[DQN] Cannot reward delivery: packet record not found" << '\n';
        return;
    }


    const auto ipv4 = dynamicPtrCast<const Ipv4Header>(getNetworkProtocolHeader(p));
        if (!ipv4) {
            EV_ERROR << "[DQN] Cannot compute hops: no IPv4 header found" << '\n';
        } else {
            uint8_t currentTTL = ipv4->getTimeToLive();
            int hopsTaken = it->second.initialTTL - currentTTL;
            it->second.currentHopCount = hopsTaken;
            EV_WARN << "[DQN] Computed hopsTaken: " << hopsTaken << '\n';
        }


        double R = successReward;
            if (it->second.currentHopCount > 0) {
                double hopBonus = hopRewardBase / it->second.currentHopCount;
                R += hopBonus;
                EV_WARN << "[DQN] Adding hop-based component to reward: " << hopBonus << '\n';
            }


    EV_WARN << "[DQN] Delivery reward: " << R
            << " for destination " << it->second.destination
            << " with hop count " << static_cast<int>(it->second.currentHopCount) << '\n';

    State currentState = getCurrentState(it->second.destination);
    sendRewardToServer(R, currentState);

    EV_WARN << "[DQN] Removing packet record after delivery" << '\n';
    packetRecords.erase(it);
}
void QLearningAODV::penalizeDropping(Packet *p)
{
    PacketId id = getPacketId(p);
    EV_WARN << "[DQN] Processing drop penalty for packet " << id << '\n';

    auto it = packetRecords.find(id);
    if (it == packetRecords.end()) {
        EV_ERROR << "[DQN] Cannot penalize dropping: packet record not found" << '\n';
        return;
    }

    EV_WARN << "[DQN] Drop penalty: " << -dropPenalty
            << " for destination " << it->second.destination << '\n';

    State currentState = getCurrentState(it->second.destination);
    sendRewardToServer(-dropPenalty, currentState);

    /* EWMA failure update */
    EV_WARN << "[DQN] Updating EWMA for failed transmission via " << it->second.lastHop << '\n';
    updateEwma(linkStats[it->second.lastHop], false);

    EV_WARN << "[DQN] Removing packet record after drop" << '\n';
    packetRecords.erase(it);
}
/* ------------------------------------------------------------------ */
/*  INetfilter hooks                                                  */
/* ------------------------------------------------------------------ */
INetfilter::IHook::Result
QLearningAODV::datagramPreRoutingHook(Packet *p)
{
    EV_WARN << "[DQN] Pre-routing hook for packet " << p->getId() << '\n';
    const auto& ipHdr = getNetworkProtocolHeader(p);
    if (!ipHdr || ipHdr->getSourceAddress().isUnspecified()) {
        EV_WARN << "[DQN] No IP header or unspecified source, delegating to Aodv" << '\n';
        return Aodv::datagramPreRoutingHook(p);
    }

    const L3Address& dest = ipHdr->getDestinationAddress();
    if (dest.isBroadcast() || dest.isMulticast()) {
        EV_WARN << "[DQN] Broadcast/multicast packet, delegating to Aodv" << '\n';
        return Aodv::datagramPreRoutingHook(p);
    }

    if (routingTable->isLocalAddress(dest)) {
        EV_WARN << "[DQN] Packet already at destination " << dest << ", accepting" << '\n';
        return ACCEPT;   // already at destination
    }

    if (useQlearning && socketConnected) {
        EV_WARN << "[DQN] Using Q-learning to select next hop for " << dest << '\n';
        L3Address nhop = selectNextHop(dest);
        if (!nhop.isUnspecified()) {
            EV_WARN << "[DQN] Selected next hop: " << nhop << ", accepting packet" << '\n';
            p->addTagIfAbsent<NextHopAddressReq>()->setNextHopAddress(nhop);
            recordPacket(p, nhop);
            return ACCEPT;
        } else {
            EV_WARN << "[DQN] Q-learning failed to select next hop, delegating to Aodv" << '\n';
        }
    } else {
        EV_WARN << "[DQN] Q-learning disabled or disconnected, delegating to Aodv" << '\n';
    }

    return Aodv::datagramPreRoutingHook(p);
}
INetfilter::IHook::Result
QLearningAODV::datagramForwardHook(Packet *p)
{
    EV_WARN << "[DQN] Forward hook for packet " << p->getId() << '\n';
    if (useQlearning && socketConnected) {
        EV_WARN << "[DQN] Processing forward reward" << '\n';
        rewardForwarding(p);
    } else {
        EV_WARN << "[DQN] Q-learning disabled or disconnected, skipping reward" << '\n';
    }
    return Aodv::datagramForwardHook(p);
}
INetfilter::IHook::Result
QLearningAODV::datagramLocalInHook(Packet *p)
{
    EV_WARN << "[DQN] Local-in hook for packet " << p->getId() << '\n';
    const auto& nh = getNetworkProtocolHeader(p);
    if (nh && routingTable->isLocalAddress(nh->getDestinationAddress()) &&
        useQlearning && socketConnected) {
        EV_WARN << "[DQN] Packet delivered to local destination, processing delivery reward" << '\n';
        rewardDelivery(p);
    } else {
        EV_WARN << "[DQN] Skipping delivery reward: conditions not met" << '\n';
    }
    return Aodv::datagramLocalInHook(p);
}
INetfilter::IHook::Result
QLearningAODV::datagramLocalOutHook(Packet *p)
{
    EV_WARN << "[DQN] Local-out hook for packet " << p->getId() << '\n';
    const auto& nh = getNetworkProtocolHeader(p);
    if (nh && useQlearning && socketConnected &&
        !routingTable->isLocalAddress(nh->getDestinationAddress())) {
        EV_WARN << "[DQN] Outgoing packet to non-local destination, selecting next hop" << '\n';
        L3Address nhop = selectNextHop(nh->getDestinationAddress());
        if (!nhop.isUnspecified()) {
            EV_WARN << "[DQN] Selected next hop: " << nhop << " for outgoing packet" << '\n';
            recordPacket(p, nhop);
            p->addTagIfAbsent<NextHopAddressReq>()->setNextHopAddress(nhop);
        } else {
            EV_WARN << "[DQN] Failed to select next hop for outgoing packet" << '\n';
        }
    } else {
        EV_WARN << "[DQN] Not applying Q-learning for this packet" << '\n';
    }
    return Aodv::datagramLocalOutHook(p);
}
INetfilter::IHook::Result
QLearningAODV::datagramPostRoutingHook(Packet *p)
{
    EV_WARN << "[DQN] Post-routing hook for packet " << p->getId() << '\n';
    /* We don't change anything at PostRouting; just delegate. */
    return Aodv::datagramPostRoutingHook(p);
}
/* ------------------------------------------------------------------ */
/*  MAC-level signal listener (EWMA update)                            */
/* ------------------------------------------------------------------ */
void QLearningAODV::handleSignal(cComponent *source, simsignal_t signalID,
                                cObject *obj, cObject *details)
{
    if (!useQlearning || !socketConnected) {
        return;
    }

    // Process packet drop signals
    if (signalID == packetDroppedSignal) {
        Packet *packet = dynamic_cast<Packet *>(obj);
        if (!packet) {
            EV_ERROR << "[DQN] Error: Dropped object is not a packet" << '\n';
            return;
        }

        // Check if this is a data packet that we care about (not a control packet)
        const auto& ipv4Header = packet->peekAtFront<Ipv4Header>();
        if (!ipv4Header) {
            return; // Not an IPv4 packet
        }

        EV_WARN << "[DQN] Detected packet drop for " << ipv4Header->getDestinationAddress()
                << ", applying penalty" << '\n';

        // Apply the drop penalty
        penalizeDropping(packet);
    }
}
/* ------------------------------------------------------------------ */
void QLearningAODV::handleMessageWhenUp(cMessage *msg)
{
    EV_WARN << "[DQN] Handling message when up: " << msg->getName() << '\n';
    Aodv::handleMessageWhenUp(msg);
}
} // namespace aodv
} // namespace inet
