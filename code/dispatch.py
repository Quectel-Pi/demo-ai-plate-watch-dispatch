#!/usr/bin/env python3
"""网约车接客调度核心（纯逻辑，无 UI 依赖，宿主机可直接单测）。

消费已确认车牌事件，驱动四车位接客调度。所有方法显式传入时间 t（秒），
返回事件 dict 列表，UI/日志层据此播报与落盘。

订单状态机: waiting(等车位) → assigned(已分配) → serving(接客中) → done(完成)
            活跃态任意时刻可 cancel(取消)。
车位状态机: free → assigned → serving → free

调度规则（对应设计方案第三节）:
  - 登记即分配：有空位直接分配，无空位进 FIFO 等待队列，释放自动补位
  - 重复识别：assigned 态再次识别保持原车位，>REANNOUNCE_S 才重播提醒
  - 已到位（serving）识别静默：车已到位，重播"请前往"语义错乱
  - 等待中识别 60s 节流：等待是常态，仅刷横幅/日志，语音层不播
  - 未登记车辆：识别到未登记车牌报告一次（冷却 60s），支持补登记
  - 超时：已分配超 TIMEOUT_S 未到位 → 标记超时（车位保留，可取消/到位）
语音原则：只播"需要司机立即行动"的时刻（分配/到位/完成/取消/超时），
未登记与等待中仅视觉提示——由 UI 层 VOICE_BY_TYPE 决定，不在此模块。
"""
import os
import time

ORDER_WAITING = 'waiting'
ORDER_ASSIGNED = 'assigned'
ORDER_SERVING = 'serving'
ORDER_DONE = 'done'
ORDER_CANCELLED = 'cancelled'

SPOT_FREE = 'free'
SPOT_ASSIGNED = 'assigned'
SPOT_SERVING = 'serving'

TIMEOUT_S = float(os.environ.get('DISPATCH_TIMEOUT_S', 120))
REANNOUNCE_S = 15.0        # 重复识别最小重播间隔
WAITING_REMIND_S = 60.0    # 等待中车辆识别提示节流（横幅刷新用，语音层不播）
UNREG_COOLDOWN_S = 60.0    # 未登记车辆重复提示冷却


class Order:
    __slots__ = ('oid', 'plate', 'state', 'registered_t', 'assigned_t',
                 'arrived_t', 'done_t', 'spot', 'timed_out', 'last_announce_t')

    def __init__(self, oid, plate, t):
        self.oid = oid
        self.plate = plate
        self.state = ORDER_WAITING
        self.registered_t = t
        self.assigned_t = 0.0
        self.arrived_t = 0.0
        self.done_t = 0.0
        self.spot = None
        self.timed_out = False
        self.last_announce_t = 0.0

    @property
    def active(self):
        return self.state in (ORDER_WAITING, ORDER_ASSIGNED, ORDER_SERVING)


class DispatchCore:
    def __init__(self, spots=('A01', 'A02', 'A03', 'A04'), timeout_s=TIMEOUT_S):
        self.spot_names = list(spots)
        self.spot_state = {s: SPOT_FREE for s in self.spot_names}
        self.timeout_s = timeout_s
        self.orders = {}            # oid -> Order
        self.active_by_plate = {}   # plate -> Order（活跃订单唯一）
        self._next_oid = 1
        self.unreg_seen = {}        # plate -> 上次提示时间
        self.completed_count = 0

    # ---------- 对外查询 ----------
    def stats(self):
        waiting = sum(1 for o in self.orders.values() if o.state == ORDER_WAITING)
        serving = sum(1 for o in self.orders.values() if o.state == ORDER_SERVING)
        used = sum(1 for s in self.spot_state.values() if s != SPOT_FREE)
        return {'done': self.completed_count, 'waiting': waiting,
                'serving': serving, 'spots_used': used,
                'spots_total': len(self.spot_names)}

    def active_orders(self):
        """活跃订单列表，等待队列按登记时间排序在前"""
        ws = sorted((o for o in self.orders.values() if o.state == ORDER_WAITING),
                    key=lambda o: o.registered_t)
        others = [o for o in self.orders.values() if o.state in (ORDER_ASSIGNED, ORDER_SERVING)]
        return ws + others

    # ---------- 登记 ----------
    def register(self, plate, t):
        """登记乘客用车。已有同牌活跃订单则拒绝；返回事件列表。"""
        plate = plate.strip().upper()
        if not plate:
            return [self._ev('register_rejected', plate, reason='empty')]
        old = self.active_by_plate.get(plate)
        if old is not None:
            return [self._ev('register_rejected', plate, reason='duplicate',
                             oid=old.oid)]
        o = Order(self._next_oid, plate, t)
        self._next_oid += 1
        self.orders[o.oid] = o
        self.active_by_plate[plate] = o
        self.unreg_seen.pop(plate, None)
        evs = [self._ev('registered', plate, oid=o.oid)]
        evs += self._try_assign(o, t)
        return evs

    def register_many(self, plates, t):
        evs = []
        for p in plates:
            evs += self.register(p, t)
        return evs

    # ---------- 识别事件入口 ----------
    def on_plate_seen(self, plate, t):
        """已确认车牌出现在画面。返回事件列表（分配/重播/未登记）。"""
        plate = plate.strip().upper()
        o = self.active_by_plate.get(plate)
        if o is None:
            last = self.unreg_seen.get(plate, -1e9)
            if t - last < UNREG_COOLDOWN_S:
                return []
            self.unreg_seen[plate] = t
            return [self._ev('unregistered', plate)]
        if o.state == ORDER_WAITING:
            # 等待是常态：仅 60s 节流刷一次横幅/日志，语音层不播报
            if o.last_announce_t and t - o.last_announce_t < WAITING_REMIND_S:
                return []
            o.last_announce_t = t
            return [self._ev('still_waiting', plate, oid=o.oid)]
        if o.state == ORDER_SERVING:
            # 已到位接客中：识别只确认车在，重播"请前往"语义错乱，静默
            return []
        # assigned：保持原车位，节流重播提醒司机到位
        if t - o.last_announce_t >= REANNOUNCE_S:
            o.last_announce_t = t
            return [self._ev('reannounce', plate, oid=o.oid, spot=o.spot,
                            state=o.state)]
        return []

    # ---------- 现场按钮动作 ----------
    def arrive(self, oid, t):
        """车辆到位：assigned → serving"""
        o = self.orders.get(oid)
        if o is None or o.state != ORDER_ASSIGNED:
            return []
        o.state = ORDER_SERVING
        o.arrived_t = t
        o.timed_out = False
        self.spot_state[o.spot] = SPOT_SERVING
        return [self._ev('arrived', o.plate, oid=o.oid, spot=o.spot)]

    def complete(self, oid, t):
        """接客完成：释放车位 → FIFO 自动补位"""
        o = self.orders.get(oid)
        if o is None or o.state != ORDER_SERVING:
            return []
        o.state = ORDER_DONE
        o.done_t = t
        self.completed_count += 1
        spot = o.spot
        self._release_spot(o, t)
        evs = [self._ev('completed', o.plate, oid=o.oid, spot=spot)]
        evs += self._refill(spot, t)
        return evs

    def cancel(self, oid, t):
        """取消订单：释放车位（如有）→ FIFO 自动补位"""
        o = self.orders.get(oid)
        if o is None or not o.active:
            return []
        o.state = ORDER_CANCELLED
        o.done_t = t
        spot = o.spot
        evs = [self._ev('cancelled', o.plate, oid=o.oid, spot=spot)]
        self._release_spot(o, t)
        if spot:
            evs += self._refill(spot, t)
        return evs

    # ---------- 超时巡检 ----------
    def tick(self, t):
        """已分配超时未到位 → 超时事件（车位保留，等到位或取消）"""
        evs = []
        for o in self.orders.values():
            if o.state == ORDER_ASSIGNED and not o.timed_out:
                if t - o.assigned_t >= self.timeout_s:
                    o.timed_out = True
                    evs.append(self._ev('timeout', o.plate, oid=o.oid, spot=o.spot))
        return evs

    # ---------- 内部 ----------
    def _try_assign(self, o, t):
        for s in self.spot_names:
            if self.spot_state[s] == SPOT_FREE:
                return self._assign(o, s, t)
        return []

    def _assign(self, o, spot, t):
        o.state = ORDER_ASSIGNED
        o.spot = spot
        o.assigned_t = t
        o.timed_out = False
        o.last_announce_t = t
        self.spot_state[spot] = SPOT_ASSIGNED
        return [self._ev('assigned', o.plate, oid=o.oid, spot=spot)]

    def _refill(self, spot, t):
        """释放后的车位按 FIFO 从等待队列补位"""
        waiting = sorted((o for o in self.orders.values()
                          if o.state == ORDER_WAITING),
                         key=lambda o: o.registered_t)
        if waiting:
            return self._assign(waiting[0], spot, t)
        return []

    def _release_spot(self, o, t):
        if o.spot and self.spot_state.get(o.spot) != SPOT_FREE:
            self.spot_state[o.spot] = SPOT_FREE
        self.active_by_plate.pop(o.plate, None)
        o.spot = None

    def _ev(self, etype, plate, **kw):
        e = {'type': etype, 'plate': plate, 't': time.strftime('%H:%M:%S')}
        e.update(kw)
        return e


# ---- 预置演示车辆（对应 materials/videos 真实车牌，播视频即可真实触发）----
PRESET_VEHICLES = [
    '苏A53D9P', '苏AQ2L95', '苏ABD220', '苏A0AR19', '苏A3XF72',  # lpr_test
    '皖ADJ6382',   # rk_2s
    '皖AA91218',   # rk_5s
    '皖AA88515',   # rk_9s
    'C98191P',     # hw_day（海外样式）
]

# ---- 预演登记计划表：(车牌, 启动后延迟秒)，_demo_tick 按表错峰登记 ----
# 错峰目的：语音播报串行队列，一次性登记 9 辆会让「请前往」积压到音画不同步；
# 分批登记后每条播报紧贴画面事件。苏A53D9P@25 为重复登记，演示拒绝分支；
# 皖ADJ6382@26 时四车位已满，演示 FIFO 候队与后续补位。
DEMO_SCHEDULE = [
    ('苏A53D9P', 8.0),    # → A01（后续二次接客主角）
    ('苏AQ2L95', 9.0),    # → A02
    ('苏ABD220', 24.0),   # → A03
    ('苏A0AR19', 24.0),   # → A04（超时演示，故意不到位）
    ('苏A53D9P', 25.0),   # 重复登记 → 拒绝（横幅提示）
    ('皖ADJ6382', 26.0),  # 四车位满 → 候队，30s 补位 A01
]
