"""Stand-in for gamst_graph_runner (C++) speaking protocol v1, for tests without a MediaPipe build.

It follows runner/gamst_graph_runner.cc command for command: the same stdin commands, the same
stdout lines, one worker thread that processes frames in timestamp order (MediaPipe's scheduler
stands in), and a FlowLimiter with max_in_flight (side packet) and max_in_queue=1.

Landmarks are synthetic but deterministic, decoded from the frame's first pixels so a test can
script a person:

- pixel (0, 0) channel 0 (B): left elbow angle in degrees (0..180); 255 = no person in view
- pixel (0, 0) channel 1 (G): > 127 raises the right wrist above the shoulder
- pixel (0, 1) channel 1 (G): > 127 adds a visible right hand

Environment: FAKE_DELAY_MS (per-frame work), FAKE_FAIL_AFTER (fatal error after N frames),
FAKE_DROP_EVERY (FlowLimiter drops every Nth frame), FAKE_REVISION, FAKE_LOG (append every command line received, for assertions).
"""
import math
import os
import queue
import re
import sys
import threading
import time

PROTOCOL = 1


def emit(lock, out, text):
    with lock:
        out.write(text + '\n')
        out.flush()


def num(value):
    return 'null' if value is None else repr(round(float(value), 6))


def rows(points):
    return '[' + ','.join('[' + ','.join(num(v) for v in p) + ']' for p in points) + ']'


def body(angle_deg, right_up):
    points = [[.5, .5, 0., 0., 1.] for _ in range(33)]

    def put(index, x, y):
        points[index] = [x, y, -.1, .99, .99]
    put(0, .5, .3)                              # nose
    for index in range(1, 11):                  # face points from pose
        put(index, .5 + .01 * (index - 5), .28)
    put(11, .45, .4); put(12, .55, .4)          # shoulders
    put(13, .40, .5); put(14, .60, .5)          # elbows
    angle = math.radians(angle_deg)
    put(15, .40 + .1 * math.sin(angle), .5 - .1 * math.cos(angle))
    put(16, .60, .3 if right_up else .6)
    put(23, .46, .7); put(24, .54, .7); put(25, .46, .85); put(26, .54, .85); put(27, .46, .98); put(28, .54, .98)
    return points


def hand(cx, cy):
    return [[cx + .005 * (i % 5), cy + .005 * (i // 5), 0., None, None] for i in range(21)]


def face(count):
    return [[.45 + .1 * (i % 26) / 26, .2 + .1 * (i // 26) / 19, 0., None, None] for i in range(count)]


class Fake:
    def __init__(self, graph, out):
        text = open(graph).read()
        top = [line for line in text.splitlines() if line and not line[0].isspace()]
        inputs = [m.group(1) for line in top for m in [re.match(r'input_stream:\s*"([^"]+)"', line)] if m]
        self.sides_declared = [m.group(1) for line in top for m in [re.match(r'input_side_packet:\s*"([^"]+)"', line)] if m]
        if 'input_video' not in inputs:
            raise SystemExit(2)
        self.branches = {name[len('enable_'):]: True for name in inputs if name.startswith('enable_')}
        self.sides, self.run, self.last_us = {}, 0, None
        self.out, self.lock = out, threading.Lock()
        self.delay = float(os.environ.get('FAKE_DELAY_MS', '0')) / 1000
        self.fail_after = int(os.environ.get('FAKE_FAIL_AFTER', '0'))
        self.drop_every = int(os.environ.get('FAKE_DROP_EVERY', '0'))
        self.revision = os.environ.get('FAKE_REVISION', 'fake')
        self.frames = 0
        self.queue = queue.Queue()
        self.cond = threading.Condition()
        self.waiting = []   # FlowLimiter queue (max_in_queue=1), timestamp order kept
        self.in_flight = 0
        self.worker = None

    def line(self, text):
        emit(self.lock, self.out, text)

    def error(self, stage, code, fatal):
        self.line('{"type":"error","stage":"%s","code":"%s","fatal":%s}' % (stage, code, 'true' if fatal else 'false'))

    def hello(self):
        branches = ','.join('"%s"' % b for b in self.branches)
        sides = ','.join('"%s"' % s for s in self.sides_declared)
        self.line('{"type":"hello","protocol":%d,"mediapipe":"%s","gpu":false,"input":"input_video",'
                  '"frame_stream":"gamst_frame","branches":[%s],"side_packets":[%s]}' % (PROTOCOL, self.revision, branches, sides))

    # -------------------------------------------------- graph run (worker = MediaPipe scheduler)

    def start(self):
        self.run += 1
        self.in_flight = 0
        self.last_us = None
        self.max_in_flight = int(self.sides.get('max_in_flight', 1))
        self.worker = threading.Thread(target=self.work, args=(self.run,), daemon=True)
        self.worker.start()
        states = ','.join('"%s":%s' % (k, 'true' if v else 'false') for k, v in self.branches.items())
        self.line('{"type":"ready","run":%d,"branches":{%s}}' % (self.run, states))

    def stop(self):
        if self.worker is None:
            return
        with self.cond:  # closing the inputs releases what the FlowLimiter still holds
            rest, self.waiting = self.waiting, []
            for us, pixels, branches in rest:
                self.queue.put((us, pixels, branches, True))
        self.queue.put(None)
        self.worker.join()
        self.worker = None

    def frame(self, us, pixels, branches):
        # FlowLimiter: admit while fewer than max_in_flight are inside, else keep the latest waiting.
        with self.cond:
            if self.drop_every and self.frames % self.drop_every == 0:
                self.queue.put((us, pixels, branches, False))
            elif self.in_flight < self.max_in_flight:
                self.in_flight += 1
                self.queue.put((us, pixels, branches, True))
            else:
                for dropped in self.waiting:
                    self.queue.put((*dropped, False))
                self.waiting = [(us, pixels, branches)]

    def work(self, run):
        while True:
            item = self.queue.get()
            if item is None:
                return
            if item[3]:
                time.sleep(self.delay)
            self.bundle(run, *item)
            if item[3]:
                with self.cond:
                    self.in_flight -= 1
                    if self.waiting:
                        self.in_flight += 1
                        self.queue.put((*self.waiting.pop(), True))

    def bundle(self, run, us, pixels, branches, processed):
        head = '{"type":"packet","run":%d,"ts_us":%d' % (run, us)
        if processed and branches.get('pose', True):
            b, g = pixels[0], pixels[1]
            if b != 255:
                points = body(min(180, b), g > 127)
                self.line(head + ',"stream":"pose_landmarks","kind":"landmarks","value":%s}' % rows(points))
                world = [[(p[0] - .5), (p[1] - .5), p[2], p[3], p[4]] for p in points]
                self.line(head + ',"stream":"pose_world_landmarks","kind":"world_landmarks","value":%s}' % rows(world))
                if branches.get('hands', False):
                    self.line(head + ',"stream":"left_hand_landmarks","kind":"landmarks","value":%s}'
                              % rows(hand(points[15][0], points[15][1])))
                    if pixels[4] > 127:
                        self.line(head + ',"stream":"right_hand_landmarks","kind":"landmarks","value":%s}'
                                  % rows(hand(points[16][0], points[16][1])))
                if branches.get('face', False):
                    count = 478 if str(self.sides.get('refine_face_landmarks', '0')) in ('1', 'true') else 468
                    self.line(head + ',"stream":"face_landmarks","kind":"landmarks","value":%s}' % rows(face(count)))
        self.line('{"type":"done","run":%d,"ts_us":%d,"processed":%s}' % (run, us, 'true' if processed else 'false'))


def main():
    graph = next((arg.split('=', 1)[1] for arg in sys.argv[1:] if arg.startswith('--graph=')), '')
    out = os.fdopen(os.dup(sys.stdout.fileno()), 'w')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    log = os.environ.get('FAKE_LOG')
    try:
        fake = Fake(graph, out)
    except (OSError, SystemExit):
        out.write('{"type":"error","stage":"load","code":"INVALID_ARGUMENT","fatal":true}\n')
        out.flush()
        return 2
    fake.hello()
    source = sys.stdin.buffer
    while True:
        raw = source.readline()
        if not raw:
            break
        line = raw.decode().rstrip('\n')
        if log:
            with open(log, 'a') as handle:
                handle.write(line + '\n')
        if not line:
            continue
        parts = line.split(' ', 3)
        op = parts[0]
        if op == 'FRAME' and len(parts) == 4:
            us, width, height = (int(v) for v in parts[1:])
            data = source.read(width * height * 3)
            if len(data) != width * height * 3:
                break
            if fake.worker is None:
                fake.error('frame', 'FAILED_PRECONDITION', False)
                continue
            if fake.last_us is not None and us <= fake.last_us:
                fake.error('frame', 'INVALID_ARGUMENT', False)
                continue
            fake.last_us = us
            fake.frames += 1
            if fake.fail_after and fake.frames > fake.fail_after:
                fake.error('frame', 'INTERNAL', True)
                return 1
            fake.frame(us, data[:6], dict(fake.branches))
        elif op == 'SIDE' and len(parts) == 4:
            name, kind, value = parts[1:]
            ok = (kind == 'bool' and value in ('0', '1', 'true', 'false')) or kind == 'string' or \
                (kind == 'int' and re.fullmatch(r'-?\d+', value)) or \
                (kind in ('float', 'double') and re.fullmatch(r'-?[\d.]+(e-?\d+)?', value))
            if not ok:
                fake.error('side', 'INVALID_ARGUMENT', False)
                continue
            fake.sides[name] = value
            fake.line('{"type":"ack","op":"side","name":"%s","applies":"%s"}'
                      % (name, 'next_run' if fake.worker else 'this_run'))
        elif op == 'BRANCH' and len(parts) == 3:
            if parts[1] not in fake.branches or parts[2] not in ('0', '1'):
                fake.error('branch', 'NOT_FOUND', False)
                continue
            fake.branches[parts[1]] = parts[2] == '1'
            fake.line('{"type":"ack","op":"branch","name":"%s","enabled":%s}'
                      % (parts[1], 'true' if parts[2] == '1' else 'false'))
        elif op == 'START' and len(parts) == 1:
            if fake.worker is not None:
                fake.error('start', 'FAILED_PRECONDITION', False)
                continue
            fake.start()
        elif op == 'RESET' and len(parts) == 1:
            fake.stop()
            fake.start()
        elif op == 'CLOSE' and len(parts) == 1:
            fake.stop()
            fake.line('{"type":"closed","run":%d}' % fake.run)
            return 0
        else:
            fake.error('command', 'INVALID_ARGUMENT', True)
            return 4
    fake.stop()
    fake.line('{"type":"closed","run":%d}' % fake.run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
