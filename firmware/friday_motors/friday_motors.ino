/* Friday's motor firmware -- Arduino Uno + L298N, 4 motors, differential.
 *
 * WIRING (exactly as Harsh built it, 2026-08-23)
 *
 *   Left side, 2 motors  -> Channel A     Right side, 2 motors -> Channel B
 *     both reds   -> OUT1                   both reds   -> OUT3
 *     both blacks -> OUT2                   both blacks -> OUT4
 *
 *   L298N   Arduino   purpose
 *   ENA     D9  PWM   left speed
 *   IN1     D8        left direction
 *   IN2     D7        left direction
 *   IN3     D6        right direction
 *   IN4     D5        right direction
 *   ENB     D3  PWM   right speed
 *
 * Also connect L298N GND to Arduino GND, or the direction pins float and the
 * motors twitch on their own.
 *
 * PROTOCOL -- newline-terminated ASCII at 115200 baud
 *
 *   P              identify        -> "FRIDAY-MOTORS 1"
 *   M <l> <r>      set speeds      -> "OK"     l and r are -255..255
 *   S              stop            -> "OK"
 *   W <ms>         watchdog window -> "OK"     0 disables (not advised)
 *   G              gyro            -> "G <yaw cdeg> <rate cdeg/s> <ok|cal|nogyro> <gaps> <errors>
 *                                       ... *XX" (GYRO 6+: XOR checksum of the line before '*')
 *                                       <tilt ddeg> <shake ddeg/s> <hpeak mg> <vpeak mg> <tilted 0|1> <bumps>"
 *                  (the peaks are since the last G; the rest are now)
 *   Z              zero the yaw    -> "OK"
 *   B              re-measure bias -> "OK" (only while stopped, else "ERR")
 *   V              version         -> "FRIDAY-MOTORS 1 GYRO 1 who=0x.."
 *   I              I2C scan        -> "I A0=b/p A1=b/p A4=b/p A5=b/p A0A1:<addrs> A1A0:.. A4A5:.. A5A4:.."
 *                  (only while stopped, else "ERR"; ~60 ms; see scanReport)
 *   anything else                  -> "ERR"
 *
 * TILTED (more than TILT_ON_DEG for TILT_ON_MS -- picked up, tipped, climbing
 * something): the motors stop HERE, whatever the Pi is running, and every M
 * answers "TILT" and is ignored until she has been under TILT_OFF_DEG for
 * TILT_OFF_MS. S still answers "OK".
 *
 * G, Z, B and V do NOT pet the watchdog: something polling the gyro must never
 * keep the motors alive for a driver that has stopped talking.
 *
 * THE WATCHDOG IS THE POINT. If no command arrives within the window, the
 * motors stop on their own. So if the Pi crashes, the USB cable is pulled,
 * or the program is killed mid-move, the robot stops instead of driving into
 * a wall. Whoever is driving must keep sending -- silence means stop.
 */

// PWM FREQUENCY (Harsh, 2026-09-28: "the motors are like making too noise").
// analogWrite's default on D9 (Timer1) and D3 (Timer2) is 490 Hz -- right in
// the band where the windings hum loudest. Prescaler 1 on both timers gives
// 16 MHz / 510 = 31.37 kHz phase-correct PWM: above hearing. Timer0 (millis,
// and so the watchdog) is untouched. The L298N switches slower than this
// likes, so it runs a little warmer and the same duty drives a little
// differently: the stiction floors (straight 80, pivot 190, measured at 490 Hz)
// must be re-measured after flashing. Set to 0 and reflash to go back.
//
// ⚠ TRIED ON THE ROBOT 2026-09-28 23:50 AND REVERTED. At 31 kHz Harsh watched
// eight labelled moves: straight at 80 AND 110 did not move at all, a pivot at
// 216 moved one way only, 229 turned. The L298N cannot switch that fast, so the
// same duty delivers far less -- and needing ~230 to turn means MORE gear noise,
// not less, which was the point ("gears make noise"). 490 Hz stays.
#define PWM_31KHZ 0
//
// ⚠ AND IT STAYED FLASHED FOR FOUR DAYS (found 2026-10-02). The "revert" at
// 23:55 that night set this to 0 and ran `arduino-cli upload` WITHOUT
// `compile`: upload re-flashes the LAST BUILD, which was the 31 kHz one (4732
// bytes; the 490 Hz build is 4708). Every run since was weak in every
// direction -- slow, bursty, reverses that would not move her -- while this
// file said 490 Hz. Re-flashed 490 Hz 2026-10-02 22:24. ALWAYS compile and
// upload together, from an explicit build:
//   ./tools/arduino-cli compile --fqbn arduino:avr:uno --output-dir B firmware/friday_motors
//   ./tools/arduino-cli upload  --fqbn arduino:avr:uno -p <by-id port> --input-dir B
// (The bootloader on this clone cannot read the flash back, so what is on the
// board can only be known from what was last built and uploaded.)

const uint8_t ENA = 9, IN1 = 8, IN2 = 7;     // left
const uint8_t IN3 = 6, IN4 = 5, ENB = 3;     // right

unsigned long lastCommandMs = 0;
unsigned long watchdogMs = 500;
bool moving = false;

// ---- GYRO (Harsh, 2026-10-07): an MPU-6050 (GY-521) on A0 = SDA, A1 = SCL ----
// A0/A1 are not the Uno's hardware I2C pins (those are A4/A5), so the bus is
// driven here by hand, OPEN-DRAIN: a line is pulled LOW by making its pin an
// output (its PORT bit is kept 0) and RELEASED by making it an input, and the
// GY-521's own pull-ups (to its 3.3 V) take it high. Never driven high, never
// the internal pull-ups: no 5 V into a 3.3 V part.
// Every bus operation is a FIXED number of steps -- it cannot hang. That is
// the point: the AVR Wire library can block forever on a bus glitch, and a
// frozen loop() would freeze the watchdog below with the motors running.
// No hardware watchdog (wdt): this clone's bootloader is non-standard (it
// cannot read flash back), and on such bootloaders a WDT reset can loop
// forever. With nothing left that can hang, it is not needed.
const uint8_t SDA_BIT = 0, SCL_BIT = 1;      // PORTC: A0 = PC0, A1 = PC1
const uint8_t MPU = 0x68;                    // GY-521 with AD0 unconnected
const float LSB_PER_DPS = 65.5f;             // GYRO_CONFIG FS_SEL=1: +-500 deg/s
const float LSB_PER_G = 16384.0f;            // ACCEL_CONFIG AFS_SEL=0: +-2 g
const unsigned long GYRO_PERIOD_US = 10000;  // 100 Hz: all six axes (14 bytes) per read
const unsigned long GYRO_MAX_DT_US = 50000;  // a longer loop stall is a "gap"
const uint16_t CAL_SAMPLES = 100;            // 1 s of stillness for the bias
// Tilt from the accelerometer, against the chip's own Z axis: the board is
// mounted FLAT (2026-10-07, underneath, chip facing the floor), so on a level
// floor gravity is along Z whichever way up. Not against a reference taken at
// boot -- she may be booted while being carried.
const float TILT_ON_DEG = 35.0f, TILT_OFF_DEG = 20.0f;
const unsigned long TILT_ON_MS = 250, TILT_OFF_MS = 500;
const float BUMP_G = 0.6f;                   // a horizontal jolt this big counts as a bump

bool gyroOk = false;
uint8_t gyroWho = 0;
uint8_t whoTries = 0;                        // the last init's WHO_AM_I attempts (4 = all failed)
float biasRaw = 0.0f;
long calSum = 0;
uint16_t calCount = 0;
float yawDeg = 0.0f;                         // integrated, + = counter-clockwise (chip up)
float rateDps = 0.0f;
unsigned long lastGyroUs = 0, lastInitMs = 0, stillSinceMs = 0;
unsigned long gyroGaps = 0, gyroErrors = 0;
uint8_t gyroFailRun = 0;
long calSumX = 0, calSumY = 0;
float biasX = 0.0f, biasY = 0.0f;            // roll/pitch-rate bias, for "shake"
float afx = 0.0f, afy = 0.0f, afz = 1.0f;    // filtered acceleration, g
bool accelSeen = false;
float tiltDeg = 0.0f, shakeDps = 0.0f;
float hPeakG = 0.0f, vPeakG = 0.0f;          // since the last G
unsigned long bumps = 0;
bool tilted = false;
// STILLNESS, for the bias. 2026-10-07: a bias measured while Harsh held her
// (B sent mid-handling) was 2-3 deg/s out, and the old follower (|rate| < 1)
// could never pull an error that size back. Stillness is judged by the
// accelerometer and the raw spread, never by the rate the bias itself makes.
int16_t calMin = 32767, calMax = -32768;
const int16_t CAL_SPREAD = 131;              // 2 deg/s of raw Z spread = not still
const float STILL_JOLT_G = 0.12f;
float lastJoltG = 0.0f;                      // this sample's jolt, any direction
unsigned long tiltSinceMs = 0, levelSinceMs = 0;

uint8_t sdaBit = SDA_BIT, sclBit = SCL_BIT;  // only the scan (I) ever moves these
static inline void sdaLow()  { DDRC |=  _BV(sdaBit); }
static inline void sdaHigh() { DDRC &= ~_BV(sdaBit); }
static inline void sclLow()  { DDRC |=  _BV(sclBit); }
static inline void sclHigh() { DDRC &= ~_BV(sclBit); }
static inline bool sdaRead() { return PINC & _BV(sdaBit); }
static inline void i2cDelay() { delayMicroseconds(5); }   // ~100 kHz

void i2cStart() {
  sdaHigh(); i2cDelay(); sclHigh(); i2cDelay();
  sdaLow(); i2cDelay(); sclLow(); i2cDelay();
}

void i2cStop() {
  sdaLow(); i2cDelay(); sclHigh(); i2cDelay(); sdaHigh(); i2cDelay();
}

void i2cRecover() {
  // A slave left mid-byte holds SDA low: up to nine clocks let it finish.
  sdaHigh();
  for (uint8_t i = 0; i < 9 && !sdaRead(); i++) {
    sclLow(); i2cDelay(); sclHigh(); i2cDelay();
  }
  i2cStop();
}

bool i2cWrite(uint8_t b) {                   // true = the slave ACKed
  for (uint8_t i = 0; i < 8; i++) {
    if (b & 0x80) sdaHigh(); else sdaLow();
    b <<= 1;
    i2cDelay(); sclHigh(); i2cDelay(); sclLow();
  }
  sdaHigh(); i2cDelay(); sclHigh(); i2cDelay();
  bool ack = !sdaRead();
  sclLow(); i2cDelay();
  return ack;
}

uint8_t i2cRead(bool ack) {
  uint8_t b = 0;
  sdaHigh();
  for (uint8_t i = 0; i < 8; i++) {
    i2cDelay(); sclHigh(); i2cDelay();
    b = (b << 1) | (sdaRead() ? 1 : 0);
    sclLow();
  }
  if (ack) sdaLow(); else sdaHigh();
  i2cDelay(); sclHigh(); i2cDelay(); sclLow(); sdaHigh(); i2cDelay();
  return b;
}

bool mpuWrite(uint8_t reg, uint8_t val) {
  i2cStart();
  bool ok = i2cWrite(MPU << 1) && i2cWrite(reg) && i2cWrite(val);
  i2cStop();
  return ok;
}

bool regRead(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t n) {
  i2cStart();
  if (!i2cWrite(addr << 1) || !i2cWrite(reg)) { i2cStop(); return false; }
  i2cStart();                                // repeated start
  if (!i2cWrite((addr << 1) | 1)) { i2cStop(); return false; }
  for (uint8_t i = 0; i < n; i++) buf[i] = i2cRead(i + 1 < n);
  i2cStop();
  return true;
}

bool mpuRead(uint8_t reg, uint8_t *buf, uint8_t n) {
  return regRead(MPU, reg, buf, n);
}

// I: WHERE IS THE CHIP? 2026-10-19: who=0x0 after two rewires and a swapped-
// pin build, and nothing could tell no power from the wrong pins from a dead
// chip. For A0, A1, A4, A5: "b/p" = the level released with no pull-up of
// ours (1 = the module's own pull-ups are powered), then with our pull-up on
// (0 = something holds the line low: a short, or an unpowered chip clamping
// it). Then every address 0x08-0x77 on four {SDA, SCL} pin pairs; an address
// that answers is followed by its register 0x75 (WHO_AM_I) as "(70)". Twelve
// or more answering = SDA stuck low ("stuck"). The pins go back as they were.
void scanReport() {
  const uint8_t pins[4] = {0, 1, 4, 5};
  Serial.print("I");
  for (uint8_t i = 0; i < 4; i++) {
    uint8_t m = _BV(pins[i]);
    DDRC &= ~m; PORTC &= ~m; delayMicroseconds(100);
    bool bare = PINC & m;
    PORTC |= m; delayMicroseconds(100);
    bool pulled = PINC & m;
    PORTC &= ~m;
    Serial.print(" A"); Serial.print(pins[i]); Serial.print('=');
    Serial.print(bare ? 1 : 0); Serial.print('/'); Serial.print(pulled ? 1 : 0);
  }
  const uint8_t pairs[4][2] = {{0, 1}, {1, 0}, {4, 5}, {5, 4}};
  for (uint8_t p = 0; p < 4; p++) {
    sdaBit = pairs[p][0]; sclBit = pairs[p][1];
    PORTC &= ~(_BV(sdaBit) | _BV(sclBit));
    sdaHigh(); sclHigh();
    i2cRecover();
    uint8_t found[4], count = 0;
    for (uint8_t a = 0x08; a < 0x78; a++) {
      i2cStart();
      bool ack = i2cWrite(a << 1);
      i2cStop();
      if (ack) { if (count < 4) found[count] = a; count++; }
    }
    Serial.print(" A"); Serial.print(sdaBit); Serial.print('A'); Serial.print(sclBit);
    Serial.print(':');
    if (count == 0) {
      Serial.print('-');
    } else if (count >= 12) {
      Serial.print("stuck");
    } else {
      for (uint8_t i = 0; i < count && i < 4; i++) {
        if (i) Serial.print(',');
        Serial.print(found[i], HEX);
        uint8_t who = 0;
        if (regRead(found[i], 0x75, &who, 1)) {
          Serial.print('('); Serial.print(who, HEX); Serial.print(')');
        }
      }
    }
    sdaHigh(); sclHigh();
  }
  sdaBit = SDA_BIT; sclBit = SCL_BIT;
  sdaHigh(); sclHigh();
  Serial.println();
}

// The MPU-6500 (who=0x70) has its OWN accelerometer low-pass, OFF by default:
// on 2026-10-07 the always-spinning LiDAR shook the unfiltered accelerometer
// +-0.08-0.10 g at rest, which read as a jolt every sample, so the bias never
// got measured. ACCEL_CONFIG2 = 0x04: ~21 Hz, still fast enough for a bump.
// An MPU-6050 has no such register (0x1D is reserved there), so a NACK is
// fine and init goes on either way.
bool accelFilter() {
  mpuWrite(0x1D, 0x04);
  return true;
}

bool gyroInit() {
  PORTC &= ~(_BV(SDA_BIT) | _BV(SCL_BIT));   // an enabled output only ever drives LOW
  sdaHigh(); sclHigh();
  i2cRecover();
  // WHO_AM_I, up to three tries. 2026-10-19: the chip answered the scan (I)
  // at 0x68 with 0x70 while this single first read failed on every retry.
  uint8_t who = 0;
  bool read = false;
  for (whoTries = 1; whoTries <= 3 && !read; whoTries++) {
    read = mpuRead(0x75, &who, 1);
    if (!read) { i2cRecover(); delayMicroseconds(200); }
  }
  if (!read) return false;
  gyroWho = who;
  return mpuWrite(0x6B, 0x01)                 // awake, clock = gyro X PLL
      && mpuWrite(0x19, 0x09)                 // 1 kHz / 10 = 100 Hz
      && mpuWrite(0x1A, 0x03)                 // low-pass ~44 Hz
      && mpuWrite(0x1B, 0x08)                 // +-500 deg/s
      && mpuWrite(0x1C, 0x00)                 // +-2 g
      && accelFilter();
}

void gyroStartCal() {
  calSum = 0; calSumX = 0; calSumY = 0; calCount = 0;
  calMin = 32767; calMax = -32768;
}

static inline int16_t be16(const uint8_t *b) {
  return (int16_t)(((uint16_t) b[0] << 8) | b[1]);
}

void accelUpdate(int16_t rx, int16_t ry, int16_t rz, float gxDps, float gyDps) {
  float ax = rx / LSB_PER_G, ay = ry / LSB_PER_G, az = rz / LSB_PER_G;
  if (!accelSeen) { afx = ax; afy = ay; afz = az; accelSeen = true; }
  // Jolts against the filtered value BEFORE it moves: horizontal = a bump,
  // along gravity = a lift or a drop.
  float hx = ax - afx, hy = ay - afy;
  float h = sqrt(hx * hx + hy * hy);
  if (h > hPeakG) hPeakG = h;
  if (h > BUMP_G) bumps++;
  float mag = sqrt(ax * ax + ay * ay + az * az);
  // Against her OWN steady reading, not an ideal 1 g: this board reads 0.91 g
  // at rest, and against 1 g that offset (0.08-0.10) passed for a jolt, so the
  // bias measurement kept restarting for 3-6 s after every reset (2026-10-07).
  float steady = sqrt(afx * afx + afy * afy + afz * afz);
  float v = mag - steady;
  if (fabs(v) > fabs(vPeakG)) vPeakG = v;
  lastJoltG = h > fabs(v) ? h : fabs(v);
  afx += 0.2f * (ax - afx); afy += 0.2f * (ay - afy); afz += 0.2f * (az - afz);
  float fm = sqrt(afx * afx + afy * afy + afz * afz);
  if (fm > 0.3f) {
    float c = fabs(afz) / fm;
    if (c > 1.0f) c = 1.0f;
    tiltDeg = acos(c) * 57.29578f;
  }
  // A robot on a floor never rolls or pitches fast; a hand shaking it does.
  float rp = sqrt(gxDps * gxDps + gyDps * gyDps);
  shakeDps += 0.1f * (rp - shakeDps);

  unsigned long now = millis();
  if (!tilted) {
    if (tiltDeg > TILT_ON_DEG) {
      if (tiltSinceMs == 0) tiltSinceMs = now;
      if (now - tiltSinceMs >= TILT_ON_MS) {
        tilted = true;
        levelSinceMs = 0;
        allStop();                            // here, whatever the Pi is doing
      }
    } else {
      tiltSinceMs = 0;
    }
  } else {
    if (tiltDeg < TILT_OFF_DEG) {
      if (levelSinceMs == 0) levelSinceMs = now;
      if (now - levelSinceMs >= TILT_OFF_MS) { tilted = false; tiltSinceMs = 0; }
    } else {
      levelSinceMs = 0;
    }
  }
}

void gyroUpdate() {
  unsigned long nowUs = micros();
  unsigned long dtUs = nowUs - lastGyroUs;
  if (dtUs < GYRO_PERIOD_US) return;
  lastGyroUs = nowUs;
  if (!gyroOk) {
    if (millis() - lastInitMs > 2000) {
      lastInitMs = millis();
      gyroOk = gyroInit();
      gyroFailRun = 0;
    }
    return;
  }
  uint8_t b[14];
  if (!mpuRead(0x3B, b, 14)) {                // ACCEL_X..Z, TEMP, GYRO_X..Z
    gyroErrors++;
    if (++gyroFailRun >= 20) gyroOk = false;  // gone: try again every 2 s
    return;
  }
  gyroFailRun = 0;
  int16_t rawX = be16(b + 8), rawY = be16(b + 10);
  int16_t raw = be16(b + 12);
  accelUpdate(be16(b), be16(b + 2), be16(b + 4),
              ((float) rawX - biasX) / LSB_PER_DPS, ((float) rawY - biasY) / LSB_PER_DPS);
  if (calCount < CAL_SAMPLES) {               // the bias, measured standing still
    bool still = !moving && !tilted && lastJoltG < STILL_JOLT_G;
    if (still) {
      if (raw < calMin) calMin = raw;
      if (raw > calMax) calMax = raw;
    }
    if (!still || calMax - calMin > CAL_SPREAD) {
      gyroStartCal();                         // moved, held, jolted: start again
    } else {
      calSum += raw; calSumX += rawX; calSumY += rawY;
      if (++calCount == CAL_SAMPLES) {
        biasRaw = (float) calSum / CAL_SAMPLES;
        biasX = (float) calSumX / CAL_SAMPLES;
        biasY = (float) calSumY / CAL_SAMPLES;
      }
    }
    rateDps = 0.0f;
    return;
  }
  if (dtUs > GYRO_MAX_DT_US) { gyroGaps++; dtUs = GYRO_MAX_DT_US; }
  rateDps = ((float) raw - biasRaw) / LSB_PER_DPS;
  yawDeg += rateDps * (dtUs * 1e-6f);
  // Drift: follow the bias slowly while she has been stopped over a second,
  // level, unjolted and not rolling/pitching (a hand holding her shakes).
  // Errors up to 3 deg/s are pulled back (time constant ~5 s); a hand turning
  // her is far faster than that.
  if (!moving && !tilted && millis() - stillSinceMs > 1000 && fabs(rateDps) < 3.0f
      && shakeDps < 1.5f && lastJoltG < STILL_JOLT_G) {
    biasRaw += 0.002f * ((float) raw - biasRaw);
  }
}

// One line, then "*XX": the XOR of every character before the '*' (as NMEA
// does). 2026-10-07: ~1 in 20 replies reached the Pi garbled mid-turn (-24.6
// deg arriving as -2.4, a field dropped) -- the reader now refuses any line
// whose checksum does not match.
void gyroReport() {
  char buf[100];
  int n = snprintf(buf, sizeof buf, "G %ld %ld %s %lu %lu %ld %ld %ld %ld %d %lu",
                   lround(yawDeg * 100.0f), lround(rateDps * 100.0f),
                   !gyroOk ? "nogyro" : (calCount < CAL_SAMPLES ? "cal" : "ok"),
                   gyroGaps, gyroErrors, lround(tiltDeg * 10.0f),
                   lround(shakeDps * 10.0f), lround(hPeakG * 1000.0f),
                   lround(vPeakG * 1000.0f), tilted ? 1 : 0, bumps);
  uint8_t cs = 0;
  for (int i = 0; i < n && i < (int) sizeof buf && buf[i]; i++) cs ^= (uint8_t) buf[i];
  Serial.print(buf);
  Serial.print('*');
  if (cs < 16) Serial.print('0');
  Serial.println(cs, HEX);
  hPeakG = 0.0f;
  vPeakG = 0.0f;
}


void setSide(uint8_t en, uint8_t a, uint8_t b, int speed) {
  if (speed > 255)  speed = 255;
  if (speed < -255) speed = -255;
  if (speed > 0) {
    digitalWrite(a, HIGH); digitalWrite(b, LOW);
    analogWrite(en, speed);
  } else if (speed < 0) {
    digitalWrite(a, LOW);  digitalWrite(b, HIGH);
    analogWrite(en, -speed);
  } else {
    // Coast rather than brake: both inputs low leaves the bridge open, which
    // is gentler on the gearboxes than shorting the motor terminals.
    digitalWrite(a, LOW);  digitalWrite(b, LOW);
    analogWrite(en, 0);
  }
}

void allStop() {
  setSide(ENA, IN1, IN2, 0);
  setSide(ENB, IN3, IN4, 0);
  moving = false;
}

void setup() {
  // Pin modes BEFORE anything else, so the L298N sees defined levels at
  // power-up instead of floating inputs.
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  allStop();
#if PWM_31KHZ
  TCCR1B = (TCCR1B & 0b11111000) | 0x01;     // Timer1 (D9, ENA): no prescale
  TCCR2B = (TCCR2B & 0b11111000) | 0x01;     // Timer2 (D3, ENB): no prescale
#endif

  Serial.begin(115200);
  lastCommandMs = millis();
  stillSinceMs = millis();
  lastInitMs = millis();
  gyroOk = gyroInit();                        // a few ms; the bias is measured in loop()
  gyroStartCal();
  lastGyroUs = micros();
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) {
      // A bare newline is a valid keepalive: it pets the watchdog without
      // changing anything, which is how a long move stays alive.
      lastCommandMs = millis();
    } else {
      char cmd = line.charAt(0);
      if (cmd == 'P' || cmd == 'p') {
        // The identity stays exactly "FRIDAY-MOTORS 1": both drivers match it.
        Serial.println("FRIDAY-MOTORS 1");
        lastCommandMs = millis();
      } else if (cmd == 'S' || cmd == 's') {
        allStop();
        Serial.println("OK");
        lastCommandMs = millis();
      } else if (cmd == 'G' || cmd == 'g') {
        gyroReport();                         // no watchdog pet
      } else if (cmd == 'Z' || cmd == 'z') {
        yawDeg = 0.0f;
        Serial.println("OK");
      } else if (cmd == 'B' || cmd == 'b') {
        if (moving) {
          Serial.println("ERR");
        } else {
          gyroStartCal();
          Serial.println("OK");
        }
      } else if (cmd == 'V' || cmd == 'v') {
        Serial.print("FRIDAY-MOTORS 1 GYRO 7 who=0x");
        Serial.print(gyroWho, HEX);
        Serial.print(" tries=");
        Serial.print(whoTries - 1);
        Serial.print(" status=");
        Serial.println(gyroOk ? "up" : "down");
      } else if (cmd == 'I' || cmd == 'i') {
        if (moving) {
          Serial.println("ERR");                // only while stopped
        } else {
          scanReport();                         // no watchdog pet
        }
      } else if (cmd == 'W' || cmd == 'w') {
        watchdogMs = (unsigned long) line.substring(1).toInt();
        Serial.println("OK");
        lastCommandMs = millis();
      } else if (cmd == 'M' || cmd == 'm') {
        int sp = line.indexOf(' ', 1);
        if (sp < 0) {
          Serial.println("ERR");
        } else {
          int l = line.substring(1, sp).toInt();
          int r = line.substring(sp + 1).toInt();
          if (tilted) {
            allStop();                        // held: she is tipped or in a hand
            Serial.println("TILT");
          } else {
            setSide(ENA, IN1, IN2, l);
            setSide(ENB, IN3, IN4, r);
            moving = (l != 0 || r != 0);
            Serial.println("OK");
          }
          lastCommandMs = millis();
        }
      } else {
        Serial.println("ERR");
      }
    }
  }

  if (moving && watchdogMs > 0 && millis() - lastCommandMs > watchdogMs) {
    allStop();
    Serial.println("WATCHDOG");
  }

  if (moving) stillSinceMs = millis();
  gyroUpdate();
}
