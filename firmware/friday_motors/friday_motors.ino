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
 *   G              gyro            -> "G <yaw cdeg> <rate cdeg/s> <ok|cal|nogyro> <gaps> <errors>"
 *   Z              zero the yaw    -> "OK"
 *   B              re-measure bias -> "OK" (only while stopped, else "ERR")
 *   V              version         -> "FRIDAY-MOTORS 1 GYRO 1 who=0x.."
 *   anything else                  -> "ERR"
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
const unsigned long GYRO_PERIOD_US = 5000;   // 200 Hz
const unsigned long GYRO_MAX_DT_US = 50000;  // a longer loop stall is a "gap"
const uint16_t CAL_SAMPLES = 200;            // 1 s of stillness for the bias

bool gyroOk = false;
uint8_t gyroWho = 0;
float biasRaw = 0.0f;
long calSum = 0;
uint16_t calCount = 0;
float yawDeg = 0.0f;                         // integrated, + = counter-clockwise (chip up)
float rateDps = 0.0f;
unsigned long lastGyroUs = 0, lastInitMs = 0, stillSinceMs = 0;
unsigned long gyroGaps = 0, gyroErrors = 0;
uint8_t gyroFailRun = 0;

static inline void sdaLow()  { DDRC |=  _BV(SDA_BIT); }
static inline void sdaHigh() { DDRC &= ~_BV(SDA_BIT); }
static inline void sclLow()  { DDRC |=  _BV(SCL_BIT); }
static inline void sclHigh() { DDRC &= ~_BV(SCL_BIT); }
static inline bool sdaRead() { return PINC & _BV(SDA_BIT); }
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

bool mpuRead(uint8_t reg, uint8_t *buf, uint8_t n) {
  i2cStart();
  if (!i2cWrite(MPU << 1) || !i2cWrite(reg)) { i2cStop(); return false; }
  i2cStart();                                // repeated start
  if (!i2cWrite((MPU << 1) | 1)) { i2cStop(); return false; }
  for (uint8_t i = 0; i < n; i++) buf[i] = i2cRead(i + 1 < n);
  i2cStop();
  return true;
}

bool gyroInit() {
  PORTC &= ~(_BV(SDA_BIT) | _BV(SCL_BIT));   // an enabled output only ever drives LOW
  sdaHigh(); sclHigh();
  i2cRecover();
  uint8_t who = 0;
  if (!mpuRead(0x75, &who, 1)) return false;  // WHO_AM_I
  gyroWho = who;
  return mpuWrite(0x6B, 0x01)                 // awake, clock = gyro X PLL
      && mpuWrite(0x19, 0x04)                 // 1 kHz / 5 = 200 Hz
      && mpuWrite(0x1A, 0x03)                 // low-pass ~44 Hz
      && mpuWrite(0x1B, 0x08);                // +-500 deg/s
}

void gyroStartCal() { calSum = 0; calCount = 0; }

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
  uint8_t b[2];
  if (!mpuRead(0x47, b, 2)) {                 // GYRO_ZOUT_H, _L
    gyroErrors++;
    if (++gyroFailRun >= 20) gyroOk = false;  // gone: try again every 2 s
    return;
  }
  gyroFailRun = 0;
  int16_t raw = (int16_t)(((uint16_t) b[0] << 8) | b[1]);
  if (calCount < CAL_SAMPLES) {               // the bias, measured standing still
    if (!moving) {
      calSum += raw;
      if (++calCount == CAL_SAMPLES) biasRaw = (float) calSum / CAL_SAMPLES;
    } else {
      gyroStartCal();                         // she moved: start again
    }
    rateDps = 0.0f;
    return;
  }
  if (dtUs > GYRO_MAX_DT_US) { gyroGaps++; dtUs = GYRO_MAX_DT_US; }
  rateDps = ((float) raw - biasRaw) / LSB_PER_DPS;
  yawDeg += rateDps * (dtUs * 1e-6f);
  // Warm-up drift: follow the bias slowly while stopped for over a second and
  // all but still. A hand turning her is far faster than 1 deg/s.
  if (!moving && millis() - stillSinceMs > 1000 && fabs(rateDps) < 1.0f) {
    biasRaw += 0.002f * ((float) raw - biasRaw);
  }
}

void gyroReport() {
  Serial.print("G ");
  Serial.print(lround(yawDeg * 100.0f));
  Serial.print(' ');
  Serial.print(lround(rateDps * 100.0f));
  Serial.print(' ');
  Serial.print(!gyroOk ? "nogyro" : (calCount < CAL_SAMPLES ? "cal" : "ok"));
  Serial.print(' ');
  Serial.print(gyroGaps);
  Serial.print(' ');
  Serial.println(gyroErrors);
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
        Serial.print("FRIDAY-MOTORS 1 GYRO 1 who=0x");
        Serial.println(gyroWho, HEX);
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
          setSide(ENA, IN1, IN2, l);
          setSide(ENB, IN3, IN4, r);
          moving = (l != 0 || r != 0);
          Serial.println("OK");
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
