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
 *   anything else                  -> "ERR"
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

const uint8_t ENA = 9, IN1 = 8, IN2 = 7;     // left
const uint8_t IN3 = 6, IN4 = 5, ENB = 3;     // right

unsigned long lastCommandMs = 0;
unsigned long watchdogMs = 500;
bool moving = false;

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
}
