#define PWM1_PIN 5
#define PWM2_PIN 6
#define SHCP_PIN 2
#define EN_PIN   7
#define DATA_PIN 8
#define STCP_PIN 4

#define TRIG_PIN 12
#define ECHO_PIN 13

const int Forward   = 163;
const int Backward  = 92;
const int Turn_Left = 106;
const int Turn_Right= 149;
const int StopMove  = 0;

const int SAFE_DISTANCE = 20; // cm

void Motor(int Dir) {
  digitalWrite(EN_PIN, LOW);
  digitalWrite(PWM1_PIN, HIGH);
  digitalWrite(PWM2_PIN, HIGH);

  digitalWrite(STCP_PIN, LOW);
  shiftOut(DATA_PIN, SHCP_PIN, MSBFIRST, Dir);
  digitalWrite(STCP_PIN, HIGH);
}

void stopRobot() { Motor(StopMove); }
void forwardRobot() { Motor(Forward); }
void backwardRobot() { Motor(Backward); }
void leftRobot() { Motor(Turn_Left); }
void rightRobot() { Motor(Turn_Right); }

float getDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  float distance = pulseIn(ECHO_PIN, HIGH) / 58.00;
  delay(10);

  return distance;
}

void executeCommand(char c) {
  switch (c) {
    case 'F': forwardRobot(); break;
    case 'B': backwardRobot(); break;
    case 'L': leftRobot(); break;
    case 'R': rightRobot(); break;
    case 'S': stopRobot(); break;
    default: stopRobot(); break;
  }
}

void setup() {
  Serial.begin(9600);

  pinMode(SHCP_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);
  pinMode(DATA_PIN, OUTPUT);
  pinMode(STCP_PIN, OUTPUT);
  pinMode(PWM1_PIN, OUTPUT);
  pinMode(PWM2_PIN, OUTPUT);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  stopRobot();
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() == 0) return;

    char c = cmd.charAt(0);
    float distance = getDistance();

    Serial.print("Command: ");
    Serial.print(c);
    Serial.print(" | Distance: ");
    Serial.print(distance);
    Serial.println(" cm");

    if (c == 'F' && distance < SAFE_DISTANCE) {
      stopRobot();
    }
    else {
      executeCommand(c);
    }
  }
}
