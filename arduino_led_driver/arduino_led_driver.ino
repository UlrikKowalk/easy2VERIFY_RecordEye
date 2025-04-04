#include <FastLED.h>

#define PIN 4             // Der Pin, an dem der WS2812B-LED-Strip angeschlossen ist
#define NUM_LEDS 237       // Anzahl der LEDs im Strip
#define LED_TYPE WS2812B  // LED-Typ (WS2812B)
#define COLOR_ORDER GRB   // Farbformat (GRB ist Standard für WS2812B)
#define LED_CENTER 119

CRGB leds[NUM_LEDS];      // Array für die LEDs

int lastLed = -1;  // Letzte LED, die eingeschaltet war (-1 bedeutet keine LED war bisher an)

void setup() {
  Serial.begin(9600);  // Serielle Kommunikation starten
  FastLED.addLeds<LED_TYPE, PIN, COLOR_ORDER>(leds, NUM_LEDS);  // WS2812B-LED-Strip initialisieren
  FastLED.clear();      // Alle LEDs zu Beginn ausschalten
  //FastLED.setBrightness(50);
  //FastLED.show();       // LEDs sofort aktualisieren
  start_show();
  FastLED.setBrightness(20);
  //FastLED.clear();
  FastLED.show();
  //delay(20);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    int ledIndex = Serial.parseInt();  // Empfangenen Integer-Wert einlesen

    if (ledIndex > 0 && ledIndex <= NUM_LEDS) {
      // Wenn der Wert im Bereich der LEDs liegt (zwischen 0 und NUM_LEDS-1)

      // Wenn eine LED vorher eingeschaltet war, schalte sie aus
      if (lastLed != -1) {
        leds[lastLed-1] = CRGB::Black;  // Setze die vorherige LED auf Schwarz (aus)
      }

      Serial.println(ledIndex);

      // Schalte die neue LED an
      leds[ledIndex-1] = CRGB::Red;  // Setze die LED auf Rot (oder eine andere Farbe)

      // Zeige die Änderungen am Strip an
      FastLED.show();

      // Merke dir, welche LED zuletzt eingeschaltet wurde
      lastLed = ledIndex;
    }
    if (ledIndex == -255) {
      FastLED.clear();
      FastLED.show();
      Serial.println("SHUTDOWN");
    }
  }
}

void start_show() {

  for (int i=0; i<LED_CENTER; i++){
    leds[LED_CENTER - i + 1 - 1] = CRGB::Black;
    leds[LED_CENTER + i - 1 - 1] = CRGB::Black;
    leds[LED_CENTER - i -1] = CRGB::Red;
    leds[LED_CENTER + i -1] = CRGB::Red;

    FastLED.show();
    delay(5);
  }
  for (int i=0; i<LED_CENTER; i++){
    leds[i-1] = CRGB::Black;
    leds[NUM_LEDS - i + 1] = CRGB::Black;
    leds[i] = CRGB::Red;
    leds[NUM_LEDS - i] = CRGB::Red;
    FastLED.show();
    delay(5);
  }
  leds[LED_CENTER] = CRGB::Black;
  FastLED.show();
}