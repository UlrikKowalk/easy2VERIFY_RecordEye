#include <FastLED.h>

#define PIN 4             // Der Pin, an dem der WS2812B-LED-Strip angeschlossen ist
#define NUM_LEDS 30       // Anzahl der LEDs im Strip
#define LED_TYPE WS2812B  // LED-Typ (WS2812B)
#define COLOR_ORDER GRB   // Farbformat (GRB ist Standard für WS2812B)

CRGB leds[NUM_LEDS];      // Array für die LEDs

int lastLed = -1;  // Letzte LED, die eingeschaltet war (-1 bedeutet keine LED war bisher an)

void setup() {
  Serial.begin(9600);  // Serielle Kommunikation starten
  FastLED.addLeds<LED_TYPE, PIN, COLOR_ORDER>(leds, NUM_LEDS);  // WS2812B-LED-Strip initialisieren
  FastLED.clear();      // Alle LEDs zu Beginn ausschalten
  //FastLED.setBrightness(50);
  //FastLED.show();       // LEDs sofort aktualisieren
  //start_show();
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
  int center = int(NUM_LEDS/2);
  int led1 = center;
  int led2 = center;
  for (int i=0; i<center; i++){
    leds[led1-2] = CRGB::Black;
    leds[led2] = CRGB::Black;
    leds[led1-1] = CRGB::Red;
    leds[led2-1] = CRGB::Red;
    led1++;
    led2--;
    FastLED.show();
    delay(20);
  }
  for (int i=center; i>0; i--){
    leds[led1] = CRGB::Black;
    leds[led2-2] = CRGB::Black;
    leds[NUM_LEDS - led1-1] = CRGB::Red;
    leds[NUM_LEDS - led2-1] = CRGB::Red;
    led1--;
    led2++;
    FastLED.show();
    delay(20);
  }

}