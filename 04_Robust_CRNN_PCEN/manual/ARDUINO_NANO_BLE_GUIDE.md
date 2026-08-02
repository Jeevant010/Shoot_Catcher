# ⚡ Phase 5: Arduino Nano 33 BLE Sense Microcontroller Deployment Guide

This guide explains how to deploy the **quantized INT8 CRNN-PCEN model** onto the **Arduino Nano 33 BLE Sense** microcontroller using TensorFlow Lite for Microcontrollers (TFLite Micro) and the built-in MP34DT05 PDM microphone.

---

## 1. Hardware Overview & Memory Budget

| Hardware Component | Arduino Nano 33 BLE Sense |
|---|---|
| **Microcontroller** | Nordic nRF52840 (ARM Cortex-M4 @ 64MHz) |
| **Flash Memory** | 1 MB |
| **RAM** | 256 KB |
| **Onboard Microphone** | MP34DT05 PDM Digital Microphone |
| **Target INT8 Model Size** | ~120 KB (Fits comfortably in Flash) |
| **Tensor Arena RAM** | ~60 KB (Fits comfortably in 256KB RAM) |

---

## 2. Software Prerequisites

1. **Arduino IDE 2.x** installed.
2. Install Core: Open **Board Manager** → install **Arduino Mbed OS Nano Boards**.
3. Install Libraries (Library Manager):
   - `Arduino_TensorFlowLite` (or `TensorFlowLite_Micro`)
   - `PDM` (built-in microphone library)

---

## 3. Preparing Model Files

1. Run `python quantize_crnn.py` in Module 04 on your PC.
2. Locate the generated C++ header file: `04_Robust_CRNN_PCEN/output/crnn_model_data.h`.
3. Copy `crnn_model_data.h` into your Arduino sketch folder.

---

## 4. Complete Arduino C++ Sketch (`GunshotDetector.ino`)

Create a new sketch in Arduino IDE and paste the code below:

```cpp
/*
 * Arduino Nano 33 BLE Sense — Edge Gunshot Detector
 * Model: Robust CRNN-PCEN (INT8 Quantized)
 */

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <PDM.h>

#include "crnn_model_data.h"

// Memory Arena for TFLite Micro
const int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Global TFLite Pointers
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Audio Settings
#define SAMPLE_RATE 16000
#define BUFFER_SIZE 512
short sample_buffer[BUFFER_SIZE];
volatile int samples_read = 0;

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sample_buffer, bytesAvailable);
  samples_read = bytesAvailable / 2;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.println("==========================================");
  Serial.println("  Arduino Nano 33 BLE Gunshot Detector   ");
  Serial.println("==========================================");

  // Set up Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load Model
  model = tflite::GetModel(g_crnn_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model schema mismatch!");
    return;
  }

  // Register Operations
  static tflite::AllOpsResolver resolver;

  // Build Interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Tensor allocation failed!");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Initialize PDM Microphone
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Failed to start PDM microphone!");
    while (1);
  }

  Serial.println("✅ Arduino Nano 33 BLE Listening...");
}

void loop() {
  if (samples_read > 0) {
    // Copy PDM audio samples to model input tensor
    for (int i = 0; i < samples_read && i < input->bytes; i++) {
      input->data.int8[i] = (int8_t)(sample_buffer[i] >> 8);
    }
    samples_read = 0;

    // Run Inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status == kTfLiteOk) {
      int8_t score_quant = output->data.int8[0];
      float score = (score_quant - output->params.zero_point) * output->params.scale;

      if (score >= 0.50f) {
        digitalWrite(LED_BUILTIN, HIGH);
        Serial.print("🚨 [GUNSHOT DETECTED] Score: ");
        Serial.println(score, 4);
        delay(500);
        digitalWrite(LED_BUILTIN, LOW);
      }
    }
  }
}
```

---

## 5. Verification on Hardware

1. Connect your Arduino Nano 33 BLE via USB.
2. Select Board: **Arduino Mbed OS Nano Boards → Arduino Nano 33 BLE**.
3. Click **Upload**.
4. Open **Serial Monitor** at 115200 baud.
5. When a gunshot transient sound occurs, the **built-in LED will flash RED/WHITE** and print `🚨 [GUNSHOT DETECTED]`.
