# Mini README – Native TruePair Stack

## Was das ist

Dieser Stack ist ein **experimenteller nativer LLM-Pfad** für dein Setup:

- **eigene CUDA-Runtime**
- **bounded/native Checkpoint-Format**
- **PairIndex / index_v7** als Tokenraum
- **Training eines kleinen Students**
- **direkte Rückauflösung der Ausgabe** in Textfragmente

Er ist **kein fertiges normales Chat-LLM**, sondern ein eigener kompakter Inferenz-/Trainingspfad.

---

## Wichtige Dateien

### Runtime
- `llm_engine_full_blast.sh`  
  Baut die native CUDA-Runtime und startet sie.

### Training + Tokenraum
- `native_truepair_complete_fixed.sh`  
  Baut Dump + trainiert den Student im **echten PairIndex-Tokenraum**.

### Sauberes Frontend
- `native_chat_complete.sh`  
  Einfacher Wrapper für:
  - Training
  - Ask
  - Chat
  - Resolve

### Resolver
- `resolve_index_ids_true.sh`  
  Löst rohe `<123><456>`-Ausgaben über `index_v7_k18192_k28192.bin` zurück in Textfragmente auf.

### Index
- `index_v7_k18192_k28192.bin`  
  Der echte PairIndex-Tokenraum.

---

## Was aktuell funktioniert

- Training läuft durch
- Checkpoint wird erzeugt
- Runtime startet stabil
- Ausgabe kommt als PairIndex-ID-Sequenz
- Rückauflösung in Textfragmente funktioniert

Beispiel:

- RAW: `<407><407><407><407><407><407><407>`
- TEXT: `WhWhWhWhWhWhWh`

---

## Was aktuell noch nicht gut ist

Die Ausgabe ist noch oft **repetitiv** oder **semantisch schwach**.

Das heißt:
- **Pipeline funktioniert**
- **Qualität ist noch experimentell**

---

## Schnellstart

### 1. Smoke-Test-Training
```bash
./native_chat_complete.sh train-smoke
```

### 2. Eine einzelne Frage testen
```bash
./native_chat_complete.sh ask "hi"
./native_chat_complete.sh ask "wie gehts dir"
```

### 3. Interaktiv testen
```bash
./native_chat_complete.sh chat
```

### 4. Rohe IDs manuell auflösen
```bash
./native_chat_complete.sh resolve '<9187><9187><10572><10572>'
```

---

## Größeres Training

Beispiel für einen sinnvolleren Lauf:

```bash
FORCE_REBUILD_CALIB=1 CALIB_LIMIT=1500 EPOCHS=4 CANDIDATES=2 HIDDEN=128 ./native_chat_complete.sh train
```

Stärker:

```bash
FORCE_REBUILD_CALIB=1 CALIB_LIMIT=4000 EPOCHS=8 CANDIDATES=4 HIDDEN=192 ./native_chat_complete.sh train
```

---

## Woran man Erfolg erkennt

Gut ist, wenn:

- nicht nur leere Ausgabe kommt
- nicht nur `<...>` ohne Rückauflösung kommt
- die Rückauflösung **lesbare Fragmente** erzeugt
- Wiederholungen weniger werden

---

## Aktueller Status in einem Satz

**Der Stack läuft jetzt durchgehend, aber die Modellqualität muss noch weiter trainiert und verbessert werden.**
